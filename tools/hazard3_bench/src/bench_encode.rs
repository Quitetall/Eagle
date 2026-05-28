//! LamQuant lossless encoder cycle bench for Hazard3 (RP2350 RISC-V).
//!
//! Boots in the Hazard3 reference testbench (Wren6991/Hazard3 +
//! tb_verilator OR tb_cxxrtl), synthesises one 21 ch x 2500 sample Q31
//! window of deterministic xorshift noise, runs the lossless encoder
//! pipeline N=8 times, and reports retired-instruction count + cycle
//! count read from the mcycle/minstret CSRs.
//!
//! Reset PC = 0x80000000. All BSS + heap + stack live in the single
//! 1 MiB testbench SRAM. Sim I/O hardware lives at 0xC000_0000 (see
//! external/Hazard3/test/sim/common/tb_cxxrtl_io.h).
//!
//! Build:
//!   cd tools/hazard3_bench && cargo build --release
//!
//! Run (Verilator):
//!   external/Hazard3/test/sim/tb_verilator/tb \
//!       --bin target/riscv32imac-unknown-none-elf/release/bench_encode \
//!       --cycles 200000000

#![no_std]
#![no_main]

extern crate alloc;

use core::arch::{asm, global_asm};

// Boot trampoline. Hazard3 reset PC is 0x80000040 — the first 64 bytes
// at 0x80000000 are a vector table (we fill with `.halt` jumps), then
// at offset 0x40 a single `j _start` hands off to riscv-rt's startup
// (BSS clear, stack init, .data copy, call `main`).
global_asm!(
    r#"
    .section .boot_vectors, "ax"
    .global _boot_vectors_start
_boot_vectors_start:
    .rept 16
        j .L_halt
    .endr

    .section .boot_trampoline, "ax"
    .global _reset_trampoline
_reset_trampoline:
    j _start

    .section .text, "ax"
.L_halt:
    wfi
    j .L_halt
    "#
);

use core::mem::MaybeUninit;
use core::ptr::{addr_of_mut, write_volatile};
use core::sync::atomic::{compiler_fence, Ordering};

use embedded_alloc::LlffHeap as Heap;
use lamquant_firmware::dsp::biquad::{NUM_CHANNELS, WINDOW_SAMPLES};
use lamquant_firmware::safety::SafetyState;
use lamquant_firmware::scheduler::{CodecMode, PipelineScheduler};
use panic_halt as _;

// ─── Hazard3 testbench MMIO (matches common/tb_cxxrtl_io.h) ─────────────────
const IO_BASE: usize = 0xC000_0000;
const IO_PRINT_CHAR: *mut u32 = IO_BASE as *mut u32;
const IO_PRINT_U32: *mut u32 = (IO_BASE + 4) as *mut u32;
const IO_EXIT: *mut u32 = (IO_BASE + 8) as *mut u32;

fn tb_putc(c: u8) {
    unsafe { write_volatile(IO_PRINT_CHAR, c as u32) }
}
fn tb_puts(s: &str) {
    for b in s.as_bytes() {
        tb_putc(*b);
    }
}
fn tb_put_u32(v: u32) {
    unsafe { write_volatile(IO_PRINT_U32, v) }
}
fn tb_exit(code: u32) -> ! {
    unsafe { write_volatile(IO_EXIT, code) }
    loop {
        unsafe { asm!("wfi", options(nomem, nostack)) };
    }
}

// ─── CSR readers (64-bit composite, RV32-safe) ─────────────────────────────
#[inline(always)]
fn read_mcycle64() -> u64 {
    loop {
        let h0: u32;
        let lo: u32;
        let h1: u32;
        unsafe {
            asm!(
                "csrr {0}, mcycleh",
                "csrr {1}, mcycle",
                "csrr {2}, mcycleh",
                out(reg) h0, out(reg) lo, out(reg) h1,
                options(nomem, nostack, preserves_flags),
            );
        }
        if h0 == h1 {
            return ((h0 as u64) << 32) | (lo as u64);
        }
    }
}

#[inline(always)]
fn read_minstret64() -> u64 {
    loop {
        let h0: u32;
        let lo: u32;
        let h1: u32;
        unsafe {
            asm!(
                "csrr {0}, minstreth",
                "csrr {1}, minstret",
                "csrr {2}, minstreth",
                out(reg) h0, out(reg) lo, out(reg) h1,
                options(nomem, nostack, preserves_flags),
            );
        }
        if h0 == h1 {
            return ((h0 as u64) << 32) | (lo as u64);
        }
    }
}

// ─── Deterministic synthetic input generator (xorshift32) ──────────────────
#[inline(always)]
fn xorshift32(s: &mut u32) -> u32 {
    let mut x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    x
}

// ─── Global allocator (codec uses transient Vec in entropy stage) ──────────
#[global_allocator]
static HEAP: Heap = Heap::empty();
const HEAP_SIZE: usize = 96 * 1024;

#[link_section = ".bss"]
static mut HEAP_MEM: [MaybeUninit<u8>; HEAP_SIZE] = [MaybeUninit::uninit(); HEAP_SIZE];

// ─── Pipeline + signal state (BSS-resident, sized once at link time) ───────
#[link_section = ".bss"]
static mut PIPELINE: MaybeUninit<PipelineScheduler> = MaybeUninit::uninit();
#[link_section = ".bss"]
static mut SAFETY: MaybeUninit<SafetyState> = MaybeUninit::uninit();
#[link_section = ".bss"]
static mut SIGNAL_A: [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] = [[0; WINDOW_SAMPLES]; NUM_CHANNELS];
#[link_section = ".bss"]
static mut SIGNAL_B: [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] = [[0; WINDOW_SAMPLES]; NUM_CHANNELS];
static ACTIVITY_MAP: [[u8; 79]; 8] = [[0; 79]; 8];

const ITERS: u32 = 8;
const CORE_CLOCK_MHZ: u64 = 150;

#[riscv_rt::entry]
fn main() -> ! {
    // Hazard3 mcycle/minstret counters are inhibited at reset:
    // mcountinhibit has bits 0+2 set (`CY` gates mcycle, `IR` gates
    // minstret). `csrci mcountinhibit, 0x5` clears those bits and
    // enables the counters. Must run before any timing read. M-mode
    // is implicit under riscv-rt's _start.
    unsafe {
        asm!("csrci mcountinhibit, 0x5", options(nomem, nostack));
    }

    unsafe {
        HEAP.init(addr_of_mut!(HEAP_MEM) as usize, HEAP_SIZE);
        PIPELINE.write(PipelineScheduler::new());
        SAFETY.write(SafetyState::default());
        (*SAFETY.as_mut_ptr()).init(0);
    }

    let pipeline: &mut PipelineScheduler = unsafe { &mut *PIPELINE.as_mut_ptr() };
    let safety: &mut SafetyState = unsafe { &mut *SAFETY.as_mut_ptr() };
    pipeline.set_codec_mode(CodecMode::Lossless);

    // Pre-fill two windows with deterministic xorshift noise BEFORE
    // the timed bracket so the encoder sees varying input on each
    // iteration (no warm-cache convergence on a single fixed input)
    // without paying the per-iteration fill cost in the cycle count.
    // Q31-scaled to ~20 effective bits (matches AFE dynamic range for
    // benign EEG).
    let signal_a: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] =
        unsafe { &mut *addr_of_mut!(SIGNAL_A) };
    let signal_b: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS] =
        unsafe { &mut *addr_of_mut!(SIGNAL_B) };
    let mut seed: u32 = 0xCAFE_BABE;
    let fill = |sig: &mut [[i32; WINDOW_SAMPLES]; NUM_CHANNELS], s: &mut u32| {
        for ch in 0..NUM_CHANNELS {
            for t in 0..WINDOW_SAMPLES {
                let v = xorshift32(s) as i32;
                sig[ch][t] = (v >> 11) & 0x000F_FFFF;
            }
        }
    };
    fill(signal_a, &mut seed);
    fill(signal_b, &mut seed);

    tb_puts("=== LamQuant lossless encoder cycle bench ===\n");
    tb_puts("target=Hazard3 RV32IMACZba_Zbb_Zbkb_Zbs (RP2350 silicon config)\n");
    tb_puts("window=21ch x 2500samp @ 250Hz (10 s per window, 52500 samples/window)\n");
    tb_puts("iters=");
    tb_put_u32(ITERS);
    tb_puts("\n");

    // Warm-up — cache lines / allocator. NOT included in the timed
    // region. Touches both buffers so both are L1-warm before bracket.
    let _ = pipeline.encode_window(signal_a, &ACTIVITY_MAP, 0, safety, 0);
    let _ = pipeline.encode_window(signal_b, &ACTIVITY_MAP, 0, safety, 0);

    let c0 = read_mcycle64();
    let i0 = read_minstret64();
    compiler_fence(Ordering::SeqCst);

    for i in 0..ITERS {
        let sig = if i & 1 == 0 { &mut *signal_a } else { &mut *signal_b };
        let r = pipeline.encode_window(sig, &ACTIVITY_MAP, 0, safety, 0);
        // Defeat LTO so the encode_window call is not folded away.
        core::hint::black_box(&r);
    }

    compiler_fence(Ordering::SeqCst);
    let c1 = read_mcycle64();
    let i1 = read_minstret64();

    let cycles_total = c1 - c0;
    let instrs_total = i1 - i0;
    let cycles_per_window = (cycles_total / ITERS as u64) as u32;
    let instrs_per_window = (instrs_total / ITERS as u64) as u32;
    let cpi_x1000 = ((cycles_total.saturating_mul(1000)) / instrs_total.max(1)) as u32;
    // wall-clock at 150 MHz, microseconds
    let window_us = (cycles_per_window as u64 / CORE_CLOCK_MHZ) as u32;
    let samples_per_window: u64 = (NUM_CHANNELS as u64) * (WINDOW_SAMPLES as u64);
    // Msa/s × 100 (fixed-point, divide by 100 host-side for the real number)
    let msa_per_s_x100 =
        ((samples_per_window.saturating_mul(100)) / window_us.max(1) as u64) as u32;

    tb_puts("cycles_per_window=");
    tb_put_u32(cycles_per_window);
    tb_puts("instrs_per_window=");
    tb_put_u32(instrs_per_window);
    tb_puts("CPI_x1000=");
    tb_put_u32(cpi_x1000);
    tb_puts("window_us@150MHz=");
    tb_put_u32(window_us);
    tb_puts("Msa_per_s_x100=");
    tb_put_u32(msa_per_s_x100);
    tb_puts("=== END BENCH ===\n");

    tb_exit(0);
}
