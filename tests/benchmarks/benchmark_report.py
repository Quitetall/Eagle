"""benchmark_report.py — Standardized benchmark report for LML compression.

Every compression benchmark produces a BenchmarkReport dataclass with
deterministic, reproducible metrics. Reports can be serialized to JSON,
printed as tables, or compared across runs.

Usage:
    # From edf_to_lml.py:
    report = run_benchmark(input_dir, output_dir)
    report.save('benchmark_report.json')
    report.print_summary()

    # From CLI:
    python edf_to_lml.py /data/tueg/ -o /data/lml/ --benchmark --report benchmark.json
"""
from __future__ import annotations

import json
import os
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class FileMetrics:
    """Per-file compression metrics."""
    source_file: str
    source_size_bytes: int
    compressed_size_bytes: int
    cr: float                    # compression ratio
    n_channels: int
    sample_rate: float
    duration_s: float
    n_windows: int
    bit_perfect: bool            # roundtrip verified?
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    encode_speed_mbps: float = 0.0   # MB/s encoding speed
    decode_speed_mbps: float = 0.0   # MB/s decoding speed


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a compression run.

    Deterministic: same input always produces same metrics.
    Reproducible: includes enough context to replicate the benchmark.
    """
    # Identity
    report_version: str = '1.0.0'
    codec_version: str = 'LML v4.1'
    created: str = ''
    run_id: str = ''

    # Dataset info
    dataset_name: str = ''
    n_files: int = 0
    n_files_ok: int = 0
    n_files_error: int = 0
    total_channels: int = 0        # sum of channels across all files
    total_duration_hours: float = 0.0

    # Compression metrics (aggregate)
    total_source_bytes: int = 0
    total_compressed_bytes: int = 0
    aggregate_cr: float = 0.0      # total_source / total_compressed
    mean_cr: float = 0.0           # mean of per-file CRs
    median_cr: float = 0.0
    min_cr: float = 0.0
    max_cr: float = 0.0
    cr_std: float = 0.0

    # Speed metrics (aggregate)
    total_encode_time_s: float = 0.0
    total_decode_time_s: float = 0.0
    mean_encode_speed_mbps: float = 0.0
    mean_decode_speed_mbps: float = 0.0

    # Integrity
    all_bit_perfect: bool = False
    n_verified: int = 0
    n_failures: int = 0

    # Space savings
    space_saved_bytes: int = 0
    space_saved_pct: float = 0.0

    # Per-file breakdown
    files: List[FileMetrics] = field(default_factory=list)

    # Environment
    hostname: str = ''
    python_version: str = ''
    numpy_version: str = ''

    def compute_aggregates(self):
        """Recompute all aggregate metrics from per-file data."""
        if not self.files:
            return

        import numpy as np

        self.n_files = len(self.files)
        self.n_files_ok = sum(1 for f in self.files if f.cr > 0)
        self.n_files_error = self.n_files - self.n_files_ok

        crs = [f.cr for f in self.files if f.cr > 0]
        if crs:
            self.mean_cr = float(np.mean(crs))
            self.median_cr = float(np.median(crs))
            self.min_cr = float(np.min(crs))
            self.max_cr = float(np.max(crs))
            self.cr_std = float(np.std(crs))

        self.total_source_bytes = sum(f.source_size_bytes for f in self.files)
        self.total_compressed_bytes = sum(f.compressed_size_bytes for f in self.files)
        if self.total_compressed_bytes > 0:
            self.aggregate_cr = self.total_source_bytes / self.total_compressed_bytes

        self.total_channels = sum(f.n_channels for f in self.files)
        self.total_duration_hours = sum(f.duration_s for f in self.files) / 3600

        self.total_encode_time_s = sum(f.encode_time_ms for f in self.files) / 1000
        self.total_decode_time_s = sum(f.decode_time_ms for f in self.files) / 1000

        encode_speeds = [f.encode_speed_mbps for f in self.files if f.encode_speed_mbps > 0]
        decode_speeds = [f.decode_speed_mbps for f in self.files if f.decode_speed_mbps > 0]
        if encode_speeds:
            self.mean_encode_speed_mbps = float(np.mean(encode_speeds))
        if decode_speeds:
            self.mean_decode_speed_mbps = float(np.mean(decode_speeds))

        self.n_verified = sum(1 for f in self.files if f.bit_perfect)
        self.n_failures = sum(1 for f in self.files if not f.bit_perfect)
        self.all_bit_perfect = self.n_failures == 0

        self.space_saved_bytes = self.total_source_bytes - self.total_compressed_bytes
        if self.total_source_bytes > 0:
            self.space_saved_pct = self.space_saved_bytes / self.total_source_bytes * 100

    def save(self, path: str):
        """Save report as JSON."""
        d = asdict(self)
        with open(path, 'w') as f:
            json.dump(d, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> 'BenchmarkReport':
        """Load report from JSON."""
        with open(path) as f:
            d = json.load(f)
        files = [FileMetrics(**fm) for fm in d.pop('files', [])]
        report = cls(**{k: v for k, v in d.items() if k != 'files'})
        report.files = files
        return report

    def print_summary(self):
        """Print human-readable summary with aligned box drawing."""
        W = 56  # inner width (between box borders)

        def row(label: str, value: str):
            content = f'  {label:<18}{value}'
            print(f'  |{content:<{W}}|')

        def sep():
            print(f'  +{"-"*W}+')

        def header(text: str):
            print(f'  |{text:^{W}}|')

        sep()
        header('LML Compression Benchmark Report')
        sep()
        row('Codec',        self.codec_version)
        row('Dataset',      self.dataset_name)
        row('Files',        f'{self.n_files_ok:,} ok, {self.n_files_error} errors')
        row('Duration',     f'{self.total_duration_hours:.1f} hours')
        sep()
        row('Source',        f'{self.total_source_bytes/1e9:.2f} GB')
        row('Compressed',   f'{self.total_compressed_bytes/1e9:.2f} GB')
        row('Saved',        f'{self.space_saved_bytes/1e9:.2f} GB ({self.space_saved_pct:.1f}%)')
        row('Aggregate CR', f'{self.aggregate_cr:.2f}:1')
        row('Mean CR',      f'{self.mean_cr:.2f}:1 +/- {self.cr_std:.2f}')
        row('Range',        f'{self.min_cr:.2f}:1 to {self.max_cr:.2f}:1')
        sep()
        row('Bit-perfect',  f'{self.n_verified}/{self.n_files}')
        status = 'ALL PASS' if self.all_bit_perfect else f'{self.n_failures} FAILURES'
        row('Status',       status)
        if self.mean_encode_speed_mbps > 0:
            row('Encode speed',  f'{self.mean_encode_speed_mbps:.1f} MB/s')
        if self.mean_decode_speed_mbps > 0:
            row('Decode speed',  f'{self.mean_decode_speed_mbps:.1f} MB/s')
        sep()


def new_report(dataset_name: str = '', codec_version: str = 'LML v4.1') -> BenchmarkReport:
    """Create a new benchmark report with environment info."""
    import platform
    import numpy as np
    return BenchmarkReport(
        codec_version=codec_version,
        dataset_name=dataset_name,
        created=datetime.now(timezone.utc).isoformat(),
        run_id=hashlib.sha256(str(time.time()).encode()).hexdigest()[:12],
        hostname=platform.node(),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
    )


__all__ = ['FileMetrics', 'BenchmarkReport', 'new_report']
