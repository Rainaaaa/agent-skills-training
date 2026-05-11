"""Console + file logging setup shared by every training phase."""

import logging
import os
import sys
from pathlib import Path


def configure_logging(output_dir: Path, log_level: str = "INFO") -> Path:
    """Set up root logger with stdout + train.log handlers.

    In DDP runs (WORLD_SIZE > 1) only rank 0 writes the shared `train.log`;
    other ranks still log to stdout for SLURM capture. The rank-prefixed
    formatter makes interleaved stdout readable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    root = logging.getLogger()
    root.setLevel(log_level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt_str = (
        f"%(asctime)s [rank{rank}] %(levelname)s %(name)s %(message)s"
        if world_size > 1
        else "%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    fmt = logging.Formatter(fmt_str)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    if rank == 0:
        file = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file.setFormatter(fmt)
        root.addHandler(file)
    return log_path
