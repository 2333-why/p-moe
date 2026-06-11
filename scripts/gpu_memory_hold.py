#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import signal
import time
from typing import List

import torch

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "uint8": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Allocate and hold CUDA memory.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gb", type=float, default=100.0)
    parser.add_argument("--chunk_gb", type=float, default=1.0)
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="uint8")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--reserve_margin_gb", type=float, default=2.0)
    parser.add_argument("--touch", action="store_true")
    parser.add_argument("--yes_i_know", action="store_true")
    return parser.parse_args()


def gib_to_bytes(gb: float) -> int:
    return int(gb * 1024**3)


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.2f} GiB"


def main():
    args = parse_args()

    if not args.yes_i_know:
        raise SystemExit("Refusing to allocate GPU memory without --yes_i_know.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    target_bytes = gib_to_bytes(args.gb)
    margin_bytes = gib_to_bytes(args.reserve_margin_gb)
    allocatable = max(0, free_bytes - margin_bytes)

    if target_bytes > allocatable:
        raise SystemExit(
            f"Requested {format_gib(target_bytes)}, but only "
            f"{format_gib(allocatable)} is safely allocatable "
            f"(free={format_gib(free_bytes)}, margin={format_gib(margin_bytes)})."
        )

    dtype = getattr(torch, args.dtype)
    dtype_bytes = DTYPE_BYTES[args.dtype]
    chunk_bytes = min(gib_to_bytes(args.chunk_gb), target_bytes)

    chunks: List[torch.Tensor] = []
    allocated_bytes = 0
    stop = False

    def handle_signal(signum, _frame):
        nonlocal stop
        print(f"Received signal {signum}; releasing memory.")
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(
        f"Device cuda:{args.gpu}: {torch.cuda.get_device_name(args.gpu)} | "
        f"total={format_gib(total_bytes)} free={format_gib(free_bytes)}"
    )
    print(f"Allocating {format_gib(target_bytes)} using {args.dtype}.")

    try:
        while allocated_bytes < target_bytes:
            this_chunk = min(chunk_bytes, target_bytes - allocated_bytes)
            numel = math.ceil(this_chunk / dtype_bytes)
            tensor = torch.empty(numel, dtype=dtype, device=device)
            if args.touch:
                tensor.fill_(1)
            chunks.append(tensor)
            allocated_bytes += numel * dtype_bytes
            torch.cuda.synchronize(device)
            print(
                f"chunk={len(chunks):04d} requested={format_gib(allocated_bytes)} "
                f"allocated={format_gib(torch.cuda.memory_allocated(device))} "
                f"reserved={format_gib(torch.cuda.memory_reserved(device))}",
                flush=True,
            )

        start = time.time()
        while not stop:
            elapsed = time.time() - start
            if args.seconds > 0 and elapsed >= args.seconds:
                break
            print(f"holding memory... elapsed={elapsed:.0f}s", flush=True)
            time.sleep(30)

    finally:
        chunks.clear()
        torch.cuda.empty_cache()
        print("Released allocated tensors and emptied CUDA cache.")


if __name__ == "__main__":
    main()
