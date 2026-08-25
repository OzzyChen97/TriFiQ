#!/usr/bin/env python3
"""Latency / throughput benchmark for the pi0.5 policy servers.

Sends synthetic observations to a running policy server and reports
round-trip latency percentiles. With --interval-ms set larger than the
service time, measures TRUE service latency instead of queueing-inflated
back-to-back latency.
"""
import argparse
import json
import time

import numpy as np
from openpi_client import websocket_client_policy


def make_obs(fmt: str, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    if fmt == "robocasa":
        return {
            "observation/state": rng.random(12).astype(np.float32) * 0.2 - 0.1,
            "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/right_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "prompt": "turn on the electric kettle",
        }
    return {
        "observation/state": rng.random(8).astype(np.float32) * 0.2 - 0.1,
        "observation/image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the black bowl and place it on the plate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--format", choices=["libero", "robocasa"], default="robocasa")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--interval-ms", type=float, default=0.0,
                        help="sleep between requests (service-time mode)")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    obs = make_obs(args.format)

    for i in range(args.warmup):
        client.infer(obs)

    latencies = []
    for i in range(args.requests):
        t0 = time.perf_counter()
        client.infer(obs)
        latencies.append(time.perf_counter() - t0)
        if args.interval_ms > 0 and i < args.requests - 1:
            time.sleep(args.interval_ms / 1000.0)

    lat = np.asarray(latencies) * 1000.0  # ms
    stats = {
        "host": args.host,
        "port": args.port,
        "format": args.format,
        "requests": args.requests,
        "warmup": args.warmup,
        "interval_ms": args.interval_ms,
        "mean_ms": float(lat.mean()),
        "std_ms": float(lat.std()),
        "p50_ms": float(np.percentile(lat, 50)),
        "p90_ms": float(np.percentile(lat, 90)),
        "p95_ms": float(np.percentile(lat, 95)),
        "min_ms": float(lat.min()),
        "max_ms": float(lat.max()),
        "throughput_req_s": float(args.requests / (lat.sum() / 1000.0)),
    }
    print(json.dumps(stats, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
