#!/usr/bin/env python3
"""pi0.5 websocket policy client for the RoboCerebra evaluation.

Talks to the QuantVLA pi0.5 LIBERO policy server
(pi05/openpi/scripts/serve_policy.py, fp16) or the quantized server
(pi05/openpi/scripts/serve_pi05_quant_policy.py, W4A8). The obs format
matches pi05/openpi/examples/libero/main.py: 2 cameras + 8-dim state.
"""

from collections import deque

import numpy as np
from openpi_client import websocket_client_policy


def create_policy_client(host: str, port: int):
    return websocket_client_policy.WebsocketClientPolicy(host, port)


def infer_chunk(client, observation, desc: str) -> np.ndarray:
    """Query the pi0.5 policy server for one action chunk."""
    element = {
        "observation/image": observation["full_image"],
        "observation/wrist_image": observation["wrist_image"],
        "observation/state": observation["state"],
        "prompt": desc,
    }
    return np.asarray(client.infer(element)["actions"])


def execute_policy_step(cfg, client, observation, desc, action_queue: deque) -> np.ndarray:
    """Run policy inference and pop one action from the queue.

    Mirrors the official pi0.5 LIBERO eval client: replan every
    cfg.replan_steps env steps using the first part of each chunk.
    """
    if not action_queue:
        chunk = infer_chunk(client, observation, desc)
        assert (
            len(chunk) >= cfg.replan_steps
        ), f"We want to replan every {cfg.replan_steps} steps, but policy only predicts {len(chunk)} steps."
        action_queue.extend(chunk[: cfg.replan_steps])
    return action_queue.popleft()
