#!/usr/bin/env python3
"""pi0.5 LIBERO utilities.

Replaces the openvla-oft `experiments.robot.*` helpers used by the original
RoboCerebra evaluation with equivalents matching the QuantVLA pi0.5 LIBERO
eval client (pi05/openpi/examples/libero/main.py).
"""

import math
import random
from datetime import datetime

import numpy as np
from openpi_client import image_tools

DATE_TIME = datetime.now().strftime("%m-%d_%H-%M-%S")


def set_seed_everywhere(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def quat2axisangle(quat):
    """
    Copied from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_libero_image(obs, resize_size=None):
    """agentview image, rotated 180 degrees to match the pi0.5 training preprocessing."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    if resize_size is not None:
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize_size, resize_size)
        )
    return img


def get_libero_wrist_image(obs, resize_size=None):
    """eye-in-hand image, rotated 180 degrees to match the pi0.5 training preprocessing."""
    img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if resize_size is not None:
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, resize_size, resize_size)
        )
    return img


def resize_image_for_policy(img, resize_size):
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(img, resize_size, resize_size)
    )


def get_libero_dummy_action() -> np.ndarray:
    """7-dim dummy action with open gripper (-1) used during the initial waiting steps."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
