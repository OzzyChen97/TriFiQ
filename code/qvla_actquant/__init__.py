"""QVLA/ActQuant RoboCasa365 reproduction utilities.

The package intentionally keeps calibration/packing independent from the
GR00T and OpenPI servers.  Model-specific entry points bind a frozen module
inventory, then use the shared algorithms and artifact validation here.
"""

from .core import (  # noqa: F401
    ActQuantType,
    HessianProxy,
    QVLA_BITS,
    actquant_greedy_l2_allocate,
    apply_actquant_gguf_bundle,
    apply_qvla_pack,
    canonical_hash,
    classify_target,
    compute_hsic_record,
    fisher_diagonal,
    qvla_compute_proxies,
    qvla_greedy_allocate,
    qvla_pack_model,
    sha256_file,
    target_inventory,
)
