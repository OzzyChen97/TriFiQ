#!/usr/bin/env python3
"""Generate architecture diagrams for the codex/d-pac-softfold branch via gpt-image-2.

Reads API credentials from config.txt (api key / api url) and renders one PNG
per prompt into the output directory.
"""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from pathlib import Path

import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.txt"
OUT_DIR = REPO_ROOT / "docs" / "improve" / "figs"
SIZE = "1536x1024"
QUALITY = "high"


def load_credentials() -> tuple[str, str]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    key_match = re.search(r"api key:\s*\n?([A-Za-z0-9_\-]+)", text)
    url_match = re.search(r"api url:\s*\n?(\S+)", text)
    if not key_match or not url_match:
        raise SystemExit("cannot parse api key / api url from config.txt")
    return key_match.group(1), url_match.group(1).strip()


def generate(api_key: str, api_url: str, prompt: str, out_path: Path) -> None:
    payload = {
        "model": "gpt-image-2",
        "prompt": prompt,
        "size": SIZE,
        "quality": QUALITY,
        "n": 1,
    }
    request = urllib.request.Request(
        api_url.rstrip("/") + "/v1/images/generations",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        body = json.loads(response.read().decode("utf-8"))
    if body.get("error"):
        raise RuntimeError(f"api error: {body['error']}")
    items = body.get("data") or []
    if not items or "b64_json" not in items[0]:
        raise RuntimeError(f"unexpected response keys: {list((items[0] or {}).keys())}")
    raw = base64.b64decode(items[0]["b64_json"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw)
    print(f"wrote {out_path} ({len(raw)} bytes)", flush=True)


PROMPTS = {
    "01_overview_dpac_softfold.png": """\
A detailed academic architecture diagram, flat vector style, pure white \
background, describing the exact pipeline of the experiment script \
"run_dpac_softfold_15x20.sh" (run id "dpac_softfold_15x20_v1") for \
Vision-Language-Action robot model quantization. Three horizontal stages \
connected by bold arrows, plus a bottom protocol strip. Crisp vector shapes, \
readable dark sans-serif labels, no photographs, no 3D render.

STAGE 1 (blue tones), title "Calibration: 9x9 SoftFold grid (currently \
running, 4 model GPUs 1/2/4/7 sharded)". Four parallel vertical lanes inside \
one container, each lane labeled: "GR00T atomic_seen", "GR00T \
composite_seen", "GR00T composite_unseen_long", "pi0.5". Each lane shows an \
input group of three small boxes stacked: "current GDSQ mask: cscka 16:1 \
W4A8 plan", "A8 activation scales", "raw static ATM/OHB factors"; an arrow \
into a middle box "SoftFold grid: gate_atm x gate_ohb = 9x9 = 81 \
configurations on 32 paired observations"; an arrow into a final pair of \
boxes "fit with D_func_v1 -> softfold_dfunc.json" and "fit with D_PAC_v1 -> \
softfold_dpac.json". A side annotation: "one-standard-error rule, then \
minimum gate amplitude; FP16 is the only teacher; no task labels used".

STAGE 2 (green tones), title "Closed-loop: 15 tasks x 20 seeds = 300 \
episodes per configuration". Left side a task matrix box listing three rows \
exactly: "atomic_seen: CloseBlenderLid, CloseFridge, CloseToasterOvenDoor, \
NavigateKitchen, OpenDrawer"; "composite_seen: DeliverStraw, KettleBoiling, \
LoadDishwasher, PrepareCoffee, WashLettuce"; "composite_unseen_long: \
WeighIngredients, WaffleReheat, WashFruitColander, MakeIceLemonade, \
ArrangeBreadBasket". Right side four configuration boxes in a row, each with \
a short subtitle: "current_gdsq (no correction, reused data)", \
"hard_selector_v8 (legacy selector, reused data)", "softfold_dfunc (NEW, \
folded)", "softfold_dpac (NEW, folded)". The two NEW boxes contain three \
small chips: "ATM folded into q_proj weight", "per-head OHB folded into \
o_proj columns", "selector_free: no runtime branch". Below the four boxes a \
deployment note: "GR00T: 100 wrapped layers, 2 replica servers per config \
(GPU 1+2 / 4+7), EGL pool 3-6; pi0.5: 80 wrapped layers, 4 servers, 5 task \
lanes".

STAGE 3 (orange tones), title "Aggregate": single box \
"aggregate_dpac_softfold_15x20.py -> table comparing 4 configurations on \
Atomic / Composite / Long success".

BOTTOM STRIP (gray) full width: "Protocol: target split, 4 denoising steps, \
replan interval 16, paired action noise sha256(task, seed, replan), official \
task horizons, success scores frozen until all configurations finish. Scope: \
D_PAC + selector-free SoftFold isolated under the current GDSQ mask; exact \
QuantVLA byte budget is a separate phase."
""",
    "02_d_pac_metric_components.png": """\
A detailed academic diagram, flat vector style, white background, showing the \
D_PAC long-horizon accumulated divergence metric for quantized robot \
Vision-Language-Action models.

Layout: a left input column with two stacked boxes "FP16 action chunks" and \
"Quantized action chunks", labeled with tensor shape "(R, H, D) over replans". \
A horizontal "time axis" runs across the middle, labeled "control time: \
replan r = 1 ... R; executed actions m = 16; chunk horizon H = 50".

Below the axis, four vertically stacked component panels, each a rounded box \
with a small icon-like glyph and a formula written in math notation:
1. "D_prefix / D_pose: cumulative prefix drift, c_n = sum e_i, weighted by \
(n/N)^2, pose via SE(3) exponential composition T_n = prod Exp(xi_i)".
2. "D_stitch: replan boundary jump difference |(AQ_{r+1,0} - AQ_{r,m-1}) - \
(AF_{r+1,0} - AF_{r,m-1})|^2".
3. "D_grip-time: soft gripper state p_n = sigma(kappa * a_grip), penalizes \
grasp events shifted earlier, later, or missed".
4. "D_overlap (pi0.5 auxiliary): forecast overlap A[16:50] vs next A[0:34]".

All four panels converge through arrows into a bottom node \
"D_PAC = Mean_s(d_s) + lambda_tail * CVaR_0.9(d_s)".

A small annotation box at the bottom right: "teacher-forced open-loop \
surrogate: replan-sequence drift; FP16-only reference". Use green accents for \
components, gray for the time axis, dark text, clean sans-serif, mathematical \
formulas rendered as compact notation, paper-figure quality.
""",
    "03_softfold_pipeline.png": """\
A professional pipeline diagram, flat vector style, white background, titled \
"SoftFold: selector-free folded compensation".

Five-stage left-to-right flow of rounded boxes connected by arrows:

Stage 1 "Calibration": box "frozen FP16 calibration buffer" split by a \
deterministic hash into two boxes "fit split (50%)" and "validation split \
(50%)", annotated "sample hash parity, no task labels".

Stage 2 "Raw factors (fit split)": boxes "ATM alpha_raw per layer" and "OHB \
beta_raw per head", annotated "RMS / std ratio from FP16 reference only".

Stage 3 "Gate search (validation split)": a 9x9 grid illustration (a small \
matrix of dots 9 columns by 9 rows) with axes labeled "gA in {0, 0.125, ..., \
1}" and "gB in {0, 0.125, ..., 1}", each grid cell evaluated by "D_PAC vs \
original FP16". Below it a box "one-standard-error rule: among statistically \
equivalent candidates pick smallest gA + gB", with a note "regularized: \
lambda1*(gA+gB) + lambdaX*gA*gB".

Stage 4 "Log-space interpolation": two formula boxes "alpha_eff = exp(gA * \
log(alpha_raw))" and "beta_eff = exp(gB * log(beta_raw))", annotated \
"g = 0 means identity, g = 1 means full ATM/OHB".

Stage 5 "Fold into weights (deployment)": boxes "ATM folded into q_proj \
weight" and "per-head OHB folded into o_proj input columns", and a final \
highlighted box "static weights, zero extra GEMMs, no runtime selector \
branch".

Use orange for gates, blue for calibration, green for deployment. Dark \
readable sans-serif labels, compact math notation, crisp paper-figure style, \
no photographs.
""",
    "04_exact_byte_budget_search.png": """\
A clean decision-flow diagram, flat vector style, white background, titled \
"Exact QuantVLA byte-budget search".

Vertical flow of rounded boxes connected by arrows:

1. Start node "Start from QuantVLA full-W4 layout" with sub-labels "GR00T \
116/116 W4" and "pi0.5 180/180 W4", annotated "byte cap B_QuantVLA computed \
with same checkpoint, same candidate scope, same rotation/scale accounting".

2. Search node "D_PAC-guided per-layer search" with sub-items "clipping \
percentile", "smooth factor", "rotation", "ATM/OHB correction" and a side \
note "CKA/CS only as cheap prescreen".

3. Decision diamond "Full W4 + SoftFold passes D_PAC vs FP16?" with two \
branches: YES arrow to "Done: deploy folded static weights (extra storage ~= \
0)"; NO arrow to "Allocate residual budget".

4. Residual node "Sensitive-layer residual budget" with sub-items "W4 main \
matrix + small sparse/outlier residual", "W4 + rank-1/2 residual for a few \
layers", sorted by "delta D_PAC / extra byte", annotated "every byte charged \
by an exact byte counter".

5. Final node "Frozen configuration -> closed-loop success evaluation only \
(Atomic, Composite, Long)".

Use blue for the start node, green for the search node, orange for the \
decision diamond, red for the residual node, gray for the final node. Clean \
sans-serif text, crisp paper-figure style, readable compact labels, no \
photographs.
""",
}


def main() -> None:
    api_key, api_url = load_credentials()
    print(f"api url: {api_url}", flush=True)
    for filename, prompt in PROMPTS.items():
        out_path = OUT_DIR / filename
        print(f"\n=== generating {filename} ===", flush=True)
        started = time.time()
        generate(api_key, api_url, prompt, out_path)
        print(f"done in {time.time() - started:.1f}s", flush=True)
    print("\nall figures written to", OUT_DIR, flush=True)


if __name__ == "__main__":
    sys.exit(main())