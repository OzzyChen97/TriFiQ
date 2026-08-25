#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home1/gyy/vla/QuantVLA"
ROOT="$REPO_ROOT/runs/gdsq_extension_preregistered_v1/omega_qvla_table1/download"
PID_FILE="$ROOT/download.pid"
LOG_FILE="$ROOT/download.log"
MANIFEST="$REPO_ROOT/checkpoints/omega_qvla/download_manifest.json"
PY="/home1/gyy/probe/miniforge3/envs/groot_test/bin/python"
TMUX_SESSION="omega_qvla_pack_download"
HF_MIRROR_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PARALLEL_FILES="${OMEGA_ARIA2_PARALLEL_FILES:-8}"
CONNECTIONS_PER_FILE="${OMEGA_ARIA2_CONNECTIONS_PER_FILE:-4}"
EXPECTED_PACK_BYTES=46805842563

live_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

start_download() {
    mkdir -p "$ROOT"
    if [[ -f "$MANIFEST" ]]; then
        echo "official Omega-QVLA pack manifest already exists: $MANIFEST"
        return
    fi
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then
        echo "official pack download already running: pid=$(<"$PID_FILE")"
        return
    fi
    tmux new-session -d -s "$TMUX_SESSION" \
        env HF_ENDPOINT="$HF_MIRROR_ENDPOINT" \
            OMEGA_ARIA2_PARALLEL_FILES="$PARALLEL_FILES" \
            OMEGA_ARIA2_CONNECTIONS_PER_FILE="$CONNECTIONS_PER_FILE" \
            "$0" run
    sleep 1
    echo "official pack download started in tmux session $TMUX_SESSION; log=$LOG_FILE"
}

run_download() {
    mkdir -p "$ROOT"
    echo "$$" >"$PID_FILE"
    touch "$LOG_FILE"
    printf '[%s] downloader session started backend=aria2 endpoint=%s parallel_files=%s connections_per_file=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$HF_MIRROR_ENDPOINT" "$PARALLEL_FILES" "$CONNECTIONS_PER_FILE" >>"$LOG_FILE"
    while [[ ! -f "$MANIFEST" ]]; do
        printf '[%s] starting/resuming official pack download\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"
        set +e
        HF_ENDPOINT="$HF_MIRROR_ENDPOINT" \
            "$PY" "$REPO_ROOT/scripts/tools/download_omega_qvla_packs.py" \
            --model all \
            --backend aria2 \
            --endpoint "$HF_MIRROR_ENDPOINT" \
            --parallel-files "$PARALLEL_FILES" \
            --connections-per-file "$CONNECTIONS_PER_FILE" >>"$LOG_FILE" 2>&1
        rc=$?
        set -e
        if [[ -f "$MANIFEST" ]]; then
            printf '[%s] official pack download complete\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$LOG_FILE"
            break
        fi
        printf '[%s] download attempt exited rc=%s; retrying in 15 seconds\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$LOG_FILE"
        sleep 15
    done
}

stop_download() {
    local pid="" command=""
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
    fi
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
        command="$(ps -p "$pid" -o args=)"
        if [[ "$command" == *run_omega_qvla_pack_download.sh* ]]; then
            kill "$pid"
        else
            echo "refusing to stop unrelated pid=$pid command=$command" >&2
            return 1
        fi
    fi
    echo "official pack downloader stopped"
}

status_download() {
    local pid="none" state="stopped" allocated=0 progress="0.0"
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null || live_pid; then state="running"; fi
    [[ -f "$MANIFEST" ]] && state="complete"
    echo "omega_pack_download pid=$pid state=$state manifest=$([[ -f "$MANIFEST" ]] && echo yes || echo no)"
    echo "omega_pack_files final=$(find "$REPO_ROOT/checkpoints/omega_qvla" -type f -name quantized.pt 2>/dev/null | wc -l) part=$(find "$REPO_ROOT/checkpoints/omega_qvla" -type f -name 'quantized.pt.part' 2>/dev/null | wc -l) expected=8"
    if [[ -d "$REPO_ROOT/checkpoints/omega_qvla" ]]; then
        allocated="$(du -s -B1 "$REPO_ROOT/checkpoints/omega_qvla" 2>/dev/null | awk '{print $1}')"
        progress="$(awk -v have="$allocated" -v total="$EXPECTED_PACK_BYTES" 'BEGIN { printf "%.1f", 100 * have / total }')"
    fi
    echo "omega_pack_allocated_bytes=$allocated expected_bytes=$EXPECTED_PACK_BYTES approximate_progress=${progress}%"
    du -sh "$REPO_ROOT/checkpoints/omega_qvla" 2>/dev/null || true
    [[ -f "$LOG_FILE" ]] && tail -n 8 "$LOG_FILE"
}

case "${1:-}" in
    start) start_download ;;
    run) run_download ;;
    stop) stop_download ;;
    restart) stop_download; start_download ;;
    status) status_download ;;
    *) echo "usage: $0 start | run | stop | restart | status" >&2; exit 2 ;;
esac
