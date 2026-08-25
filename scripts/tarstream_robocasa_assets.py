#!/usr/bin/env python3
"""Stream RoboCasa assets from the remote server as a tar archive over SSH.

Much faster than per-file SFTP for directories with many small files: the
remote runs `tar -cf - <dir>`, the stream is piped into a local `tar -xf -`
with --skip-old-files so files that are already synced (same path exists
locally) are not overwritten.

Usage:
  python tarstream_robocasa_assets.py [subdir ...]   # default: all top-level dirs
"""
import os
import subprocess
import sys
import time

import paramiko

HOST = "112.65.216.193"
PORT = 50029
USER = "wubohan"
PASSWORD = "DlibWuBoHan2027"
REMOTE_ROOT = "/data1/wubohan/robocasa/robocasa/models/assets"
LOCAL_ROOT = "/home1/gyy/vla/QuantVLA/code/robocasa/robocasa/models/assets"

SUBDIRS = sys.argv[1:] or [
    "arenas", "box_links", "fixtures", "generative_textures",
    "groot_dataset_assets", "novel_instructions", "objects", "scenes", "textures",
]


def stream_one(client, subdir: str) -> None:
    remote_dir = f"{REMOTE_ROOT}/{subdir}"
    local_dir = LOCAL_ROOT
    os.makedirs(local_dir, exist_ok=True)

    start = time.time()
    cmd = f"tar -C {REMOTE_ROOT} -cf - {subdir}"
    stdin, stdout, stderr = client.exec_command(cmd, timeout=None)
    # Use a big window so remote tar never blocks.
    stdout.channel.settimeout(300)

    extract = subprocess.Popen(
        ["tar", "-xf", "-", "-C", local_dir, "--skip-old-files"],
        stdin=subprocess.PIPE,
    )
    assert extract.stdin is not None
    total = 0
    while True:
        chunk = stdout.channel.recv(1024 * 1024)
        if not chunk:
            break
        extract.stdin.write(chunk)
        total += len(chunk)
    extract.stdin.close()
    rc = extract.wait()
    err = stderr.read().decode(errors="replace")
    if err.strip():
        print(f"[remote-tar] {err.strip()[:300]}", flush=True)
    print(
        f"DONE {subdir}: streamed {total/1e9:.2f} GB in {time.time()-start:.0f}s "
        f"({total/1e6/(time.time()-start):.1f} MB/s), extract rc={rc}",
        flush=True,
    )


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    print(f"Connected to {HOST}:{PORT}", flush=True)
    for sub in SUBDIRS:
        try:
            stream_one(client, sub)
        except Exception as e:
            print(f"[ERR] {sub}: {type(e).__name__}: {e}", flush=True)
    client.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
