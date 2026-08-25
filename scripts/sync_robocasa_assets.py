#!/usr/bin/env python3
"""Sync RoboCasa assets from the remote server via SFTP (paramiko).

Only downloads files that are missing locally or have a different size
(user requirement: skip identical files). Uses .part files for resume.
"""
import os
import stat
import sys
import time

import paramiko

HOST = "112.65.216.193"
PORT = 50029
USER = "wubohan"
PASSWORD = "DlibWuBoHan2027"
REMOTE_ROOT = "/data1/wubohan/robocasa/robocasa/models/assets"
LOCAL_ROOT = "/home1/gyy/vla/QuantVLA/code/robocasa/robocasa/models/assets"

# Optional: only sync the given subdirectory (for parallel workers).
SUBDIR = sys.argv[1] if len(sys.argv) > 1 else None

stats = {"scanned": 0, "skipped": 0, "downloaded": 0, "bytes": 0}


def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    sftp = client.open_sftp()
    print(f"Connected to {HOST}:{PORT}, walking {REMOTE_ROOT}", flush=True)

    start = time.time()

    def walk(remote_dir, local_dir):
        for entry in sorted(sftp.listdir_attr(remote_dir), key=lambda e: e.filename):
            rp = f"{remote_dir}/{entry.filename}"
            lp = os.path.join(local_dir, entry.filename)
            mode = entry.st_mode
            if stat.S_ISDIR(mode):
                os.makedirs(lp, exist_ok=True)
                walk(rp, lp)
            elif stat.S_ISREG(mode):
                stats["scanned"] += 1
                need = True
                if os.path.exists(lp) and os.path.getsize(lp) == entry.st_size:
                    need = False
                if need:
                    part = lp + ".part"
                    done = os.path.getsize(part) if os.path.exists(part) else 0
                    if done > entry.st_size:
                        os.remove(part)
                        done = 0
                    try:
                        with open(part, "ab" if done else "wb") as f:
                            sftp.getfo(rp, f)
                        os.replace(part, lp)
                        stats["downloaded"] += 1
                        stats["bytes"] += entry.st_size
                        print(f"[+] {rp} ({entry.st_size/1e6:.1f} MB)", flush=True)
                    except Exception as e:
                        print(f"[ERR] {rp}: {e}", flush=True)
                else:
                    stats["skipped"] += 1
                if stats["scanned"] % 200 == 0:
                    print(
                        f"progress: scanned={stats['scanned']} "
                        f"skipped={stats['skipped']} "
                        f"downloaded={stats['downloaded']} "
                        f"bytes={stats['bytes']/1e9:.2f} GB "
                        f"elapsed={time.time()-start:.0f}s",
                        flush=True,
                    )
            else:
                # symlink / other: skip
                pass

    if SUBDIR:
        walk(f"{REMOTE_ROOT}/{SUBDIR}", os.path.join(LOCAL_ROOT, SUBDIR))
    else:
        walk(REMOTE_ROOT, LOCAL_ROOT)
    sftp.close()
    client.close()
    print(
        f"DONE: scanned={stats['scanned']} skipped={stats['skipped']} "
        f"downloaded={stats['downloaded']} bytes={stats['bytes']/1e9:.2f} GB "
        f"elapsed={time.time()-start:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
