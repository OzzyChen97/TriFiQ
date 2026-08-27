#!/usr/bin/env python3
"""Audit and exactly replace the GDSQ-VLA Overleaf worktree.

Credentials are parsed locally from config.txt, passed to Git only through an
ephemeral GIT_ASKPASS environment variable, and never written to the audit.
The remote history is preserved: apply creates a normal child commit on master
instead of force-pushing.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "docs/gdsq_vla_cvpr2026"
CONFIG = REPO_ROOT / "config.txt"
AUDIT_ROOT = REPO_ROOT / "runs/gdsq_extension_preregistered_v1/overleaf_sync"
BEFORE = AUDIT_ROOT / "audit_before.json"
AFTER = AUDIT_ROOT / "audit_after.json"
EXCLUDED_PARTS = {".build", "__pycache__", ".git"}
CACHED_CLONE = Path("/tmp/quantvla-paper")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def credentials() -> tuple[str, str]:
    text = CONFIG.read_text(encoding="utf-8")
    project_matches = re.findall(r"https://www\.overleaf\.com/project/([A-Za-z0-9_-]+)", text)
    token_matches = re.findall(r"\bolp_[A-Za-z0-9_-]+\b", text)
    require(len(project_matches) == 1, "config.txt must contain exactly one Overleaf project URL")
    require(len(token_matches) == 1, "config.txt must contain exactly one Overleaf Git token")
    return project_matches[0], token_matches[0]


def source_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(SOURCE.rglob("*")):
        relative = path.relative_to(SOURCE)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        require(not path.is_symlink(), f"refusing to upload symlink: {relative}")
        if path.is_file():
            files[relative.as_posix()] = path
    require("main.tex" in files and "main.pdf" in files, "paper source/PDF missing")
    require(all(not name.startswith("config") for name in files), "credential file entered paper tree")
    return files


def tree_record(files: dict[str, Path]) -> dict[str, Any]:
    rows = [
        {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for relative, path in sorted(files.items())
    ]
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode("utf-8") + b"\0")
        digest.update(row["sha256"].encode("ascii") + b"\n")
    return {"file_count": len(rows), "tree_sha256": digest.hexdigest(), "files": rows}


class GitAccess:
    def __init__(self, project_id: str, token: str, temp_root: Path):
        self.project_id = project_id
        self.token = token
        self.remote = f"https://git@git.overleaf.com/{project_id}"
        self.askpass = temp_root / "askpass.sh"
        self.askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "  *Username*) printf '%s\\n' git ;;\n"
            "  *) printf '%s\\n' \"$OVERLEAF_GIT_TOKEN\" ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        self.askpass.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        self.env = os.environ.copy()
        self.env.update({
            "GIT_ASKPASS": str(self.askpass),
            "GIT_TERMINAL_PROMPT": "0",
            "OVERLEAF_GIT_TOKEN": token,
        })

    def sanitize(self, text: str) -> str:
        return text.replace(self.token, "<redacted-token>").replace(self.project_id, "<redacted-project>")

    def run(self, args: list[str], cwd: Path | None = None) -> str:
        command = [
            "git",
            "-c", "credential.helper=",
            "-c", "http.version=HTTP/1.1",
            "-c", "http.lowSpeedLimit=1",
            "-c", "http.lowSpeedTime=1200",
            *args,
        ]
        completed = subprocess.run(command, cwd=cwd, env=self.env, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = self.sanitize((completed.stderr or completed.stdout).strip())
            raise RuntimeError(f"Git command failed ({args[0]}): {detail}")
        return completed.stdout.strip()

    def run_input(self, args: list[str], input_text: str, cwd: Path | None = None) -> str:
        command = [
            "git",
            "-c", "credential.helper=",
            "-c", "http.version=HTTP/1.1",
            *args,
        ]
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=self.env,
            text=True,
            input=input_text,
            capture_output=True,
        )
        if completed.returncode != 0:
            detail = self.sanitize((completed.stderr or completed.stdout).strip())
            raise RuntimeError(f"Git command failed ({args[0]}): {detail}")
        return completed.stdout.strip()

    def clone(self, destination: Path) -> None:
        self.run(["clone", "--quiet", "--depth", "1", "--single-branch", self.remote, str(destination)])


def remote_record(access: GitAccess, clone: Path) -> dict[str, Any]:
    commit = access.run(["rev-parse", "HEAD"], cwd=clone)
    branch = access.run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=clone)
    names = [name for name in access.run(["ls-files"], cwd=clone).splitlines() if name]
    files = {name: clone / name for name in names}
    return {"commit": commit, "branch": branch, **tree_record(files)}


def resolve_remote_branch(access: GitAccess) -> str:
    output = access.run(["ls-remote", "--symref", access.remote, "HEAD"])
    for line in output.splitlines():
        match = re.match(r"ref:\s+refs/heads/([^\s]+)\s+HEAD$", line)
        if match:
            return match.group(1)
    if BEFORE.is_file():
        return str(json.loads(BEFORE.read_text(encoding="utf-8"))["remote"]["branch"])
    raise ValueError("unable to resolve Overleaf default branch")


def resolve_remote_head(access: GitAccess, branch: str) -> str:
    output = access.run(["ls-remote", access.remote, f"refs/heads/{branch}"])
    rows = [line.split() for line in output.splitlines() if line.strip()]
    require(len(rows) == 1 and len(rows[0]) == 2, f"unable to resolve remote head for {branch}")
    return rows[0][0]


def fetch_remote_metadata(access: GitAccess, repo: Path, branch: str) -> dict[str, Any]:
    access.run(["init", "--quiet", str(repo)])
    access.run(["remote", "add", "origin", access.remote], cwd=repo)
    access.run([
        "-c", "protocol.version=2", "fetch", "--quiet", "--depth", "1",
        "--filter=blob:none", "origin", f"refs/heads/{branch}"
    ], cwd=repo)
    commit = access.run(["rev-parse", "FETCH_HEAD"], cwd=repo)
    tree_oid = access.run(["rev-parse", "FETCH_HEAD^{tree}"], cwd=repo)
    rows = []
    output = access.run(["ls-tree", "-r", "--full-tree", "FETCH_HEAD"], cwd=repo)
    for line in output.splitlines():
        metadata, path = line.split("\t", 1)
        mode, object_type, oid = metadata.split()
        require(object_type == "blob", f"unexpected remote Git object: {object_type}")
        rows.append({"path": path, "mode": mode, "git_blob_oid": oid})
    return {
        "commit": commit,
        "branch": branch,
        "git_tree_oid": tree_oid,
        "file_count": len(rows),
        "files": rows,
        "blob_content_downloaded": False,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_remote() -> dict[str, Any]:
    project_id, token = credentials()
    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-audit-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        clone = temp_root / "remote"
        access.clone(clone)
        remote = remote_record(access, clone)
    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_pre_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": hashlib.sha256(project_id.encode("utf-8")).hexdigest(),
        "remote": remote,
        "credential_persisted": False,
        "mutation_performed": False,
    }
    write_json(BEFORE, value)
    return value


def audit_remote_metadata() -> dict[str, Any]:
    project_id, token = credentials()
    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-metadata-audit-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        branch = resolve_remote_branch(access)
        remote = fetch_remote_metadata(access, temp_root / "remote", branch)
    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_pre_replace_metadata_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": hashlib.sha256(project_id.encode("utf-8")).hexdigest(),
        "remote": remote,
        "credential_persisted": False,
        "mutation_performed": False,
    }
    write_json(BEFORE, value)
    return value


def audit_remote_head() -> dict[str, Any]:
    project_id, token = credentials()
    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-head-audit-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        branch = resolve_remote_branch(access)
        commit = resolve_remote_head(access, branch)
    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_pre_replace_head_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": hashlib.sha256(project_id.encode("utf-8")).hexdigest(),
        "remote": {
            "commit": commit,
            "branch": branch,
            "metadata_only": True,
        },
        "credential_persisted": False,
        "mutation_performed": False,
    }
    write_json(BEFORE, value)
    return value


def apply_remote() -> dict[str, Any]:
    require(BEFORE.is_file(), "run audit before apply")
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    project_id, token = credentials()
    fingerprint = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    require(before["project_id_sha256"] == fingerprint, "Overleaf project changed after audit")
    local_files = source_files()
    local_tree = tree_record(local_files)
    secret_bytes = (token,)
    for relative, path in local_files.items():
        data = path.read_bytes()
        require(all(secret.encode("utf-8") not in data for secret in secret_bytes), f"token leaked into source: {relative}")

    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-apply-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        clone = temp_root / "remote"
        access.clone(clone)
        require(access.run(["rev-parse", "HEAD"], cwd=clone) == before["remote"]["commit"], "remote HEAD changed after audit; re-audit required")
        access.run(["rm", "-r", "--ignore-unmatch", "--", "."], cwd=clone)
        for relative, source in local_files.items():
            destination = clone / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        access.run(["add", "--all"], cwd=clone)
        tracked = [name for name in access.run(["ls-files"], cwd=clone).splitlines() if name]
        require(set(tracked) == set(local_files), "staged Overleaf tree differs from local paper tree")
        changed = subprocess.run(
            ["git", "-c", "credential.helper=", "diff", "--cached", "--quiet"],
            cwd=clone,
            env=access.env,
        ).returncode != 0
        if changed:
            access.run(["-c", "user.name=Codex", "-c", "user.email=codex@local.invalid", "commit", "--quiet", "-m", "Replace project with audited GDSQ-VLA paper tree"], cwd=clone)
        new_commit = access.run(["rev-parse", "HEAD"], cwd=clone)
        access.run(["push", "--quiet", "origin", f"HEAD:{before['remote']['branch']}"], cwd=clone)

        verify_clone = temp_root / "verify"
        access.clone(verify_clone)
        verified_remote = remote_record(access, verify_clone)
        require(verified_remote["commit"] == new_commit, "fresh clone did not observe pushed commit")
        require(verified_remote["tree_sha256"] == local_tree["tree_sha256"], "fresh remote tree hash differs from local")
        require(verified_remote["file_count"] == local_tree["file_count"], "fresh remote file count differs from local")

    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_post_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": fingerprint,
        "previous_commit": before["remote"]["commit"],
        "new_commit": new_commit,
        "normal_non_force_push": True,
        "remote_tree_replaced": True,
        "credential_persisted": False,
        "verified_by_fresh_clone": True,
        "local": local_tree,
        "remote": verified_remote,
    }
    write_json(AFTER, value)
    return value


def verify_remote() -> dict[str, Any]:
    require(BEFORE.is_file(), "missing pre-replace audit")
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    project_id, token = credentials()
    fingerprint = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    require(before["project_id_sha256"] == fingerprint, "Overleaf project changed after audit")
    local_tree = tree_record(source_files())
    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-verify-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        clone = temp_root / "remote"
        access.clone(clone)
        remote = remote_record(access, clone)
    require(remote["tree_sha256"] == local_tree["tree_sha256"], "remote tree hash differs from local")
    require(remote["file_count"] == local_tree["file_count"], "remote file count differs from local")
    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_post_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": fingerprint,
        "previous_commit": before["remote"]["commit"],
        "new_commit": remote["commit"],
        "normal_non_force_push": True,
        "remote_tree_replaced": True,
        "credential_persisted": False,
        "verified_by_fresh_clone": True,
        "local": local_tree,
        "remote": remote,
    }
    write_json(AFTER, value)
    return value


def apply_remote_metadata() -> dict[str, Any]:
    require(BEFORE.is_file(), "run metadata-audit before metadata-apply")
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    project_id, token = credentials()
    fingerprint = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    require(before["project_id_sha256"] == fingerprint, "Overleaf project changed after audit")
    local_files = source_files()
    local_tree = tree_record(local_files)
    for relative, path in local_files.items():
        require(token.encode("utf-8") not in path.read_bytes(), f"token leaked into source: {relative}")

    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-metadata-apply-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        branch = str(before["remote"]["branch"])
        repo = temp_root / "repo"
        current = fetch_remote_metadata(access, repo, branch)
        require(current["commit"] == before["remote"]["commit"], "remote HEAD changed after metadata audit")
        access.run(["update-ref", f"refs/heads/{branch}", current["commit"]], cwd=repo)
        access.run(["symbolic-ref", "HEAD", f"refs/heads/{branch}"], cwd=repo)
        access.run(["read-tree", "--empty"], cwd=repo)
        for relative, source in local_files.items():
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        access.run(["add", "--all"], cwd=repo)
        tracked = [name for name in access.run(["ls-files"], cwd=repo).splitlines() if name]
        require(set(tracked) == set(local_files), "staged Overleaf tree differs from local paper tree")
        local_git_tree = access.run(["write-tree"], cwd=repo)
        access.run([
            "-c", "user.name=Codex", "-c", "user.email=codex@local.invalid",
            "commit", "--quiet", "-m", "Replace project with audited GDSQ-VLA paper tree"
        ], cwd=repo)
        new_commit = access.run(["rev-parse", "HEAD"], cwd=repo)
        access.run(["push", "--quiet", "origin", f"HEAD:{branch}"], cwd=repo)

        verify = fetch_remote_metadata(access, temp_root / "verify", branch)
        require(verify["commit"] == new_commit, "metadata fetch did not observe pushed commit")
        require(verify["git_tree_oid"] == local_git_tree, "remote Git tree differs from staged local tree")
        require({row["path"] for row in verify["files"]} == set(local_files), "remote path set differs from local")

    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_post_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": fingerprint,
        "previous_commit": before["remote"]["commit"],
        "new_commit": new_commit,
        "normal_non_force_push": True,
        "remote_tree_replaced": True,
        "credential_persisted": False,
        "verified_by_fresh_metadata_fetch": True,
        "verified_by_fresh_clone": False,
        "local": local_tree,
        "local_git_tree_oid": local_git_tree,
        "remote": verify,
    }
    write_json(AFTER, value)
    return value


def apply_remote_from_head() -> dict[str, Any]:
    """Create a normal child commit without downloading the parent's blobs.

    The audited remote HEAD is recorded as a shallow boundary.  The new commit
    object still names it as its parent, so the server applies the update only
    as a normal fast-forward; no force option is used.
    """
    require(BEFORE.is_file(), "run head-audit before head-apply")
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    project_id, token = credentials()
    fingerprint = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    require(before["project_id_sha256"] == fingerprint, "Overleaf project changed after audit")
    local_files = source_files()
    local_tree = tree_record(local_files)
    for relative, path in local_files.items():
        require(token.encode("utf-8") not in path.read_bytes(), f"token leaked into source: {relative}")

    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-head-apply-", dir="/tmp") as temporary:
        temp_root = Path(temporary)
        access = GitAccess(project_id, token, temp_root)
        branch = str(before["remote"]["branch"])
        parent = str(before["remote"]["commit"])
        require(resolve_remote_head(access, branch) == parent, "remote HEAD changed after head audit")
        repo = temp_root / "repo"
        access.run(["init", "--quiet", str(repo)])
        access.run(["remote", "add", "origin", access.remote], cwd=repo)
        for relative, source in local_files.items():
            destination = repo / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        access.run(["add", "--all"], cwd=repo)
        tracked = [name for name in access.run(["ls-files"], cwd=repo).splitlines() if name]
        require(set(tracked) == set(local_files), "staged Overleaf tree differs from local paper tree")
        local_git_tree = access.run(["write-tree"], cwd=repo)
        timestamp = int(datetime.now(timezone.utc).timestamp())
        identity = f"Codex <codex@local.invalid> {timestamp} +0000"
        commit_body = (
            f"tree {local_git_tree}\n"
            f"parent {parent}\n"
            f"author {identity}\n"
            f"committer {identity}\n\n"
            "Replace project with audited GDSQ-VLA paper tree\n"
        )
        new_commit = access.run_input(["hash-object", "-t", "commit", "-w", "--stdin"], commit_body, cwd=repo)
        git_dir = Path(access.run(["rev-parse", "--git-dir"], cwd=repo))
        if not git_dir.is_absolute():
            git_dir = repo / git_dir
        (git_dir / "shallow").write_text(parent + "\n", encoding="ascii")
        access.run(["update-ref", f"refs/heads/{branch}", new_commit], cwd=repo)
        access.run(["symbolic-ref", "HEAD", f"refs/heads/{branch}"], cwd=repo)
        access.run(["push", "--quiet", "origin", f"HEAD:{branch}"], cwd=repo)
        observed = resolve_remote_head(access, branch)
        require(observed == new_commit, "fresh ls-remote did not observe pushed commit")

    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_post_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": fingerprint,
        "previous_commit": parent,
        "new_commit": new_commit,
        "normal_non_force_push": True,
        "remote_tree_replaced": True,
        "credential_persisted": False,
        "verified_by_fresh_ls_remote": True,
        "verified_by_fresh_clone": False,
        "local": local_tree,
        "local_git_tree_oid": local_git_tree,
        "remote": {
            "commit": new_commit,
            "branch": branch,
            "git_tree_oid": local_git_tree,
            "file_count": local_tree["file_count"],
            "tree_sha256": local_tree["tree_sha256"],
            "verification_basis": "The observed remote commit is the locally hashed commit object that binds the audited parent and exact local Git tree.",
        },
    }
    write_json(AFTER, value)
    return value


def apply_remote_from_cached_clone() -> dict[str, Any]:
    """Exactly replace the project using an already fetched clean clone."""
    require(BEFORE.is_file(), "run head-audit before cached-apply")
    require((CACHED_CLONE / ".git").is_dir(), f"missing cached clone: {CACHED_CLONE}")
    before = json.loads(BEFORE.read_text(encoding="utf-8"))
    project_id, token = credentials()
    fingerprint = hashlib.sha256(project_id.encode("utf-8")).hexdigest()
    require(before["project_id_sha256"] == fingerprint, "Overleaf project changed after audit")
    local_files = source_files()
    local_tree = tree_record(local_files)
    for relative, path in local_files.items():
        require(token.encode("utf-8") not in path.read_bytes(), f"token leaked into source: {relative}")

    with tempfile.TemporaryDirectory(prefix="gdsq-overleaf-cached-apply-", dir="/tmp") as temporary:
        access = GitAccess(project_id, token, Path(temporary))
        branch = str(before["remote"]["branch"])
        parent = str(before["remote"]["commit"])
        require(resolve_remote_head(access, branch) == parent, "remote HEAD changed after head audit")
        require(access.run(["rev-parse", "HEAD"], cwd=CACHED_CLONE) == parent, "cached clone is stale")
        require(not access.run(["status", "--porcelain"], cwd=CACHED_CLONE), "cached clone is dirty")
        access.run(["rm", "-r", "--ignore-unmatch", "--", "."], cwd=CACHED_CLONE)
        for relative, source in local_files.items():
            destination = CACHED_CLONE / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        access.run(["add", "--all"], cwd=CACHED_CLONE)
        tracked = [name for name in access.run(["ls-files"], cwd=CACHED_CLONE).splitlines() if name]
        require(set(tracked) == set(local_files), "staged Overleaf tree differs from local paper tree")
        access.run([
            "-c", "user.name=Codex", "-c", "user.email=codex@local.invalid",
            "commit", "--quiet", "-m", "Replace project with audited GDSQ-VLA paper tree",
        ], cwd=CACHED_CLONE)
        new_commit = access.run(["rev-parse", "HEAD"], cwd=CACHED_CLONE)
        local_git_tree = access.run(["rev-parse", "HEAD^{tree}"], cwd=CACHED_CLONE)
        access.run(["push", "--quiet", "origin", f"HEAD:{branch}"], cwd=CACHED_CLONE)
        require(resolve_remote_head(access, branch) == new_commit, "fresh ls-remote did not observe pushed commit")

    value = {
        "schema_version": 1,
        "kind": "gdsq_overleaf_post_replace_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_id_sha256": fingerprint,
        "previous_commit": parent,
        "new_commit": new_commit,
        "normal_non_force_push": True,
        "remote_tree_replaced": True,
        "credential_persisted": False,
        "verified_by_fresh_ls_remote": True,
        "verified_by_fresh_clone": False,
        "local": local_tree,
        "local_git_tree_oid": local_git_tree,
        "remote": {
            "commit": new_commit,
            "branch": branch,
            "git_tree_oid": local_git_tree,
            "file_count": local_tree["file_count"],
            "tree_sha256": local_tree["tree_sha256"],
            "verification_basis": "Fresh ls-remote observes the commit created from the exact audited local tree in the previously fetched clean clone.",
        },
    }
    write_json(AFTER, value)
    return value


def summary(value: dict[str, Any]) -> dict[str, Any]:
    remote = value.get("remote") or {}
    local = value.get("local") or {}
    return {
        "kind": value["kind"],
        "commit": value.get("new_commit") or remote.get("commit"),
        "file_count": remote.get("file_count"),
        "tree_sha256": remote.get("tree_sha256") or local.get("tree_sha256"),
        "git_tree_oid": remote.get("git_tree_oid"),
        "verified_by_fresh_clone": value.get("verified_by_fresh_clone", False),
        "verified_by_fresh_metadata_fetch": value.get("verified_by_fresh_metadata_fetch", False),
        "credential_persisted": value.get("credential_persisted", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "audit", "metadata-audit", "head-audit", "apply", "metadata-apply",
            "head-apply", "cached-apply", "verify", "status"
        ),
    )
    args = parser.parse_args()
    if args.command == "audit":
        value = audit_remote()
    elif args.command == "metadata-audit":
        value = audit_remote_metadata()
    elif args.command == "head-audit":
        value = audit_remote_head()
    elif args.command == "apply":
        value = apply_remote()
    elif args.command == "metadata-apply":
        value = apply_remote_metadata()
    elif args.command == "head-apply":
        value = apply_remote_from_head()
    elif args.command == "cached-apply":
        value = apply_remote_from_cached_clone()
    elif args.command == "verify":
        value = verify_remote()
    else:
        value = {
            "before": json.loads(BEFORE.read_text(encoding="utf-8")) if BEFORE.is_file() else None,
            "after": json.loads(AFTER.read_text(encoding="utf-8")) if AFTER.is_file() else None,
        }
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(json.dumps(summary(value), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
