#!/usr/bin/env python3
"""Generate one architecture candidate through the user-provided Images API."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def config_value(path: Path, label: str) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    for index, line in enumerate(lines[:-1]):
        if label in line.lower():
            for value in lines[index + 1 :]:
                if value:
                    return value
    raise ValueError(f"missing {label!r} entry in {path}")


def decode_image(payload: dict) -> tuple[bytes, str]:
    data = payload.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        item = data[0]
        encoded = item.get("b64_json") or item.get("b64")
        if isinstance(encoded, str):
            return base64.b64decode(encoded), "data[0].b64_json"
        remote = item.get("url")
        if isinstance(remote, str):
            with urlopen(remote, timeout=180) as response:
                return response.read(), "data[0].url"

    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message", {})
        images = message.get("images") if isinstance(message, dict) else None
        if isinstance(images, list) and images:
            image = images[0]
            if isinstance(image, dict):
                image_url = image.get("image_url", image)
                if isinstance(image_url, dict):
                    value = image_url.get("url")
                    if isinstance(value, str) and value.startswith("data:"):
                        return base64.b64decode(value.split(",", 1)[1]), "choices.image.data_url"
    raise ValueError("API response did not contain a supported image payload")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.txt")
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--size", default="2048x1152")
    args = parser.parse_args()

    key = config_value(args.config, "api key")
    base = config_value(args.config, "api url").rstrip("/") + "/"
    endpoint = urljoin(base, "v1/images/generations")
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    request_body = {
        "model": args.model,
        "prompt": prompt,
        "n": 1,
        "quality": args.quality,
        "size": args.size,
        "response_format": "b64_json",
    }
    request = Request(
        endpoint,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            raw = response.read()
            status = response.status
    except HTTPError as error:
        raw = error.read()
        # Never print the response body: some gateways echo request headers.
        raise RuntimeError(
            f"image API failed with HTTP {error.code}; response_bytes={len(raw)}"
        ) from error
    payload = json.loads(raw)
    image, response_field = decode_image(payload)
    if not image.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("generated asset is not a PNG")

    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(image)
    metadata = {
        "schema_version": 1,
        "kind": "gdsq_vla_architecture_candidate",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": {
            "scheme": urlparse(endpoint).scheme,
            "host": urlparse(endpoint).hostname,
            "path": urlparse(endpoint).path,
        },
        "http_status": status,
        "model": args.model,
        "quality": args.quality,
        "requested_size": args.size,
        "prompt": str(args.prompt.resolve().relative_to(REPO_ROOT)),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "response_field": response_field,
        "output": str(output.relative_to(REPO_ROOT)),
        "output_bytes": len(image),
        "output_sha256": sha256_bytes(image),
    }
    sidecar = output.with_suffix(output.suffix + ".json")
    sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
