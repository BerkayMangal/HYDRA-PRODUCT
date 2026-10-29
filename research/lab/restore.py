"""Restore the exact reviewed dataset and frozen research outputs, offline."""

import gzip
import hashlib
import json
from pathlib import Path

from .paths import ROOT

EVIDENCE = Path(__file__).resolve().parents[2] / "docs/research/evidence"


def restore(root=ROOT, evidence=EVIDENCE):
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    manifest = json.loads((evidence / "bundle-manifest.json").read_text())
    pending = []
    for item in manifest["files"]:
        target = (root / item["path"]).resolve()
        archive = (evidence / item["archive"]).resolve()
        if not target.is_relative_to(root) or not archive.is_relative_to(evidence):
            raise ValueError("Bundle paths must stay inside their research directories")
        packed = archive.read_bytes()
        if hashlib.sha256(packed).hexdigest() != item["archive_sha256"]:
            raise ValueError(f"Archive checksum mismatch: {archive.name}")
        payload = gzip.decompress(packed)
        if (
            len(payload) != item["bytes"]
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"Payload checksum mismatch: {item['path']}")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
                raise FileExistsError(f"Preserve the different existing file: {target}")
        else:
            pending.append((target, payload))
    # Validate the whole bundle and all existing destinations before writing.
    for target, payload in pending:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(payload)
    return {
        "verified_files": len(manifest["files"]),
        "restored_files": len(pending),
        "root": str(root),
    }


if __name__ == "__main__":
    print(json.dumps(restore(), indent=2))
