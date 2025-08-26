#!/usr/bin/env python3
"""
make_manifest.py

Usage:
  python make_manifest.py /path/to/instagram_export  [--out manifest.json]

Creates a manifest of ALL .json files under the given folder (recursively),
including only: relative_path, size_bytes, modified_time.
It DOES NOT read file contents.
"""

import argparse
from pathlib import Path
import json
from datetime import datetime

def build_manifest(root: Path):
    items = []
    for p in root.rglob("*.json"):
        try:
            stat = p.stat()
            items.append({
                "relative_path": str(p.relative_to(root).as_posix()),
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
            })
        except Exception as e:
            items.append({
                "relative_path": str(p),
                "error": str(e)
            })
    # Sort by folder then name
    items.sort(key=lambda x: x.get("relative_path", ""))
    return {
        "root": str(root),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(items),
        "files": items
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str, help="Path to unzipped Instagram export folder")
    ap.add_argument("--out", type=str, default="manifest.json", help="Output JSON filename")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root path not found: {root}")

    manifest = build_manifest(root)
    out_path = Path(args.out).resolve()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest written to: {out_path}")
    print(f"Files indexed: {manifest['count']}")

if __name__ == "__main__":
    main()
