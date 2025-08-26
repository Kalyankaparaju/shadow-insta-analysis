#!/usr/bin/env python3
"""
merge_tag_suggestions.py
Safely merge tag_suggestions.json into labels_config.json.
- Keeps all existing mappings.
- Adds only new accounts/hashtags/keywords or new tags under them.
- Deduplicates tags and sorts them.
"""

import json
from pathlib import Path
import argparse

KEYS = ["account_to_tags", "hashtag_to_tags", "keyword_to_tags"]

def normalize(d):
    # case-insensitive keys for account/hashtag/keyword maps
    out = {}
    for k, v in d.items():
        out[k.lower()] = list(dict.fromkeys(v))  # dedupe tags
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="labels_config.json to update")
    ap.add_argument("--suggestions", required=True, help="tag_suggestions.json to merge in")
    ap.add_argument("--out", help="Output path. If omitted, overwrites --config in-place")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    sug_path = Path(args.suggestions)
    out_path = Path(args.out) if args.out else cfg_path

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    sug = json.loads(sug_path.read_text(encoding="utf-8"))

    # ensure keys exist
    for k in KEYS:
        cfg.setdefault(k, {})
        sug.setdefault(k, {})

    # normalize maps (lowercase keys)
    for k in KEYS:
        cfg[k] = normalize(cfg[k])
        sug[k] = normalize(sug[k])

    # merge: keep existing, add new
    for k in KEYS:
        for key, tags in sug[k].items():
            if key not in cfg[k]:
                cfg[k][key] = tags[:]
            else:
                merged = list(dict.fromkeys([*cfg[k][key], *tags]))
                cfg[k][key] = merged

    # pretty sort for readability
    for k in KEYS:
        cfg[k] = {kk: sorted(vv) for kk, vv in sorted(cfg[k].items())}

    out_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged suggestions into: {out_path}")

if __name__ == "__main__":
    main()
