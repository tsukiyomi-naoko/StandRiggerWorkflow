#!/usr/bin/env python3
"""Pin a StandRig reference manifest to the model you just imported.

After importing the generated PSD in the StandRig browser UI, run:
    python tools/pin_manifest.py path/to/references/reference-manifest.json [http://127.0.0.1:5180]

Fills source.revision / source.semanticHash from GET /api/context. Standard
library only.
"""
import datetime, json, sys, urllib.request

man_path = sys.argv[1]
url = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:5180").rstrip("/")
ctx = json.load(urllib.request.urlopen(url + "/api/context", timeout=10))
c = ctx.get("context") or {}
if not ctx.get("ok") or "revision" not in c:
    sys.exit(f"unexpected /api/context response: {str(ctx)[:300]}")
man = json.load(open(man_path, encoding="utf-8"))
man["source"].update(revision=c["revision"], semanticHash=c.get("semanticHash"),
                     pinnedAt=datetime.datetime.now().astimezone().isoformat())
json.dump(man, open(man_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("pinned", c["revision"])
