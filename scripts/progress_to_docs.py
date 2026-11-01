"""Convierte reports/training_progress.json (y live_feed.json) en documentos JSON para la base de
datos del artifact publicado del dashboard: progress/meta, progress/final, progress/live, experiments/<id>.

    python scripts/progress_to_docs.py <out_dir>
Escribe <out_dir>/<colección>__<doc>.json y lista en stdout los que han cambiado desde la última vez.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    st = json.loads((ROOT / "reports" / "training_progress.json").read_text())
    docs = {("progress", "meta"): st["meta"]}
    if st.get("final"):
        docs[("progress", "final")] = st["final"]
    live = ROOT / "reports" / "live_feed.json"
    if live.exists():
        docs[("progress", "live")] = json.loads(live.read_text())
    for e in st["experiments"]:
        it = e.get("iters") or []
        docs[("experiments", e["id"])] = {
            "round": e["round"], "config": e["config"], "status": e["status"], "val": e.get("val"),
            "val_curve": e.get("val_curve"), "iters_J": [r[1] for r in it], "iters_fold": [r[2] for r in it]}
    hashes_f = out / "_hashes.json"
    old = json.loads(hashes_f.read_text()) if hashes_f.exists() else {}
    new, changed = {}, []
    for (c, d), body in docs.items():
        txt = json.dumps(body, separators=(",", ":"))
        h = hashlib.md5(txt.encode()).hexdigest()
        key = f"{c}__{d}"
        new[key] = h
        (out / f"{key}.json").write_text(txt)
        if old.get(key) != h:
            changed.append(key)
    hashes_f.write_text(json.dumps(new))
    # lote de escrituras para ArtifactData (if_version = última versión conocida; se asume éxito)
    vf = out / "_versions.json"
    ver = json.loads(vf.read_text()) if vf.exists() else {}
    writes = []
    for key in changed:
        c, d = key.split("__")
        w = {"op": "set", "collection": c, "doc_id": d, "file_path": str((out / f"{key}.json").resolve())}
        if key in ver:
            w["if_version"] = ver[key]
        writes.append(w)
        ver[key] = ver.get(key, 0) + 1
    vf.write_text(json.dumps(ver))
    print(json.dumps(writes))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
