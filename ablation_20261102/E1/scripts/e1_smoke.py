"""E1 smoke test: 2 firms x 3 conditions x 2 models, synchronous, exact request bodies.

Checks access, that the text parses (structure only) and that the returned model equals the expected
version. Responses are saved under E1/smoke/ and are never scored or analysed. Prints no action values.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_providers as prov  # noqa: E402
from e1_process import EXPECTED_MODEL, load_processing_context, process_line  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--only", choices=["openai", "google"], help="repeat the calls of one provider only (after a transport fix)")
    args = ap.parse_args()
    root = args.kit_root.resolve()
    e1 = root / "ablation_20261102/E1"
    smoke = e1 / "smoke"
    smoke.mkdir(parents=True, exist_ok=True)
    prov.load_keys()
    man = pd.read_csv(e1 / "requests/E1_request_manifest.csv", dtype=str)
    non_oe = man[man.own_C3E_action != "OE"].row_id.astype(int).min()
    oe = man[man.own_C3E_action == "OE"].row_id.astype(int).min()
    chosen = man[man.row_id.astype(int).isin([non_oe, oe])]
    assert len(chosen) == 12, len(chosen)
    if args.only:
        chosen = chosen[chosen.provider == args.only]
    want = set(chosen.e1_request_id)
    records: dict[str, dict] = {}
    for path in sorted((e1 / "requests/batches").glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for text in handle:
                rec = json.loads(text)
                rid = rec.get("custom_id") or rec.get("key")
                if rid in want:
                    records[rid] = rec
    assert set(records) == want
    design = load_processing_context(root)
    meta = chosen.set_index("e1_request_id")

    def call(rid: str) -> dict:
        provider = meta.loc[rid, "provider"]
        t0 = time.time()
        try:
            line = prov.sync_openai(records[rid]) if provider == "openai" else prov.sync_google(records[rid], EXPECTED_MODEL["google"])
            err = None
        except Exception as exc:  # noqa: BLE001
            line, err = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        (smoke / f"smoke_{rid[:12]}.json").write_text(json.dumps({"request_id": rid, "line": line, "error": err}, ensure_ascii=False) + "\n", encoding="utf-8")
        row = {"provider": provider, "condition": meta.loc[rid, "e1_condition"], "row_id": int(meta.loc[rid, "row_id"]),
               "seconds": round(time.time() - t0, 1), "error": err}
        if line is not None:
            res = process_line(provider, line, design)
            row.update(outcome=res["outcome"], returned_model=res["returned_model"], model_ok=res["returned_model_ok"],
                       strict_usable=res["strict_usable"], completion=res.get("completion_kind"), codes=res["error_codes"])
            try:
                usage = (line["response"]["body"].get("usage") if provider == "openai" else line["response"].get("usageMetadata"))
                row["usage"] = usage
            except Exception:  # noqa: BLE001
                pass
        return row

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(call, sorted(want)))
    (smoke / "smoke_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    ok = all(r["error"] is None and r.get("outcome") == "DELIVERED" and r.get("model_ok") is True for r in rows)
    print(json.dumps([{k: v for k, v in r.items() if k != "usage"} for r in rows], ensure_ascii=False, indent=1, default=str))
    print("SMOKE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
