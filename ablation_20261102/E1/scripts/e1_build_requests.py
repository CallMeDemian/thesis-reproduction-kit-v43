"""Build the E1 request set (575 firms x 3 conditions x 2 models = 3,450) from frozen inputs.

E1-C4R and E1-RL reuse the September High C4R and C6-E requests of the same firm and model
(cell free8 / IC-b / B1 / replicate 1) byte for byte; only the request id changes.
E1-OE is the September C6-E request with the reference program replaced by candidate OE.

No provider is called. The script fails closed: any failed check raises and nothing is written.

Usage (from the kit root):
    python ablation_20261102/E1/scripts/e1_build_requests.py --kit-root . \
        --out-dir ablation_20261102/E1/requests
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HIGH_HASH = "d5d878bf50718bac65ede73df3fc88c16723870b6c90a790557ff35b063b7ba7"
NAMESPACE = "E1_REFERENCE_CONTENT_CONTROL_V1"
SHUFFLE_SEED = 20261020
SHARD_SIZE = 500
CELL = {"mode": "free8", "info": "IC-b", "budget": "B1", "replicate": 1}
MODELS = ("openai_gpt54mini", "google_gemini31flashlite")
PROVIDER = {"openai_gpt54mini": "openai", "google_gemini31flashlite": "google"}
CURRENCY = "thousand Korean won (KRW)"
V13 = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN"
S9 = "frozen/original_release/evaluation/llm_final_evaluation_20260913/stage9/HIGH_STRICT_ITT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"E1 BUILD BLOCKED: {message}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.kit_root.resolve()
    sys.path.insert(0, str(root / "src"))
    from types import SimpleNamespace

    from credit_recourse.final_release.common import canonical_hash
    from credit_recourse.final_release.prompting import pretty_json, render_policy_prompt
    from credit_recourse.final_release.contract import candidate_vector

    high = root / "frozen/original_release/llm/final_plan3" / HIGH_HASH
    report: dict = {"namespace": NAMESPACE, "high_release_hash": HIGH_HASH, "shuffle_seed": SHUFFLE_SEED,
                    "shard_size": SHARD_SIZE, "checks": {}, "inputs": {}}

    # ---- frozen contracts (hash-checked against the released design) ----
    llm_cfg = root / "frozen/evidence/llm"
    design_release = json.loads((llm_cfg / "design_release.json").read_text(encoding="utf-8"))
    prompt_path = llm_cfg / "prompt_contract.json"
    action_path = root / "frozen/original_release/rl/stage2/candidate_action_contract_v4_3.json"
    for name, path in (("prompt_contract", prompt_path), ("action_contract", action_path)):
        # The released hashes were taken on Windows (CRLF). A Linux checkout of a text file has LF,
        # so accept the file if either its bytes or its CRLF form match, and record which.
        raw = path.read_bytes()
        crlf = raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        expected = design_release["source_hashes"][name]
        got_raw, got_crlf = hashlib.sha256(raw).hexdigest(), hashlib.sha256(crlf).hexdigest()
        check(expected in (got_raw, got_crlf), f"{name} hash {got_raw} (CRLF {got_crlf}) != released {expected}")
        report["inputs"][name] = {"path": str(path.relative_to(root)).replace("\\", "/"), "released_sha256": expected,
                                  "matched_as": "raw" if got_raw == expected else "crlf"}
    design = SimpleNamespace(prompts=json.loads(prompt_path.read_text(encoding="utf-8")),
                             action_contract=json.loads(action_path.read_text(encoding="utf-8")))
    protocol = json.loads((high / "HIGH_REASONING_PROTOCOL.json").read_text(encoding="utf-8"))
    rules = protocol["one_treatment_rule"]

    # ---- cell inventory ----
    lr = pd.read_parquet(high / "logical_requests.parquet")
    cell = lr[(lr["mode"] == CELL["mode"]) & (lr["info"] == CELL["info"]) & (lr["budget"] == CELL["budget"])
              & (lr["replicate"] == CELL["replicate"])]
    for cond in ("C4", "C4R", "C6-E"):
        for model in MODELS:
            n = int(((cell["condition"] == cond) & (cell["model_key"] == model)).sum())
            check(n == 575, f"cell {cond}/{model} has {n} rows, expected 575")
    report["checks"]["cell_inventory"] = "PASS 575 x 2 models for C4, C4R, C6-E"

    # ---- own C3-E action: frozen observations vs stage9 C3-E ----
    obs = pd.read_parquet(root / V13 / "frozen_inputs/stage8/canonical_itt_observations.parquet",
                          columns=["policy_condition", "row_id", "firm_key", "reference_candidate_id"])
    c6 = obs[obs["policy_condition"] == "C6-E"]
    check(int(c6.groupby("row_id")["reference_candidate_id"].nunique().max()) == 1, "own action not constant per firm")
    own = (c6.groupby(["row_id", "firm_key"])["reference_candidate_id"].first().reset_index()
           .rename(columns={"reference_candidate_id": "own_C3E_action"}))
    s9 = pd.read_csv(root / S9 / "llm_stage9_llm_rl_comparison.csv", low_memory=False,
                     usecols=["policy", "analysis_population", "row_id", "candidate_id"])
    c3 = s9[(s9["policy"] == "C3-E") & (s9["analysis_population"] == "itt")][["row_id", "candidate_id"]]
    merged = own.merge(c3, on="row_id", how="outer", validate="one_to_one")
    check(len(merged) == 575 and merged["own_C3E_action"].eq(merged["candidate_id"]).all(),
          "own C3-E action disagrees between observations and stage9")
    own = own.set_index("row_id")
    counts = own["own_C3E_action"].value_counts().to_dict()
    check(counts.get("OE") == 232 and len(own) - counts.get("OE", 0) == 343, f"unexpected action counts {counts}")
    report["checks"]["own_action_map"] = f"PASS 575/575 agree; counts {counts}"

    # ---- September request bodies (wave 2, replicate 1) ----
    bodies: dict[str, dict] = {}
    for provider in ("openai", "google"):
        # request inputs only; the same folder also holds provider output files with the same ids
        for path in sorted((high / "provider_batches/wave2" / provider / "r1").glob("high_shard_*_attempt1.jsonl")):
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    rid = record.get("custom_id") or record.get("key")
                    bodies[rid] = record

    def body_texts(record: dict) -> tuple[str, str]:
        if "custom_id" in record:
            items = record["body"]["input"]
            return items[0]["content"][0]["text"], items[1]["content"][0]["text"]
        req = record["request"]
        return req["systemInstruction"]["parts"][0]["text"], req["contents"][0]["parts"][0]["text"]

    def set_user_text(record: dict, text: str) -> None:
        if "custom_id" in record:
            record["body"]["input"][1]["content"][0]["text"] = text
        else:
            record["request"]["contents"][0]["parts"][0]["text"] = text

    def set_id(record: dict, rid: str) -> None:
        if "custom_id" in record:
            record["custom_id"] = rid
        else:
            record["key"] = rid

    def check_settings(record: dict, model: str) -> None:
        rule = rules[model]
        if "custom_id" in record:
            body = record["body"]
            check(body["model"] == rule["exact_model_id"], "OpenAI model id drift")
            check(body["reasoning"] == {"effort": rule["reasoning"]["effort"]}, "OpenAI reasoning effort drift")
            check(body["max_output_tokens"] == rule["max_output_tokens"], "OpenAI max_output_tokens drift")
            check(set(body) == {"model", "input", "reasoning", "max_output_tokens", "store"}, "OpenAI body keys drift")
            check(record["url"] == "/v1/responses" and record["method"] == "POST", "OpenAI endpoint drift")
        else:
            gen = record["request"]["generationConfig"]
            check(gen == {"maxOutputTokens": rule["max_output_tokens"],
                          "thinkingConfig": {"thinkingLevel": rule["thinking"]["thinkingLevel"]}},
                  "Gemini generationConfig drift")
            check(set(record["request"]) == {"contents", "generationConfig", "systemInstruction"}, "Gemini keys drift")

    # ---- ledger (read-only) ----
    # immutable=1: read-only without creating -wal/-shm files next to the frozen ledger
    ledger = sqlite3.connect(f"file:{(high / 'generation_ledger.sqlite').as_posix()}?mode=ro&immutable=1", uri=True)
    ledger.row_factory = sqlite3.Row

    def ledger_payload(rid: str):
        return ledger.execute("SELECT system_text,user_text,visible_payload_sha256 FROM payloads WHERE request_id=?",
                              (rid,)).fetchone()

    def parent_text(pid: str) -> tuple[str, str]:
        lock = ledger.execute("SELECT selected_attempt_index,selected_raw_sha256 FROM logical_requests WHERE request_id=?",
                              (pid,)).fetchone()
        att = ledger.execute("SELECT raw_visible_text,raw_visible_text_sha256 FROM attempts WHERE request_id=? "
                             "AND attempt_index=?", (pid, lock["selected_attempt_index"])).fetchone()
        text = att["raw_visible_text"]
        text = text.decode("utf-8") if isinstance(text, bytes) else str(text)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        check(digest == att["raw_visible_text_sha256"] == lock["selected_raw_sha256"], f"parent hash drift {pid}")
        check(len(text) > 0, f"empty parent draft {pid}")
        return text, digest

    firms = pd.read_parquet(high / "firm_payload_source.parquet").set_index("firm_key")
    oe_json = pretty_json(candidate_vector(design, "OE"))

    rows: list[dict] = []
    records: dict[str, dict] = {}
    n_rerender = n_placebo = n_swap = 0
    for model in MODELS:
        sub = cell[cell["model_key"] == model]
        by = {(c, int(r)): x for c, r, x in zip(sub["condition"], sub["row_id"], sub.to_dict("records"))}
        for row_id in range(575):
            c4r, c6e = by[("C4R", row_id)], by[("C6-E", row_id)]
            check(c4r["parent_request_id"] == c6e["parent_request_id"] == by[("C4", row_id)]["request_id"],
                  f"parent link drift {model}/{row_id}")
            firm_key = c6e["firm_key"]
            action = own.loc[row_id, "own_C3E_action"]
            check(own.loc[row_id, "firm_key"] == firm_key, f"firm key drift {row_id}")
            ptext, phash = parent_text(c6e["parent_request_id"])
            firm = firms.loc[firm_key].to_dict()
            for cond, src, ref in (("E1-C4R", c4r, None), ("E1-RL", c6e, action), ("E1-OE", c6e, "OE")):
                sept = bodies.get(src["request_id"])
                check(sept is not None, f"September body missing for {src['request_id']}")
                pay = ledger_payload(src["request_id"])
                s_sys, s_user = body_texts(sept)
                check((s_sys, s_user) == (pay["system_text"], pay["user_text"]), f"body/ledger text drift {src['request_id']}")
                check(canonical_hash({"system": s_sys, "user": s_user}) == pay["visible_payload_sha256"],
                      f"visible hash drift {src['request_id']}")
                sept_cond = "C4R" if cond == "E1-C4R" else "C6-E"
                rendered = render_policy_prompt(design, row=firm, condition=sept_cond, mode=CELL["mode"], info=CELL["info"],
                                                budget=CELL["budget"], currency_unit=CURRENCY, raw_parent_text=ptext,
                                                reference_candidate_id=ref)
                if cond in ("E1-C4R", "E1-RL"):
                    check(rendered.visible_payload_sha256 == pay["visible_payload_sha256"],
                          f"re-render mismatch {cond} {model}/{row_id}")
                    n_rerender += 1
                    user_text = s_user
                else:
                    user_text = rendered.user
                    own_json = pretty_json(candidate_vector(design, action))
                    check(s_user.count(own_json) == 1, f"reference JSON not unique {model}/{row_id}")
                    check(s_user.replace(own_json, oe_json) == user_text, f"E1-OE differs beyond reference {model}/{row_id}")
                    if action == "OE":
                        check(user_text == s_user, f"placebo prompt not identical {model}/{row_id}")
                        n_placebo += 1
                    else:
                        check(user_text != s_user, f"E1-OE equals E1-RL for non-OE firm {model}/{row_id}")
                        n_swap += 1
                rid = canonical_hash({"namespace": NAMESPACE, "high_release_hash": HIGH_HASH, "model_key": model,
                                      "firm_key": firm_key, "e1_condition": cond})
                record = copy.deepcopy(sept)
                set_id(record, rid)
                set_user_text(record, user_text)
                check_settings(record, model)
                records[rid] = record
                rows.append({
                    "e1_request_id": rid, "provider": PROVIDER[model], "model_key": model,
                    "exact_model_id": rules[model]["exact_model_id"], "row_id": row_id, "firm_key": firm_key,
                    "e1_condition": cond, "reference_candidate_id": ref if ref is not None else "",
                    "own_C3E_action": action, "source_request_id": src["request_id"],
                    "parent_request_id": src["parent_request_id"], "parent_raw_sha256": phash,
                    "visible_payload_sha256": canonical_hash({"system": s_sys, "user": user_text}),
                    "provider_request_body_sha256": canonical_hash(record),
                })
    ledger.close()
    check(len(rows) == 3450 and len(records) == 3450, "request count is not 3,450")
    check(n_rerender == 2300 and n_placebo == 464 and n_swap == 686, f"check counts {n_rerender}/{n_placebo}/{n_swap}")
    report["checks"]["september_bodies_vs_ledger"] = "PASS 2,300 bodies equal ledger payloads; visible hashes recomputed"
    report["checks"]["re_render"] = "PASS 2,300 C4R/C6-E prompts re-rendered identically"
    report["checks"]["e1_oe"] = "PASS differs only in the reference JSON; 464 placebo prompts identical to E1-RL"
    report["checks"]["settings"] = "PASS every body matches HIGH_REASONING_PROTOCOL.json"

    # ---- order and shards ----
    manifest = pd.DataFrame(rows).sort_values(["provider", "model_key", "row_id", "e1_condition"]).reset_index(drop=True)
    parts = []
    for provider, group in manifest.groupby("provider", sort=True):
        group = group.reset_index(drop=True)
        order = np.random.Generator(np.random.PCG64(SHUFFLE_SEED)).permutation(len(group))
        group = group.iloc[order].reset_index(drop=True)
        group["position"] = np.arange(len(group))
        group["shard"] = group["position"] // SHARD_SIZE + 1
        parts.append(group)
    manifest = pd.concat(parts, ignore_index=True)
    for (provider, shard), group in manifest.groupby(["provider", "shard"]):
        check(set(group["e1_condition"]) == {"E1-C4R", "E1-RL", "E1-OE"}, f"shard {provider}/{shard} lacks a condition")

    out = args.out_dir.resolve()
    (out / "batches").mkdir(parents=True, exist_ok=True)
    shard_rows = []
    for (provider, shard), group in manifest.groupby(["provider", "shard"], sort=True):
        path = out / "batches" / f"{provider}_e1_shard_{int(shard):05d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for rid in group["e1_request_id"]:
                handle.write(json.dumps(records[rid], ensure_ascii=False, sort_keys=True) + "\n")
        shard_rows.append({"provider": provider, "shard": int(shard), "requests": int(len(group)),
                           "file": path.name, "sha256": sha256_file(path)})
    manifest_path = out / "E1_request_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    report["shards"] = shard_rows
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["counts"] = {"requests": 3450, "by_condition": manifest["e1_condition"].value_counts().to_dict(),
                        "own_action": counts}
    if (root / ".git").exists():
        import subprocess
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "frozen"],
                               capture_output=True, text=True, check=True).stdout.strip()
        check(dirty == "", f"frozen/ changed or has untracked files:\n{dirty}")
        report["checks"]["frozen_untouched"] = "PASS git status --porcelain -- frozen is empty"
    (out / "E1_build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "manifest_sha256": report["manifest_sha256"], "shards": shard_rows}, indent=2))


if __name__ == "__main__":
    main()
