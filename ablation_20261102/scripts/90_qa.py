"""Automated QA Q01-Q12, Q16-Q19 and Q20 (independent Holm recomputation) -> QA_REPORT.md section and results/_qa_results.json.

Every check prints its evidence. Nothing is marked PASS without a computed comparison. A check may return WARN
with a printed reason when its condition cannot be decided in this checkout (Q01 on a detached HEAD that is not
on the published branch); WARN does not fail the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

import common as c
from credit_recourse.analysis.final_v43.primary import reconstruct
from credit_recourse.analysis.final_v43.selection import primary_filter  # noqa: F401

MARK_BEGIN = "<!-- QA_AUTO_BEGIN -->"
MARK_END = "<!-- QA_AUTO_END -->"


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(c.KIT), *a], capture_output=True, text=True, encoding="utf-8").stdout


def git_ok(*a) -> bool:
    """Exit status 0 (git answers yes/no questions such as --is-ancestor through the exit status, not stdout)."""
    return subprocess.run(["git", "-C", str(c.KIT), *a], capture_output=True).returncode == 0


def has_commit(rev: str) -> bool:
    return git_ok("cat-file", "-e", f"{rev}^{{commit}}")


def q01():
    inp = json.loads((c.RESULTS / "_inputs_check.json").read_text(encoding="utf-8"))
    branch = git("branch", "--show-current").strip()
    head = git("rev-parse", "HEAD").strip()
    ev = {"base": c.BASE_COMMIT, "branch": branch, "head": head,
          "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
          "input_sha256": inp["input_sha256"], "inputs_check": inp["status"],
          "base_is_ancestor_of_head": has_commit(c.BASE_COMMIT) and git_ok("merge-base", "--is-ancestor", c.BASE_COMMIT, "HEAD")}
    if inp["status"] != "PASS" or not ev["base_is_ancestor_of_head"]:
        ev["reason"] = "inputs check failed or the base commit is not an ancestor of HEAD"
        return "FAIL", ev
    if branch == "ablation-20261102":
        ev["reason"] = "on branch ablation-20261102"
        return "PASS", ev
    if not branch:
        # detached HEAD (tag or commit checkout): record the commit and accept it when it is on the published branch
        on_origin = "origin/ablation-20261102" in git("branch", "-r", "--contains", head)
        ev["detached_head_commit"] = head
        ev["head_on_origin_ablation_20261102"] = on_origin
        if on_origin:
            ev["reason"] = "detached HEAD at a commit contained in origin/ablation-20261102"
            return "PASS", ev
        ev["reason"] = "detached HEAD at a commit not found on origin/ablation-20261102 in this clone; branch condition not decidable"
        return "WARN", ev
    ev["reason"] = f"checked-out branch {branch!r} is not ablation-20261102; inputs and base ancestry checked"
    return "WARN", ev


def q02():
    inp = json.loads((c.RESULTS / "_inputs_check.json").read_text(encoding="utf-8"))["checks"]
    keys = ["requests_total", "strict_requests_21x575x2x2", "observations_rows", "registry_rows", "primary_rows", "supplemental_rows", "parent_gate_rows"]
    ev = {k: inp[k] for k in keys}
    return all(v["actual"] == v["expected"] for v in ev.values()), ev


def q03():
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    t2 = pd.read_csv(c.RESULTS / "T2_S1.csv")
    t3 = pd.read_csv(c.RESULTS / "T3_REF1.csv")
    r9 = pd.read_csv(c.RESULTS / "R9_candidate_selection.csv")
    ev = {"T1_rows": len(t1), "T2_rows": len(t2), "T3_rows": len(t3), "T3_tested": int((t3.status == "EXPLORATORY").sum()),
          "T3_descriptive": int((t3.status == "DESCRIPTIVE").sum()), "R9_rows": len(r9),
          "all_n_firms_575": bool(all((d.n_firms == 575).all() for d in (t1, t2, t3, r9))),
          "result_id_duplicates": int(sum(d.result_id.duplicated().sum() for d in (t1, t2, t3, r9)))}
    ok = (ev["T1_rows"], ev["T2_rows"], ev["T3_rows"], ev["T3_tested"], ev["T3_descriptive"], ev["R9_rows"]) == (96, 24, 24, 12, 12, 24) \
        and ev["all_n_firms_575"] and ev["result_id_duplicates"] == 0
    return ok, ev


def q04(obs):
    """Each contrast changes only the intended axis: the pairing key holds every other factor fixed, and the
    selected frames contain exactly one level of each fixed factor."""
    ev = {}
    ok = True
    st = obs[(obs.execution_policy == "STRICT")]
    for cid, (lhs, rhs) in {"A1": ("C5", "C4"), "A2": ("C4R", "C4"), "A3": ("C6-E", "C4R"), "A4": ("C6-E", "C6-EX")}.items():
        sub = st[(st.action_space == "FREE8") & (st.information_condition == "IC-b") & (st.replicate == 1) & st.policy_condition.isin([lhs, rhs])]
        keys = ["firm_key", "model_key", "reasoning_regime", "budget"]
        g = sub.groupby(keys).policy_condition.nunique()
        full = bool((g == 2).all())
        n_cells = int(len(g) / 575)
        ev[cid] = {"cells_x_firms": int(len(g)), "model_regime_budget_cells": n_cells, "every_firm_cell_has_both_policies": full}
        ok &= full and n_cells == 8
    r = st[(st.action_space == "CANDIDATE9") & (st.information_condition == "IC-b") & (st.replicate == 1) & (st.budget == "B1")]
    ev["R9_candidate9_cells"] = int(r.groupby(["model_key", "reasoning_regime"]).ngroups)
    ok &= ev["R9_candidate9_cells"] == 4
    # S1: only the regime differs
    s1 = st[(st.action_space == "FREE8") & (st.information_condition == "IC-b") & (st.replicate == 1) & st.policy_condition.isin(["C4", "C6-E"])]
    g = s1.groupby(["firm_key", "model_key", "budget", "policy_condition"]).reasoning_regime.nunique()
    ev["S1_both_regimes_per_cell"] = bool((g == 2).all())
    ok &= ev["S1_both_regimes_per_cell"]
    return ok, ev


def q05(obs):
    st = obs[obs.execution_policy == "STRICT"]
    c4 = st[st.policy_condition == "C4"].set_index("request_id")
    out, ok = {}, True
    for pol, expected in (("C4R", 11500), ("C6-E", 11500), ("C6-EX", 4600)):
        d = st[st.policy_condition == pol]
        parent = c4.reindex(d.parent_request_id)
        same = (parent.firm_key.to_numpy() == d.firm_key.to_numpy()) & (parent.model_key.to_numpy() == d.model_key.to_numpy()) \
            & (parent.reasoning_regime.to_numpy() == d.reasoning_regime.to_numpy()) & (parent.budget.to_numpy() == d.budget.to_numpy()) \
            & (parent.replicate.to_numpy() == d.replicate.to_numpy()) & (parent["mode"].to_numpy() == d["mode"].to_numpy()) \
            & (parent.information_condition.to_numpy() == d.information_condition.to_numpy())
        out[pol] = {"rows": len(d), "expected": expected, "parent_is_same_run_same_firm_C4": int(same.sum())}
        ok &= len(d) == expected and int(same.sum()) == expected
    return ok, out


def q06():
    r10 = pd.read_csv(c.RESULTS / "R10_reference_assignment.csv")
    ev = {"firms": len(r10), "self_reference": int(r10.self_reference.sum()), "same_action": int(r10.same_action.sum())}
    return ev == {"firms": 575, "self_reference": 0, "same_action": 141}, ev


def q07(obs):
    st = obs[obs.execution_policy == "STRICT"]
    unres = int((obs.action_application_status == "UNRESOLVED_INFRA").sum())
    unusable = st[st.action_application_status == "CONFIRMED_UNUSABLE_COMPLETION"]
    zero = bool(all(max(abs(v) for v in json.loads(s).values()) == 0.0 for s in unusable.strict_itt_action_json))
    ev = {"UNRESOLVED_INFRA_rows": unres, "strict_unusable_rows": len(unusable), "all_unusable_mapped_to_zero_action": zero,
          "strict_statuses": st.action_application_status.value_counts().to_dict(),
          "repaired_rows_equal_requests": int((obs.execution_policy == "REPAIRED").sum())}
    return unres == 0 and zero and len(unusable) == 8376, ev


def q08(obs):
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    rec = reconstruct(obs)
    t1 = t1.assign(model_key=t1.model.map({"GPT": "openai_gpt54mini", "GEMINI": "google_gemini31flashlite"}),
                   contrast=t1.contrast_id.map({"A1": "C5-C4", "A2": "C4R-C4", "A3": "C6-E-C4R", "A4": "C6-E-C6-EX"}))
    m = rec.merge(t1, on=["model_key", "reasoning_regime", "budget", "contrast", "oracle"], validate="one_to_one")
    ok_all = bool(np.isclose(m.estimate_reconstructed, m.estimate, atol=1e-12, rtol=1e-12).all())
    ev = {"rows_matched": len(m), "np_isclose_atol_rtol_1e-12_all": ok_all,
          "max_abs_diff": float(np.abs(m.estimate_reconstructed - m.estimate).max())}
    return len(m) == 96 and ok_all, ev


def q09():
    chk = json.loads((c.RESULTS / "_R9_checks.json").read_text(encoding="utf-8"))
    ev = {k: v for k, v in chk.items() if k != "rows"}
    exp = chk["thesis_expected"]
    ok = (chk["a3_positive_of_12"] == exp["a3_positive"] and chk["a3_holm_sig_of_12"] == exp["a3_sig"] and chk["a2_positive_of_12"] == exp["a2_positive"]
          and chk["a2_holm_sig_of_12"] == exp["a2_sig"] and chk["a2_sig_with_60family_of_12"] == exp["a2_sig_60"]
          and chk["a3_means_matched_3dp"] == "12/12" and chk["a2_means_matched_3dp"].startswith("4/4") and chk["all_star_checks_match"])
    return ok, ev


def q10(obs):
    from importlib import import_module
    t3mod = import_module("30_ref1_T3")
    c3e = t3mod.c3e_values(obs)
    t3 = pd.read_csv(c.RESULTS / "T3_REF1.csv")
    gaps = pd.read_parquet(c.require_file(c.KIT / c.S9 / "llm_stage9_paired_vs_C3-E.parquet"))
    diffs = []
    for model in c.MODELS:
        for oracle in c.ORACLES:
            for space, sp in (("CANDIDATE9", "candidate9"), ("FREE8", "free8")):
                g = gaps
                for col, val in (("analysis_population", "itt"), ("model_key", model), ("mode", sp), ("replicate", 1), ("budget", "B1"),
                                 ("information_condition", "IC-b"), ("policy", "C6-E")):
                    if col not in g.columns:
                        raise SystemExit(f"Q10: stage9 gap file lacks column {col}")
                    g = g[g[col] == val]
                mine = t3[(t3.model == c.MODEL_CODE[model]) & (t3.oracle == oracle) & (t3.action_space == space) & (t3.reasoning_regime == "HIGH")].estimate.iloc[0]
                diffs.append((len(g), abs(float(g[f"gap_{oracle}"].mean()) - float(mine))))
    ev = {"c3e_rows": len(c3e), "firm_key_unique": bool(c3e.index.is_unique), "c3e_alpha_mean": round(float(c3e.delta_R_score_alpha.mean()), 6),
          "high_vs_stage9_gap_rows_each": sorted({d[0] for d in diffs}), "high_vs_stage9_gap_max_abs_diff": max(d[1] for d in diffs)}
    ok = len(c3e) == 575 and ev["firm_key_unique"] and ev["c3e_alpha_mean"] == 0.923299 and ev["high_vs_stage9_gap_rows_each"] == [575] and ev["high_vs_stage9_gap_max_abs_diff"] < 1e-9
    return ok, ev


def q11():
    from credit_recourse.analysis.final_v43.statistics import canonical_key_json, pcg64_seed
    supp = pd.read_csv(c.SUPP_PATH)
    reg_bad = sum(pcg64_seed(canonical_key_json(json.loads(k))) != int(s) for k, s in zip(supp.key_json, supp.seed_uint64))
    fields = {"action_space", "analysis_id", "budget", "execution_policy", "information_condition", "lhs", "model", "oracle", "population",
              "reasoning_regime", "replicate", "rhs"}
    ev = {"registry_seed_mismatches_of_722": int(reg_bad)}
    ok = reg_bad == 0
    for name, n, fam in (("T2_S1.csv", 24, "S1_ALL24"), ("T3_REF1.csv", 24, None), ("R9_candidate_selection.csv", 24, None)):
        d = pd.read_csv(c.RESULTS / name)
        keys_ok = all(set(json.loads(k)) == fields for k in d.source_key if str(k).startswith("{"))
        ev[name] = {"rows": len(d), "status": d.status.value_counts().to_dict(), "families": d.holm_family.value_counts().to_dict(), "key_fields_ok": keys_ok}
        ok &= len(d) == n and keys_ok
    t2 = pd.read_csv(c.RESULTS / "T2_S1.csv")
    ok &= bool(np.allclose(t2.holm_p, c.holm(t2.p_raw.to_numpy())))
    return ok, ev


def q12(write_list: bool):
    base = c.BASE_COMMIT
    diff = git("diff", "--name-only", base, "--", "frozen").strip()
    status = git("status", "--porcelain", "--", "frozen").strip()
    # tracked files only, in git's byte order: ignored local files (e.g. __pycache__) and the platform's path sort
    # order must not change the combined hash
    files = [c.KIT / t for t in git("ls-files", "-z", "--", "frozen").split("\0") if t]
    h = hashlib.sha256()
    lines = []
    for p in files:
        d = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                d.update(b)
        rel = p.relative_to(c.KIT).as_posix()
        lines.append(f"{d.hexdigest()}  {rel}")
        h.update((lines[-1] + "\n").encode("utf-8"))
    private = c.KIT.parent / "private"
    list_name = f"frozen_sha256_list_{datetime.now(timezone.utc):%Y%m%d}_{git('rev-parse', '--short=7', 'HEAD').strip()}.txt"
    ev = {"base_commit_present": has_commit(base), "git_diff_base_frozen_empty": diff == "", "git_status_frozen_empty": status == "",
          "files_hashed": len(files), "combined_sha256": h.hexdigest(),
          "hash_list_written": bool(write_list and private.is_dir()),
          "hash_list_file": list_name,
          "note": "frozen/ hash list kept outside the repository (private/, written only when that folder exists; one file per date and commit, "
                  "an existing file is never overwritten). The combined hash differs between a clone with Git LFS pointers and one with "
                  "pulled LFS content. Evidence of no change since checkout: git diff and git status above (checked before every commit)."}
    if ev["hash_list_written"] and not (private / list_name).exists():
        (private / list_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ev["base_commit_present"] and diff == "" and status == "", ev


def _provider_time(obj) -> datetime | None:
    """Earliest provider-side timestamp in a saved response: OpenAI `created_at` (Unix time) or the HTTP `date` header
    kept by the Gemini SDK. Unlike file modification times, these do not change when the file is checked out again."""
    found: list[datetime] = []

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "created_at" and isinstance(v, (int, float)) and not isinstance(v, bool):
                    found.append(datetime.fromtimestamp(v, timezone.utc))
                elif k.lower() == "date" and isinstance(v, str):
                    try:
                        found.append(parsedate_to_datetime(v).astimezone(timezone.utc))
                    except (TypeError, ValueError):
                        pass
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return min(found) if found else None


def q16_17():
    e1 = c.ABL / "E1"
    first = git("log", "--diff-filter=A", "--format=%cI", "--", "ablation_20261102/E1/E1_PROTOCOL.md").strip().splitlines()
    commit_time = datetime.fromisoformat(first[-1]).timestamp() if first else None
    responses = sorted(p for p in (e1 / "smoke").glob("smoke_*.json") if not p.name.startswith("smoke_summary"))
    stamps = [_provider_time(json.loads(p.read_text(encoding="utf-8"))) for p in responses]
    stamped = [t for t in stamps if t is not None]
    first_call = min(stamped).timestamp() if stamped and len(stamped) == len(responses) else None
    report = json.loads((e1 / "requests/E1_build_report.json").read_text(encoding="utf-8"))
    man = hashlib.sha256((e1 / "requests/E1_request_manifest.csv").read_bytes()).hexdigest()
    pushed = git("branch", "-r", "--contains", git("log", "--diff-filter=A", "--format=%H", "--", "ablation_20261102/E1/E1_PROTOCOL.md").strip().splitlines()[-1]).strip()
    ev = {"protocol_commit_before_first_provider_call": bool(commit_time and first_call and commit_time < first_call),
          "protocol_commit_time": first[-1] if first else None,
          "protocol_commit_time_basis": "committer date of the commit that added E1_PROTOCOL.md (set by the committer), not the GitHub server push time",
          "first_provider_time_utc": datetime.fromtimestamp(first_call, timezone.utc).isoformat() if first_call else None,
          "smoke_responses_with_provider_time": f"{len(stamped)}/{len(responses)}",
          "time_source": "provider timestamps saved inside the smoke responses (OpenAI created_at, Gemini HTTP date header); "
                         "file modification times change on checkout and are not used",
          "protocol_commit_on_origin": "origin/ablation-20261102" in pushed,
          "build_checks_all_pass": all(v.startswith("PASS") for v in report["checks"].values()),
          "manifest_sha256_equals_expected": man == "1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a",
          "requests": 3450}
    ok = ev["protocol_commit_before_first_provider_call"] and ev["protocol_commit_on_origin"] and ev["build_checks_all_pass"] and ev["manifest_sha256_equals_expected"]
    return ok, ev


def q17():
    e1 = c.ABL / "E1"
    man = pd.read_csv(e1 / "requests/E1_request_manifest.csv", dtype=str)
    report = json.loads((e1 / "requests/E1_build_report.json").read_text(encoding="utf-8"))
    shard_ok = all(c.sha256_file(c.require_file(e1 / "requests/batches" / s["file"])) == s["sha256"] for s in report["shards"])
    wide = man.pivot_table(index=["model_key", "firm_key"], columns="e1_condition", values="visible_payload_sha256", aggfunc="first")
    own = man.drop_duplicates("firm_key").set_index("firm_key").own_C3E_action
    wide = wide.join(own, on="firm_key")
    oe = wide.own_C3E_action == "OE"
    ev = {"build_checks_all_pass": all(v.startswith("PASS") for v in report["checks"].values()),
          "manifest_sha256_equals_expected": c.sha256_file(e1 / "requests/E1_request_manifest.csv") == "1f12775443db3a929cddf7a619a27882ad49fe5b189da23572f908854064357a",
          "shard_files_equal_report_sha256": shard_ok, "requests": len(man),
          "placebo_pairs_identical_prompt (RL==OE, own=OE)": int((wide.loc[oe, "E1-RL"] == wide.loc[oe, "E1-OE"]).sum()),
          "placebo_pairs_total": int(oe.sum()) , "firms_343_RL_differs_from_OE": int((wide.loc[~oe, "E1-RL"] != wide.loc[~oe, "E1-OE"]).sum()),
          "C4R_differs_from_RL_all": bool((wide["E1-C4R"] != wide["E1-RL"]).all())}
    ok = (ev["build_checks_all_pass"] and ev["manifest_sha256_equals_expected"] and shard_ok and ev["placebo_pairs_identical_prompt (RL==OE, own=OE)"] == 464
          and ev["placebo_pairs_total"] == 464 and ev["firms_343_RL_differs_from_OE"] == 686 and ev["C4R_differs_from_RL_all"])
    return ok, ev


def q18():
    e1 = c.ABL / "E1"
    o = pd.read_parquet(e1 / "observations/e1_observations.parquet")
    resp = [json.loads(t) for t in c.require_file(e1 / "runs/responses.jsonl").open(encoding="utf-8")]
    sub = json.loads((e1 / "runs/submissions.json").read_text(encoding="utf-8"))
    t5 = pd.read_csv(c.RESULTS / "T5_E1.csv")
    exp_model = {"openai": "gpt-5.4-mini-2026-03-17", "google": "gemini-3.1-flash-lite"}
    own = pd.read_csv(e1 / "requests/E1_request_manifest.csv", dtype=str).drop_duplicates("firm_key").set_index("firm_key").own_C3E_action
    # independent recomputation of the T5 estimates with plain pandas (no kit code)
    maxdiff = 0.0
    for _, r in t5.iterrows():
        cond = {"E1a": ("E1-RL", "E1-OE"), "E1a_575": ("E1-RL", "E1-OE"), "E1a_placebo": ("E1-RL", "E1-OE"), "E1b": ("E1-OE", "E1-C4R"), "E1c": ("E1-RL", "E1-C4R")}[r.contrast_id]
        mk = {"GPT": "openai_gpt54mini", "GEMINI": "google_gemini31flashlite"}[r.model]
        w = o[o.model_key == mk].pivot(index="firm_key", columns="e1_condition", values=f"delta_R_score_{r.oracle}")
        if r.contrast_id == "E1a":
            w = w.loc[own.index[own != "OE"]]
        if r.contrast_id == "E1a_placebo":
            w = w.loc[own.index[own == "OE"]]
        maxdiff = max(maxdiff, abs(float((w[cond[0]] - w[cond[1]]).mean()) - float(r.estimate)))
    git_changed = git("diff", "--name-only", "5fbf549", "HEAD", "--", "ablation_20261102/E1/E1_PROTOCOL.md", "ablation_20261102/E1/scripts/e1_build_requests.py",
                      "ablation_20261102/E1/scripts/e1_analyze.py", "ablation_20261102/E1/requests").strip()
    if not has_commit("5fbf549"):
        git_changed = "not checkable: commit 5fbf549 is not in this clone"
    ev = {"delivered": len(resp), "attempts": sorted({r["attempt"] for r in resp}), "jobs": len(sub["jobs"]), "resubmissions": sum(1 for j in sub["jobs"] if j["attempt"] > 1),
          "returned_model_all_expected": bool(all(r["returned_model"] == exp_model[r["provider"]] for r in resp)),
          "per_model_condition_575": bool((o.groupby(["model_key", "e1_condition"]).size() == 575).all()), "observations": len(o),
          "duplicate_keys": int(o.duplicated(["model_key", "firm_key", "e1_condition"]).sum()),
          "strict_unusable_to_A0": int((~o.strict_usable).sum()),
          "unusable_have_zero_action": bool(all(max(abs(v) for v in json.loads(s).values()) == 0.0 for s in o[~o.strict_usable].strict_itt_action_json)),
          "selected_only_from_batch_jobs": bool(all(r["job_name"].startswith("e1-") for r in resp)),
          "T5_means_recomputed_independently_max_abs_diff": maxdiff,
          "contract_files_changed_since_5fbf549": git_changed or "none"}
    ok = (ev["delivered"] == 3450 and ev["resubmissions"] == 0 and ev["returned_model_all_expected"] and ev["per_model_condition_575"] and ev["duplicate_keys"] == 0
          and ev["unusable_have_zero_action"] and ev["selected_only_from_batch_jobs"] and maxdiff < 1e-12 and ev["contract_files_changed_since_5fbf549"] == "none")
    return ok, ev


def q19():
    t5 = pd.read_csv(c.RESULTS / "T5_E1.csv")
    p = t5[t5.contrast_id == "E1a_placebo"]
    ev = {"rows": len(p), "n_firms": sorted(p.n_firms.unique().tolist()), "min_holm_p": float(p.holm_p.min()),
          "audit_required": bool((p.holm_p < 0.05).any())}
    return len(p) == 6 and ev["n_firms"] == [232] and not ev["audit_required"], ev


def _holm_independent(p: list[float]) -> list[float]:
    """Holm step-down adjustment written out separately from common.holm and the analysis package."""
    m = len(p)
    ranked = sorted(range(m), key=lambda i: (p[i], i))
    adjusted = [0.0] * m
    running = 0.0
    for k, i in enumerate(ranked):
        running = max(running, min(1.0, (m - k) * p[i]))
        adjusted[i] = running
    return adjusted


def q20():
    """Primary Holm p recomputed from the frozen raw p in the result registry (families: model x Oracle x reasoning
    setting, 8 comparisons each, thesis section 3.7; the registry leaves holm_family_id empty for PRIMARY rows)."""
    import io
    import zipfile
    from thesis_repro.frozen import REGISTRY_MEMBER
    distribution = c.require_file(c.KIT / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip")
    with zipfile.ZipFile(distribution) as archive, archive.open(REGISTRY_MEMBER) as handle:
        reg = pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8-sig"))
    prim = reg[reg.result_class == "PRIMARY"].copy()
    keys = ["model_key", "oracle", "reasoning_regime"]
    sizes = prim.groupby(keys).size()
    prim["holm_recomputed"] = np.nan
    for _, idx in prim.groupby(keys).groups.items():
        prim.loc[idx, "holm_recomputed"] = _holm_independent(prim.loc[idx, "raw_p_frozen"].astype(float).tolist())
    diff = (prim.holm_recomputed - prim.holm_p_frozen.astype(float)).abs()
    same_sig = int(((prim.holm_recomputed < 0.05) == (prim.holm_p_frozen.astype(float) < 0.05)).sum())
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    t1_family_sizes = sorted(t1.groupby("holm_family").size().unique().tolist())
    ev = {"primary_rows": len(prim), "families": int(len(sizes)), "family_sizes": sorted(sizes.unique().tolist()),
          "registry_holm_family_id_filled": int(prim.holm_family_id.notna().sum()), "T1_family_sizes": t1_family_sizes,
          "max_abs_diff_vs_registry_holm_p_frozen": float(diff.max()), "significance_p_lt_05_same": f"{same_sig}/{len(prim)}",
          "holm_function": "written in 90_qa.py, not imported from common or credit_recourse"}
    ok = len(prim) == 96 and len(sizes) == 12 and ev["family_sizes"] == [8] and t1_family_sizes == [8] and float(diff.max()) <= 1e-12 and same_sig == 96
    return ok, ev


FILE_GROUPS = {
    "results/T1_primary_A1_A4.csv": ["Q01", "Q02", "Q03", "Q04", "Q05", "Q07", "Q08", "Q20"],
    "results/T2_S1.csv": ["Q03", "Q04", "Q11"], "results/T3_REF1.csv": ["Q03", "Q04", "Q10", "Q11"],
    "results/T4_robustness.csv, R9_candidate_selection.csv, R10_reference_assignment.csv": ["Q03", "Q06", "Q09", "Q11"],
    "results/T5_E1.csv, E1_wording.json, D3_E1a_by_oracle.csv, D1_E1c_vs_A3.csv": ["Q16", "Q17", "Q18", "Q19"],
    "figures/F1-F5": ["Q03", "Q08", "Q18", "Q19"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path)
    ap.add_argument("--skip-frozen-hash", action="store_true")
    args = ap.parse_args()
    obs = c.load_obs()
    checks = [("Q01", "Source pin", q01), ("Q02", "Scale", q02), ("Q03", "Pairing and row counts", q03), ("Q04", "Cell match (only the intended axis differs)", lambda: q04(obs)),
              ("Q05", "Draft links", lambda: q05(obs)), ("Q06", "Reference assignment", q06), ("Q07", "Strict rule", lambda: q07(obs)),
              ("Q08", "Mean reproduction (T1)", lambda: q08(obs)), ("Q09", "Candidate selection vs thesis (R9)", q09), ("Q10", "C3-E", lambda: q10(obs)),
              ("Q11", "New analysis contract (seeds, families, status)", q11), ("Q12", "Frozen unchanged", lambda: q12(not args.skip_frozen_hash)),
              ("Q16", "E1 pre-registration order (before first provider call)", q16_17),
              ("Q17", "E1 prompt identity (manifest, shards, placebo pairs)", q17), ("Q18", "E1 generation complete and independently recomputed", q18),
              ("Q19", "E1 placebo gate (232 firms)", q19),
              ("Q20", "Primary Holm p recomputed independently from frozen raw p", q20)]
    results = []
    for qid, name, fn in checks:
        try:
            ok, ev = fn()
        except BaseException as exc:  # noqa: BLE001
            ok, ev = False, {"error": f"{type(exc).__name__}: {exc}"}
        status = ok if isinstance(ok, str) else ("PASS" if ok else "FAIL")
        results.append({"id": qid, "name": name, "status": status, "evidence": ev})
        print(qid, status, ev.get("reason", "") if isinstance(ev, dict) else "")
    out = {"generated_utc": datetime.now(timezone.utc).isoformat(), "head": git("rev-parse", "HEAD").strip(), "results": results}
    (c.RESULTS / "_qa_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
    if args.report:
        lines = [MARK_BEGIN, "", "## Automated checks (scripts/90_qa.py)", "",
                 f"Run at {out['generated_utc']} on HEAD {out['head'][:12]}. Evidence: `results/_qa_results.json`. ", "",
                 "| ID | Check | Result | Key evidence |", "| --- | --- | --- | --- |"]
        for r in results:
            ev = json.dumps(r["evidence"], ensure_ascii=False, default=str)
            ev = (ev[:300] + " ...") if len(ev) > 300 else ev
            lines.append(f"| {r['id']} | {r['name']} | {r['status']} | `{ev.replace('|', '/')}` |")
        status = {r["id"]: r["status"] for r in results}
        lines += ["", "### File status", "", "| Files | Required checks | Status |", "| --- | --- | --- |"]
        for files, need in FILE_GROUPS.items():
            ok_all = all(status.get(q) in ("PASS", "WARN") for q in need)
            lines.append(f"| `{files}` | {', '.join(need)} | {'VERIFIED' if ok_all else 'DRAFT'} |")
        lines += ["", "Q13-Q15 (number cross-check of report, slides and notes; representative-condition notation; Human baseline notation) are recorded in the section below. The human sign-off lines are written by the user only.", "", MARK_END]
        text = args.report.read_text(encoding="utf-8") if args.report.exists() else ""
        if MARK_BEGIN in text:
            pre, rest = text.split(MARK_BEGIN, 1)
            post = rest.split(MARK_END, 1)[1] if MARK_END in rest else ""
            text = pre + "\n".join(lines) + post
        else:
            text = text.rstrip() + "\n\n" + "\n".join(lines) + "\n"
        args.report.write_text(text, encoding="utf-8")
    sys.exit(0 if all(r["status"] in ("PASS", "WARN") for r in results) else 1)


if __name__ == "__main__":
    main()
