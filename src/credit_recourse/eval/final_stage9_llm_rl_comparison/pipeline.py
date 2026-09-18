from __future__ import annotations

"""Final V4.3 Stage 9 comparison and pre-specified Plan-3 contrasts.

This stage consumes frozen Stage7 actions, Stage8 Oracle scores, and the
current Stage6 575x9 payoff surface. It never chooses, rewrites, clips,
reranks, or feeds an action back into inference.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.contracts.stage_paths import stage_dir
from credit_recourse.rl.common.io import read_parquet_required, write_json
from .revision_metrics import (
    PRIMARY_CONTRASTS,
    build_identity_contrast_table,
    build_primary_contrasts,
    build_revision_table,
    load_action_geometry,
)

ACTION_ORDER = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
CONDITIONS = {"C4", "C5", "C4R", "C6-E", "C6-EX"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _current_stage6(project_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    pointer_path = stage_dir(project_root, "stage6") / "CURRENT_RELEASE.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    payoff_path = project_root / str(pointer["payoff_surface_path"])
    if _sha256(payoff_path) != str(pointer["payoff_surface_sha256"]):
        raise ValueError("Current Stage6 payoff surface hash mismatch")
    payoffs = read_parquet_required(payoff_path)
    required = {"row_id", "firm_id", "action", "Alpha", "Beta", "Gamma"}
    if required - set(payoffs):
        raise ValueError(f"Stage6 payoff surface lacks {sorted(required-set(payoffs))}")
    if len(payoffs) != 575 * 9 or payoffs["row_id"].nunique() != 575:
        raise ValueError("Stage6 payoff surface is not the canonical 575x9 grid")
    if set(payoffs["action"].astype(str)) != set(ACTION_ORDER):
        raise ValueError("Stage6 payoff surface action contract mismatch")
    if payoffs.duplicated(["row_id", "action"]).any():
        raise ValueError("Stage6 payoff surface has duplicate firm-action rows")

    release_dir = project_root / str(pointer["artifact_path"])
    deployed = read_parquet_required(release_dir / "C3E_firm_actions_payoffs.parquet")
    if len(deployed) != 575 or deployed["row_id"].nunique() != 575:
        raise ValueError("Current deployed C3-E table is not the canonical 575 firms")
    return pointer, payoffs, deployed


def _reference_ladder(payoffs: pd.DataFrame, deployed: pd.DataFrame) -> pd.DataFrame:
    a0 = payoffs[payoffs["action"].astype(str).eq("A0")][
        ["row_id", "Alpha", "Beta", "Gamma"]
    ].rename(columns={k: f"A0_{k}" for k in ("Alpha", "Beta", "Gamma")})

    fixed = payoffs.rename(columns={"action": "policy"}).copy()
    fixed["candidate_id"] = fixed["policy"].astype(str)
    fixed["source"] = "stage6_fixed_action_surface"
    fixed["mode"] = "reference"

    c2 = deployed[
        ["row_id", "firm_id", "fiscal_year", "C2_action", "C2_Alpha", "C2_Beta", "C2_Gamma"]
    ].rename(columns={
        "C2_action": "candidate_id", "C2_Alpha": "Alpha",
        "C2_Beta": "Beta", "C2_Gamma": "Gamma",
    })
    c2["policy"] = "C2"
    c2["source"] = "stage6_current_deployed_release"
    c2["mode"] = "reference"

    c3 = deployed[
        ["row_id", "firm_id", "fiscal_year", "action_id", "Alpha", "Beta", "Gamma"]
    ].rename(columns={"action_id": "candidate_id"})
    c3["policy"] = "C3-E"
    c3["source"] = "stage6_current_deployed_release"
    c3["mode"] = "reference"

    out = pd.concat([fixed, c2, c3], ignore_index=True, sort=False).merge(
        a0, on="row_id", how="left", validate="many_to_one"
    )
    for backend, src in (("alpha", "Alpha"), ("beta", "Beta"), ("gamma", "Gamma")):
        out[f"R_score_{backend}"] = pd.to_numeric(out[src], errors="raise")
        out[f"delta_R_score_{backend}"] = (
            out[f"R_score_{backend}"] - pd.to_numeric(out[f"A0_{src}"], errors="raise")
        )
    return out


def _load_stage8(project_root: Path) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    path = stage_dir(project_root, "stage8") / "llm_stage8_multi_oracle_scores_all_populations.parquet"
    meta_path = stage_dir(project_root, "stage8") / "metadata.json"
    scores = read_parquet_required(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("semantic_contract_version") != "V4.3_FINAL_20260912":
        raise ValueError("Stage9 refuses a non-final V4.3 Stage8 artifact")
    if not set(scores["policy"].astype(str)).issubset(CONDITIONS):
        raise ValueError("Stage9 refuses non-Plan-3 condition codes")
    if set(scores["analysis_population"].astype(str)) != {"itt", "per_protocol"}:
        raise ValueError("Stage9 requires separate ITT and per-protocol populations")
    if scores.duplicated(["request_id", "analysis_population"]).any():
        raise ValueError("Stage8 request identity is duplicated within an analysis population")
    return scores, meta, path


def _stack_comparison(stage8: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    llm = stage8.copy()
    llm["source"] = "stage8_llm"
    if "candidate_id" not in llm:
        llm["candidate_id"] = "FREE8"
    refs = []
    for population in ("itt", "per_protocol"):
        piece = references.copy()
        piece["analysis_population"] = population
        refs.append(piece)
    return pd.concat([llm, *refs], ignore_index=True, sort=False)


def _policy_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    group = ["source", "analysis_population", "policy", "mode"]
    rows: list[dict[str, Any]] = []
    for keys, frame in comparison.groupby(group, dropna=False, sort=True):
        rec = dict(zip(group, keys))
        rec["n_rows"] = int(frame["row_id"].nunique())
        for backend in ("alpha", "beta", "gamma"):
            s = pd.to_numeric(frame[f"R_score_{backend}"], errors="coerce")
            d = pd.to_numeric(frame[f"delta_R_score_{backend}"], errors="coerce")
            rec[f"mean_R_score_{backend}"] = float(s.mean())
            rec[f"mean_delta_R_score_{backend}"] = float(d.mean())
            rec[f"sd_delta_R_score_{backend}"] = float(d.std(ddof=1))
            rec[f"positive_fraction_{backend}"] = float((d > 0).mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def _paired_against(stage8: pd.DataFrame, references: pd.DataFrame, policy: str) -> pd.DataFrame:
    ref = references[references["policy"].astype(str).eq(policy)][
        ["row_id", "candidate_id", *[f"R_score_{b}" for b in ("alpha", "beta", "gamma")]]
    ].rename(columns={
        "candidate_id": "reference_candidate_id",
        **{f"R_score_{b}": f"reference_R_score_{b}" for b in ("alpha", "beta", "gamma")},
    })
    if ref.empty:
        raise ValueError(f"Missing current Stage6 reference {policy}")
    out = stage8.merge(ref, on="row_id", how="left", validate="many_to_one")
    out["reference_policy"] = policy
    for backend in ("alpha", "beta", "gamma"):
        out[f"gap_{backend}"] = (
            out[f"R_score_{backend}"] - out[f"reference_R_score_{backend}"]
        )
    return out


def _aggregate_failure_audit(project_root: Path) -> pd.DataFrame:
    path = stage_dir(project_root, "stage8") / "llm_stage8_failure_audit_enriched.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    groups = [
        c for c in ("policy", "model_key", "mode", "information_condition", "budget")
        if c in df
    ]
    counts = df.groupby(groups, dropna=False).size().reset_index(name="n_rows")
    bool_cols = [c for c in df if c.startswith("fail_") or c.startswith("has_")]
    if "routed_to_simulator" in df:
        bool_cols.append("routed_to_simulator")
    if not bool_cols:
        return counts
    sums = df.groupby(groups, dropna=False)[bool_cols].sum().reset_index()
    return counts.merge(sums, on=groups, how="left", validate="one_to_one")


def run_stage9(*, project_root: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    out = stage_dir(project_root, "stage9")
    out.mkdir(parents=True, exist_ok=True)

    pointer, payoffs, deployed = _current_stage6(project_root)
    stage8, stage8_meta, stage8_path = _load_stage8(project_root)
    geometry = load_action_geometry(project_root)
    if tuple(geometry.fixed_candidates) != ACTION_ORDER:
        raise ValueError("Stage9 current action order mismatch")

    references = _reference_ladder(payoffs, deployed)
    comparison = _stack_comparison(stage8, references)
    comparison.to_parquet(out / "llm_stage9_llm_rl_comparison.parquet", index=False)
    comparison.to_csv(out / "llm_stage9_llm_rl_comparison.csv", index=False, encoding="utf-8-sig")
    _policy_summary(comparison).to_csv(
        out / "llm_stage9_policy_summary.csv", index=False, encoding="utf-8-sig"
    )

    paired_files: dict[str, str] = {}
    for reference in ("A0", "OE", "C2", "C3-E"):
        paired = _paired_against(stage8, references, reference)
        name = f"llm_stage9_paired_vs_{reference}.parquet"
        paired.to_parquet(out / name, index=False)
        paired_files[reference] = name

    action_tables = {
        "itt": stage_dir(project_root, "stage7") / "llm_stage7_action_table_itt.parquet",
        "per_protocol": stage_dir(project_root, "stage7") / "llm_stage7_action_table_per_protocol.parquet",
    }
    revision_parts = []
    for population, path in action_tables.items():
        actions = read_parquet_required(path)
        scores = stage8[stage8["analysis_population"].astype(str).eq(population)].copy()
        revision_parts.append(
            build_revision_table(action_table=actions, stage8_scores=scores, geometry=geometry)
        )
    revision = pd.concat(revision_parts, ignore_index=True, sort=False)
    revision.to_parquet(out / "llm_stage9_revision_metrics.parquet", index=False)
    revision.to_csv(out / "llm_stage9_revision_metrics.csv", index=False, encoding="utf-8-sig")
    identity = build_identity_contrast_table(revision)
    identity.to_parquet(out / "llm_stage9_identity_contrast.parquet", index=False)
    identity.to_csv(out / "llm_stage9_identity_contrast.csv", index=False, encoding="utf-8-sig")

    firm, contrast_summary, interaction, interaction_summary = build_primary_contrasts(
        stage8, bootstrap_replicates=10_000, bootstrap_seed=20_260_912
    )
    expected = {label for label, _, _ in PRIMARY_CONTRASTS}
    if set(firm["contrast"].astype(str)) != expected:
        raise ValueError("Stage9 did not materialize all four pre-specified contrasts")
    if set(firm["budget"].astype(str)) != {"B1", "BINF"}:
        raise ValueError("Stage9 primary contrast table does not preserve B1/BINF")
    if "model_key" not in contrast_summary or contrast_summary["model_key"].isna().any():
        raise ValueError("Stage9 summaries must keep models separate")
    firm.to_parquet(out / "llm_stage9_primary_contrast_firm_level.parquet", index=False)
    firm.to_csv(out / "llm_stage9_primary_contrast_firm_level.csv", index=False, encoding="utf-8-sig")
    contrast_summary.to_csv(out / "llm_stage9_primary_contrast_summary.csv", index=False, encoding="utf-8-sig")
    interaction.to_parquet(out / "llm_stage9_budget_interaction_firm_level.parquet", index=False)
    interaction.to_csv(out / "llm_stage9_budget_interaction_firm_level.csv", index=False, encoding="utf-8-sig")
    interaction_summary.to_csv(out / "llm_stage9_budget_interaction_summary.csv", index=False, encoding="utf-8-sig")

    failure = _aggregate_failure_audit(project_root)
    failure.to_csv(out / "llm_stage9_failure_audit.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "stage": "final_stage9_llm_rl_comparison",
        "status": "PASS",
        "created_utc": _now(),
        "scientific_contract_version": "V4.3_FINAL_20260912",
        "analysis_contract": "final_plan3_four_primary_contrasts_v1",
        "primary_contrasts": [label for label, _, _ in PRIMARY_CONTRASTS],
        "information_budgets": ["B1", "BINF"],
        "budget_interaction": "firm_level_B1_minus_BINF",
        "models_analyzed_separately": True,
        "analysis_populations": ["itt", "per_protocol"],
        "bootstrap": {"replicates": 10_000, "rng": "PCG64", "seed": 20_260_912},
        "current_stage6_release_id": pointer["release_id"],
        "current_stage6_release_hash": pointer["release_hash"],
        "stage6_payoff_surface_sha256": pointer["payoff_surface_sha256"],
        "stage8_input": str(stage8_path),
        "stage8_input_sha256": _sha256(stage8_path),
        "stage8_final_paper_run_allowed": bool(stage8_meta.get("final_paper_run_allowed")),
        "rows": {
            "comparison": int(len(comparison)),
            "revision": int(len(revision)),
            "identity": int(len(identity)),
            "primary_contrast_firm": int(len(firm)),
            "budget_interaction_firm": int(len(interaction)),
        },
        "no_clipping": True,
        "no_projection": True,
        "no_rescale": True,
        "no_rerank": True,
        "conditional_descriptive_statistics_only": True,
        "development_surface_caveat": (
            "2024 is a repeatedly used development/search evaluation surface, not a pristine holdout."
        ),
        "outputs": {
            "comparison": "llm_stage9_llm_rl_comparison.parquet",
            "policy_summary": "llm_stage9_policy_summary.csv",
            "paired_references": paired_files,
            "revision_metrics": "llm_stage9_revision_metrics.parquet",
            "identity_contrast": "llm_stage9_identity_contrast.parquet",
            "primary_contrast_firm_level": "llm_stage9_primary_contrast_firm_level.parquet",
            "primary_contrast_summary": "llm_stage9_primary_contrast_summary.csv",
            "budget_interaction_firm_level": "llm_stage9_budget_interaction_firm_level.parquet",
            "budget_interaction_summary": "llm_stage9_budget_interaction_summary.csv",
            "failure_audit": "llm_stage9_failure_audit.csv",
        },
    }
    write_json(out / "metadata.json", metadata)
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Final V4.3 Stage9 analysis")
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run_stage9(project_root=Path(args.project_root)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

