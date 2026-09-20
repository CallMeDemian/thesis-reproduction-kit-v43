from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .parent_gate import load as load_parent
from .paths import frozen_v13_root
from .primary import reconstruct
from .registry import result_id
from .schema import validate
from .selection import source_path
from .supplemental import load as load_supplemental


MODEL_LABELS = {"openai_gpt54mini": "GPT", "google_gemini31flashlite": "GEMINI"}


def _frozen_contract(root: Path, profile: str, run_id: str):
    if profile == "MANUSCRIPT_FROZEN":
        base = frozen_v13_root(root)
        return (pd.read_csv(base / "frozen_inputs/stage8/V13_BASELINE_STRICT_PRIMARY_FULLSTREAM.csv"),
                pd.read_csv(base / "frozen_inputs/stage8/V13_HIGH_STRICT_PRIMARY_FULLSTREAM.csv"),
                load_supplemental(root), load_parent(root))
    stage9 = root / "runs" / run_id / "stage9"
    required = [stage9 / "PRIMARY_RESULTS.csv", stage9 / "SUPPLEMENTAL_RESULTS.csv", stage9 / "PARENT_GATE_RESULTS.csv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("BLOCKED_CLEAN_RUN_INPUT_MISSING:" + ",".join(missing))
    return pd.read_csv(required[0]), pd.DataFrame(), pd.read_csv(required[1]), pd.read_csv(required[2])


def _primary_results(root: Path, frame: pd.DataFrame, profile: str, run_id: str) -> pd.DataFrame:
    reconstructed = reconstruct(frame)
    if len(reconstructed) != 96:
        raise ValueError(f"primary contract requires 96 rows, got {len(reconstructed)}")
    stored_base, stored_high, _, _ = _frozen_contract(root, profile, run_id)
    if profile == "MANUSCRIPT_FROZEN":
        stored_high = stored_high.copy()
        if "arm" not in stored_high:
            stored_high.insert(0, "arm", "HIGH_STRICT_ITT")
        stored = pd.concat([stored_base, stored_high], ignore_index=True)
        stored["model_key"] = stored["model"].map({v: k for k, v in MODEL_LABELS.items()}).fillna(stored["model"])
        stored["reasoning_regime"] = stored["arm"].astype(str).str.split("_").str[0]
        stored = stored.rename(columns={
            "N": "N_frozen", "estimate": "estimate_frozen", "ci_low": "ci_low_frozen",
            "ci_high": "ci_high_frozen", "raw_p": "raw_p_frozen", "holm_p": "holm_p_frozen",
        })
        join = ["model_key", "reasoning_regime", "budget", "contrast", "oracle"]
        merged = reconstructed.merge(stored, on=join, how="left", validate="one_to_one", suffixes=("", "_frozen"))
        if merged["estimate_frozen"].isna().any() or merged["N_frozen"].isna().any():
            raise ValueError("primary frozen production stream does not cover reconstructed keys")
        if not np.isclose(merged["estimate_reconstructed"], merged["estimate_frozen"], atol=1e-12, rtol=1e-12).all():
            raise ValueError("primary reconstructed estimate mismatch")
        merged["estimate"] = merged["estimate_frozen"]
        merged["ci_low"] = merged["ci_low_frozen"]
        merged["ci_high"] = merged["ci_high_frozen"]
        merged["raw_p"] = merged["raw_p_frozen"]
        merged["holm_p"] = merged["holm_p_frozen"]
        merged["ci_source"] = "FROZEN_PRODUCTION_STREAM"
        merged["raw_p_source"] = "FROZEN_PRODUCTION_STREAM"
        merged["holm_source"] = "FROZEN_PRODUCTION_STREAM"
    else:
        merged = reconstructed.merge(stored_base, on=["model_key", "reasoning_regime", "budget", "contrast", "oracle"], how="left", validate="one_to_one")
        merged["estimate"] = merged["estimate_reconstructed"]
        merged["estimate_frozen"] = np.nan
        merged["ci_source"] = "CLEAN_REEXECUTION"
        merged["raw_p_source"] = "CLEAN_REEXECUTION"
        merged["holm_source"] = "CLEAN_REEXECUTION"
    merged["result_class"] = "PRIMARY"
    merged["thesis_role"] = np.where(merged.reasoning_regime.eq("BASELINE"), "PRIMARY_THESIS", "SUPPLEMENTAL_REASONING")
    merged["model"] = merged.model_key.map(MODEL_LABELS).fillna(merged.model_key)
    merged["source"] = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN" if profile == "MANUSCRIPT_FROZEN" else f"runs/{run_id}/stage8"
    merged["result_id"] = merged.apply(result_id, axis=1)
    if merged.result_id.duplicated().any():
        raise ValueError("primary result_id collision")
    return merged


def _supplemental_results(root: Path, profile: str, run_id: str) -> pd.DataFrame:
    _, _, supplemental, _ = _frozen_contract(root, profile, run_id)
    frame = supplemental.copy()
    frame["result_class"] = frame.analysis_id
    frame["thesis_role"] = "SUPPLEMENTAL"
    frame["estimate_frozen"] = frame["estimate"]
    frame["estimate_reconstructed"] = np.nan
    frame["source"] = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/canonical_v13" if profile == "MANUSCRIPT_FROZEN" else f"runs/{run_id}/stage9"
    frame["result_id"] = frame.apply(result_id, axis=1)
    return frame


def _parent_results(root: Path, profile: str, run_id: str) -> pd.DataFrame:
    _, _, _, frame = _frozen_contract(root, profile, run_id)
    frame = frame.copy()
    frame["result_class"] = "PARENT_GATE"
    frame["thesis_role"] = "DESCRIPTIVE_SENSITIVITY"
    frame["estimate"] = frame["parent_gate_mean"]
    frame["estimate_frozen"] = frame["parent_gate_mean"]
    frame["estimate_reconstructed"] = np.nan
    frame["ci_low"] = np.nan
    frame["ci_high"] = np.nan
    frame["raw_p"] = np.nan
    frame["holm_p"] = np.nan
    frame["source"] = "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/canonical_v13" if profile == "MANUSCRIPT_FROZEN" else f"runs/{run_id}/stage9"
    frame["result_id"] = frame.apply(result_id, axis=1)
    return frame


def _write_outputs(root: Path, run_id: str, profile: str, primary: pd.DataFrame, supplemental: pd.DataFrame, parent: pd.DataFrame, source: Path) -> dict:
    out = root / "runs" / run_id / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    pp = supplemental.loc[supplemental.analysis_id.eq("PP_CONTRAST")].copy()
    rq4 = supplemental.loc[supplemental.analysis_id.eq("RQ4_ACTION_SPACE")].copy()
    rq5 = supplemental.loc[supplemental.analysis_id.eq("RQ5_INFORMATION")].copy()
    interactions = supplemental.loc[supplemental.analysis_id.isin(["BUDGET_INTERACTION", "REASONING_REGIME_INTERACTION", "EXECUTION_SENSITIVITY"])].copy()
    for name, current in (("PRIMARY_RESULTS.csv", primary), ("SUPPLEMENTAL_RESULTS.csv", supplemental), ("PP_RESULTS.csv", pp), ("RQ4_RESULTS.csv", rq4), ("RQ5_RESULTS.csv", rq5), ("INTERACTION_RESULTS.csv", interactions), ("PARENT_GATE_RESULTS.csv", parent)):
        current.to_csv(out / name, index=False)
    registry = pd.concat([primary, supplemental, parent], ignore_index=True, sort=False)
    if len(registry) != 914 or registry.result_id.duplicated().any():
        raise ValueError("result registry must contain 914 unique stable result IDs")
    registry.to_csv(out / "RESULT_REGISTRY.csv", index=False)
    manifest = {
        "run_id": run_id, "profile": profile, "source": str(source), "source_rows": 96600,
        "primary_rows": len(primary), "supplemental_rows": len(supplemental), "parent_rows": len(parent),
        "thesis_contract_rows": len(registry), "pairing_replicate_key": True,
        "primary_filters": {"reasoning_regime": "BASELINE thesis / HIGH supplemental", "execution_policy": "STRICT", "action_space": "FREE8", "replicate": 1, "information_condition": "IC-b", "population": "ITT"},
        "primary_provenance": {"estimate": "firm_level_stage8_reconstructed", "ci": "FROZEN_PRODUCTION_STREAM", "raw_p": "FROZEN_PRODUCTION_STREAM", "holm": "FROZEN_PRODUCTION_STREAM"},
        "supplemental_counts": supplemental.analysis_id.value_counts().to_dict(),
        "parent_gate": "DESCRIPTIVE_SENSITIVITY_ONLY",
    }
    (out / "ANALYSIS_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def run(root: Path, run_id: str, profile: str) -> dict:
    root = Path(root).resolve()
    source = source_path(root, run_id, profile)
    frame = pd.read_parquet(source)
    validate(frame)
    primary = _primary_results(root, frame, profile, run_id)
    supplemental = _supplemental_results(root, profile, run_id)
    parent = _parent_results(root, profile, run_id)
    return {"status": "PASS", **_write_outputs(root, run_id, profile, primary, supplemental, parent, source)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default="ANALYSIS_REVIEW")
    parser.add_argument("--profile", choices=["MANUSCRIPT_FROZEN", "CLEAN_REEXECUTION"], default="MANUSCRIPT_FROZEN")
    args = parser.parse_args(argv)
    try:
        result = run(Path(args.project_root), args.run_id, args.profile)
    except FileNotFoundError as exc:
        print(f"BLOCKED_CLEAN_RUN_INPUT_MISSING:{exc}")
        return 2
    except Exception as exc:
        print(f"FINAL_ANALYSIS_FAIL:{type(exc).__name__}:{exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False))
    print("FINAL_ANALYSIS_SMOKE_PASS" if args.profile == "CLEAN_REEXECUTION" else "FINAL_ANALYSIS_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
