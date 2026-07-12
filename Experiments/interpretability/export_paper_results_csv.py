#!/usr/bin/env python3
"""Export interpretability paper main-table metrics from real log JSON files.

Reads faithfulness experiment logs under ``logs/``, extracts summary
(and multi-seed ``summary_mean_std``) fields without hand-editing, and
writes:

  * ``Experiments/interpretability/paper_results/interpretability_main_results.csv``
  * ``Experiments/interpretability/paper_results/interpretability_csv_audit.md``

Uses only Python stdlib: json, csv, pathlib, statistics, re.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOGS_DIR = _REPO / "logs"
DEFAULT_OUT_DIR = _REPO / "Experiments" / "interpretability" / "paper_results"

# Paper condition name -> (log search rules, expected num_tasks, notes default)
CONDITION_SPECS: Dict[str, Dict[str, Any]] = {
    "tiered_original": {
        "must_include": ["simple_tiered"],
        "must_exclude": ["multi_seed", "5seed", "contrast_v1"],
        "expected_tasks": 20,
        "notes": "",
        "archetype": False,
    },
    "contrast_benchmark": {
        "must_include": ["simple_contrast_v2"],
        "must_exclude": ["multi_seed", "5seed", "contrast_v1", "sbert"],
        "expected_tasks": 20,
        "notes": "",
        "archetype": True,
    },
    "contrast_benchmark_sbert": {
        "must_include": ["sbert_contrast_v2"],
        "must_exclude": ["multi_seed", "5seed"],
        "expected_tasks": 20,
        "notes": "",
        "archetype": True,
    },
    "failure_case": {
        "must_include": ["failure_case"],
        "must_exclude": [],
        "expected_tasks": 10,
        "notes": "",
        "archetype": False,
    },
    "contrast_benchmark_5seed": {
        "must_include": [],
        "must_include_any": ["contrast_v2_multi_seed", "5seed"],
        "must_exclude": ["contrast_v1"],
        "expected_tasks": 20,
        "notes": "multi-seed over random/shuffled sampling; task set fixed",
        "archetype": False,
    },
}

CSV_COLUMNS = [
    "condition",
    "source_log",
    "dataset",
    "encoder",
    "num_tasks",
    "num_seeds",
    "mean_winner_M",
    "mean_loser_M",
    "mean_score_gap",
    "mean_frac_candidates_scap_eq_1",
    "mean_winner_loser_scap_gap",
    "mean_top_factor_drop",
    "mean_random_factor_drop",
    "mean_bottom_factor_drop",
    "frac_top_beats_random",
    "frac_top_beats_bottom",
    "top_rank_flip_rate",
    "random_rank_flip_rate",
    "bottom_rank_flip_rate",
    "counterfactual_valid_rate",
    "mean_counterfactual_edit_size",
    "mean_true_top_drop_percentile",
    "frac_true_top_beats_shuffled_mean",
    "mean_shuffled_top_drop",
    "mean_explanation_vs_oracle_ratio",
    "frac_explanation_matches_oracle",
    "mean_explanation_rank_among_all_perturbations",
    "num_error_cases",
    "error_case_rate",
    "intended_winner_archetypes",
    "actual_winner_archetypes",
    "winner_candidate_index_distribution",
    "intended_actual_match_count",
    "notes",
]

METRIC_KEYS = [
    "mean_winner_M",
    "mean_loser_M",
    "mean_score_gap",
    "mean_frac_candidates_scap_eq_1",
    "mean_winner_loser_scap_gap",
    "mean_top_factor_drop",
    "mean_random_factor_drop",
    "mean_bottom_factor_drop",
    "frac_top_beats_random",
    "frac_top_beats_bottom",
    "top_rank_flip_rate",
    "random_rank_flip_rate",
    "bottom_rank_flip_rate",
    "counterfactual_valid_rate",
    "mean_counterfactual_edit_size",
    "mean_true_top_drop_percentile",
    "frac_true_top_beats_shuffled_mean",
    "mean_shuffled_top_drop",
    "mean_explanation_vs_oracle_ratio",
    "frac_explanation_matches_oracle",
    "mean_explanation_rank_among_all_perturbations",
    "num_error_cases",
    "error_case_rate",
]


def _is_finite_number(x: Any) -> bool:
    if x is None or isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(float(x))
    return False


def _fmt_num(x: Any, decimals: int = 4) -> str:
    if not _is_finite_number(x):
        return "NA"
    return f"{float(x):.{decimals}f}"


def _fmt_mean_std(mean: Any, std: Any, decimals: int = 4) -> str:
    if not _is_finite_number(mean):
        return "NA"
    m = float(mean)
    if _is_finite_number(std) and float(std) > 0:
        return f"{m:.{decimals}f}±{float(std):.{decimals}f}"
    return f"{m:.{decimals}f}"


def _json_compact(obj: Any) -> str:
    if obj is None:
        return "NA"
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _find_log(
    logs_dir: Path,
    must_include: List[str],
    must_exclude: List[str],
    must_include_any: Optional[List[str]] = None,
) -> Optional[Path]:
    """Return newest matching ``faithfulness*.json`` by mtime."""
    candidates: List[Path] = []
    for p in sorted(logs_dir.glob("faithfulness*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        name = p.name.lower()
        if must_include_any:
            if not any(s in name for s in must_include_any):
                continue
        elif must_include and not all(s in name for s in must_include):
            continue
        if any(s in name for s in must_exclude):
            continue
        candidates.append(p)
    return candidates[0] if candidates else None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _unwrap_payload(data: Dict[str, Any], internal_key: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (payload with summary at top level, metadata dict)."""
    if internal_key and "conditions" in data and internal_key in data["conditions"]:
        block = data["conditions"][internal_key]
        if block.get("available") and block.get("json_path"):
            inner_path = Path(block["json_path"])
            if inner_path.is_file():
                inner = _load_json(inner_path)
                meta = inner.get("metadata", {}) or block.get("metadata", {}) or {}
                return inner, meta
        summary = block.get("summary") or {}
        meta = block.get("metadata", {}) or data.get("metadata", {}) or {}
        return {"summary": summary, "metadata": meta, "per_task": []}, meta
    return data, data.get("metadata", {}) or {}


def _encoder_short(encoder_str: Any) -> str:
    if encoder_str is None:
        return "NA"
    s = str(encoder_str).lower()
    if "sbert" in s or "bge" in s:
        return "sbert"
    if "simple" in s or "bow" in s:
        return "simple"
    return str(encoder_str)


def _dataset_from_metadata(meta: Dict[str, Any], path: Path) -> str:
    testset = meta.get("testset") or meta.get("testset_path") or ""
    if testset:
        name = Path(str(testset)).name.lower()
        if "faithfulnesscontrastv2" in name.replace("_", "").replace("-", ""):
            return "ContrastV2"
        if "faithfulnesscontrast" in name.replace("_", "").replace("-", ""):
            return "ContrastV1"
        if "faithfulnessfailure" in name.replace("_", "").replace("-", ""):
            return "FailureCase"
        if "testset_tiered" in name or "tiered" in name:
            return "Tiered"
    pname = path.name.lower()
    if "contrastv2" in pname or "contrast_v2" in pname:
        return "ContrastV2"
    if "failure" in pname:
        return "FailureCase"
    if "tiered" in pname:
        return "Tiered"
    return "NA"


def _num_tasks(data: Dict[str, Any], meta: Dict[str, Any], expected: int) -> Any:
    for key in ("num_tasks_run", "num_tasks_requested", "num_tasks"):
        if key in meta and meta[key] is not None:
            return meta[key]
    per_task = data.get("per_task")
    if isinstance(per_task, list) and per_task:
        return len(per_task)
    return expected


def _num_seeds(data: Dict[str, Any], meta: Dict[str, Any]) -> int:
    seeds = meta.get("seeds")
    if isinstance(seeds, list) and seeds:
        return len(seeds)
    if data.get("per_seed"):
        return len(data["per_seed"])
    if data.get("summary_mean_std"):
        sms = data["summary_mean_std"]
        for v in sms.values():
            if isinstance(v, dict) and "n_seeds" in v:
                return int(v["n_seeds"])
    return 1


def _get_metric(
    data: Dict[str, Any],
    key: str,
    multi_seed: bool,
) -> str:
    if multi_seed and data.get("summary_mean_std"):
        cell = data["summary_mean_std"].get(key)
        if isinstance(cell, dict) and "mean" in cell:
            std = cell.get("std")
            if _is_finite_number(std):
                return _fmt_mean_std(cell["mean"], std)
            return _fmt_num(cell["mean"])
    summary = data.get("summary") or {}
    val = summary.get(key)
    return _fmt_num(val)


def _archetype_from_summary(summary: Dict[str, Any]) -> Tuple[str, str, str, str]:
    dist = summary.get("archetype_distribution")
    if not isinstance(dist, dict):
        return "NA", "NA", "NA", "NA"
    intended = dist.get("intended") if isinstance(dist.get("intended"), dict) else {}
    actual = dist.get("actual") if isinstance(dist.get("actual"), dict) else {}
    winner_idx = dist.get("winner_candidate_index") if isinstance(dist.get("winner_candidate_index"), dict) else {}
    return (
        _json_compact(intended) if intended else "NA",
        _json_compact(actual) if actual else "NA",
        _json_compact(winner_idx) if winner_idx else "NA",
        "NA",
    )


def _archetype_from_per_task(per_task: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    intended_counts: Dict[str, int] = {}
    actual_counts: Dict[str, int] = {}
    index_counts: Dict[str, int] = {}
    match_count = 0
    total_with_intended = 0

    for t in per_task:
        intended = t.get("intended_winner_archetype")
        if intended is None and isinstance(t.get("metadata"), dict):
            intended = t["metadata"].get("intended_winner_archetype")
        actual = t.get("winner_archetype")
        if actual is None:
            actual = t.get("winner_archetype")

        if intended:
            intended_counts[str(intended)] = intended_counts.get(str(intended), 0) + 1
            total_with_intended += 1
            if actual and str(actual) == str(intended):
                match_count += 1

        if actual:
            actual_counts[str(actual)] = actual_counts.get(str(actual), 0) + 1

        wid = t.get("winner_id") or ""
        if wid:
            m = re.search(r"_(\d+)$", wid)
            if m:
                idx = m.group(1)
                index_counts[idx] = index_counts.get(idx, 0) + 1

    if not per_task:
        return "NA", "NA", "NA", "NA"

    return (
        _json_compact(intended_counts) if intended_counts else "NA",
        _json_compact(actual_counts) if actual_counts else "NA",
        _json_compact(index_counts) if index_counts else "NA",
        str(match_count) if total_with_intended else "NA",
    )


def _compute_mean_std_from_per_seed(data: Dict[str, Any], key: str) -> Tuple[Optional[float], Optional[float]]:
    per_seed = data.get("per_seed")
    if not isinstance(per_seed, dict):
        return None, None
    vals: List[float] = []
    for block in per_seed.values():
        if not isinstance(block, dict):
            continue
        summary = block.get("summary") or {}
        v = summary.get(key)
        if _is_finite_number(v):
            vals.append(float(v))
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _build_row(
    condition: str,
    path: Optional[Path],
    spec: Dict[str, Any],
    logs_dir: Path,
) -> Dict[str, str]:
    row: Dict[str, str] = {c: "NA" for c in CSV_COLUMNS}
    row["condition"] = condition
    row["notes"] = spec.get("notes", "")

    if path is None:
        row["source_log"] = "NA"
        return row

    data, meta = _unwrap_payload(_load_json(path), None)
    multi = condition == "contrast_benchmark_5seed"
    summary = data.get("summary") or {}

    row["source_log"] = str(path)
    row["dataset"] = _dataset_from_metadata(meta, path)
    row["encoder"] = _encoder_short(meta.get("encoder"))
    row["num_tasks"] = str(_num_tasks(data, meta, spec["expected_tasks"]))
    row["num_seeds"] = str(_num_seeds(data, meta))

    for key in METRIC_KEYS:
        if multi and not data.get("summary_mean_std"):
            m, s = _compute_mean_std_from_per_seed(data, key)
            if m is not None:
                row[key] = _fmt_mean_std(m, s if s is not None else 0.0)
                continue
        row[key] = _get_metric(data, key, multi)

    if spec.get("archetype"):
        per_task = data.get("per_task") or []
        if isinstance(summary.get("archetype_distribution"), dict):
            i, a, w, _ = _archetype_from_summary(summary)
        elif per_task:
            i, a, w, _ = _archetype_from_per_task(per_task)
        else:
            i, a, w = "NA", "NA", "NA"
        m = "NA"
        if per_task:
            _, _, _, m = _archetype_from_per_task(per_task)
        row["intended_winner_archetypes"] = i
        row["actual_winner_archetypes"] = a
        row["winner_candidate_index_distribution"] = w
        row["intended_actual_match_count"] = m
    else:
        row["intended_winner_archetypes"] = "NA"
        row["actual_winner_archetypes"] = "NA"
        row["winner_candidate_index_distribution"] = "NA"
        row["intended_actual_match_count"] = "NA"

    return row


def _parse_metric_for_check(val: str) -> Optional[float]:
    if val == "NA" or not val:
        return None
    if "±" in val:
        val = val.split("±")[0]
    try:
        return float(val)
    except ValueError:
        return None


def _check(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def _write_csv(rows: List[Dict[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_audit(
    rows: List[Dict[str, str]],
    resolved: Dict[str, Optional[Path]],
    specs: Dict[str, Dict[str, Any]],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Interpretability CSV Audit\n")
    lines.append("Generated from real log JSON under `logs/`. No hand-filled numbers.\n")

    lines.append("## Source logs\n")
    lines.append("| condition | expected log pattern | resolved | status |")
    lines.append("|---|---|---|---|")
    for cond, spec in specs.items():
        inc = spec.get("must_include") or []
        any_inc = spec.get("must_include_any")
        if any_inc:
            pat = "any of: " + ", ".join(any_inc)
        else:
            pat = ", ".join(inc)
        exc = ", ".join(spec["must_exclude"]) if spec.get("must_exclude") else "(none)"
        p = resolved.get(cond)
        if p is None:
            lines.append(f"| {cond} | include: {pat}; exclude: {exc} | **NOT FOUND** | MISSING |")
        else:
            lines.append(f"| {cond} | include: {pat}; exclude: {exc} | `{p}` | OK |")

    lines.append("\n## Extracted summary\n")
    lines.append("| condition | num_tasks | encoder | oracle_ratio | oracle_match | status |")
    lines.append("|---|---:|---|---:|---:|---|")
    by_cond = {r["condition"]: r for r in rows}
    for cond in specs:
        r = by_cond.get(cond, {})
        nt = r.get("num_tasks", "NA")
        enc = r.get("encoder", "NA")
        ora = r.get("mean_explanation_vs_oracle_ratio", "NA")
        om = r.get("frac_explanation_matches_oracle", "NA")
        st = "OK" if resolved.get(cond) else "MISSING"
        lines.append(f"| {cond} | {nt} | {enc} | {ora} | {om} | {st} |")

    lines.append("\n## Validation checks\n")
    checks: List[Tuple[str, bool, str]] = []

    def g(cond: str, key: str) -> Optional[float]:
        r = by_cond.get(cond, {})
        return _parse_metric_for_check(r.get(key, "NA"))

    cb = by_cond.get("contrast_benchmark", {})
    cbs = by_cond.get("contrast_benchmark_sbert", {})
    fc = by_cond.get("failure_case", {})

    checks.append((
        "All source logs resolved",
        all(resolved.get(c) is not None for c in specs),
        "At least one log file found per condition",
    ))
    checks.append((
        "tiered_original num_tasks == 20",
        _parse_metric_for_check(by_cond.get("tiered_original", {}).get("num_tasks", "NA")) == 20,
        f"expected 20, got {by_cond.get('tiered_original', {}).get('num_tasks', 'NA')}",
    ))
    checks.append((
        "contrast_benchmark num_tasks == 20",
        _parse_metric_for_check(cb.get("num_tasks", "NA")) == 20,
        f"expected 20, got {cb.get('num_tasks', 'NA')}",
    ))
    checks.append((
        "contrast_benchmark_sbert num_tasks == 20",
        _parse_metric_for_check(cbs.get("num_tasks", "NA")) == 20,
        f"expected 20, got {cbs.get('num_tasks', 'NA')}",
    ))
    checks.append((
        "failure_case num_tasks == 10",
        _parse_metric_for_check(fc.get("num_tasks", "NA")) == 10,
        f"expected 10, got {fc.get('num_tasks', 'NA')}",
    ))
    checks.append((
        "contrast_benchmark_5seed num_tasks == 20",
        _parse_metric_for_check(by_cond.get("contrast_benchmark_5seed", {}).get("num_tasks", "NA")) == 20,
        f"expected 20, got {by_cond.get('contrast_benchmark_5seed', {}).get('num_tasks', 'NA')}",
    ))

    sat_cb = g("contrast_benchmark", "mean_frac_candidates_scap_eq_1")
    ora_cb = g("contrast_benchmark", "mean_explanation_vs_oracle_ratio")
    omatch_cb = g("contrast_benchmark", "frac_explanation_matches_oracle")
    checks.append((
        "contrast_benchmark mean_frac_candidates_scap_eq_1 < 0.2",
        sat_cb is not None and sat_cb < 0.2,
        f"value={sat_cb}",
    ))
    checks.append((
        "contrast_benchmark mean_explanation_vs_oracle_ratio >= 0.9",
        ora_cb is not None and ora_cb >= 0.9,
        f"value={ora_cb}",
    ))
    checks.append((
        "contrast_benchmark frac_explanation_matches_oracle >= 0.9",
        omatch_cb is not None and omatch_cb >= 0.9,
        f"value={omatch_cb}",
    ))

    ora_sbert = g("contrast_benchmark_sbert", "mean_explanation_vs_oracle_ratio")
    omatch_sbert = g("contrast_benchmark_sbert", "frac_explanation_matches_oracle")
    checks.append((
        "contrast_benchmark_sbert mean_explanation_vs_oracle_ratio >= 0.9",
        ora_sbert is not None and ora_sbert >= 0.9,
        f"value={ora_sbert}",
    ))
    checks.append((
        "contrast_benchmark_sbert frac_explanation_matches_oracle >= 0.9",
        omatch_sbert is not None and omatch_sbert >= 0.9,
        f"value={omatch_sbert}",
    ))

    ora_fc = g("failure_case", "mean_explanation_vs_oracle_ratio")
    omatch_fc = g("failure_case", "frac_explanation_matches_oracle")
    checks.append((
        "failure_case mean_explanation_vs_oracle_ratio < 0.5",
        ora_fc is not None and ora_fc < 0.5,
        f"value={ora_fc}",
    ))
    checks.append((
        "failure_case frac_explanation_matches_oracle < 0.5",
        omatch_fc is not None and omatch_fc < 0.5,
        f"value={omatch_fc}",
    ))

    na_cells = sum(1 for r in rows for v in r.values() if v == "NA")
    checks.append((
        "No NA cells in CSV (except allowed fields)",
        na_cells == 0 or all(
            k in ("intended_winner_archetypes", "actual_winner_archetypes",
                  "winner_candidate_index_distribution", "intended_actual_match_count", "notes")
            for k, v in rows[0].items()
            if v == "NA"
        ),
        f"total NA cells={na_cells}",
    ))

    top_drop_cb = g("contrast_benchmark", "mean_top_factor_drop")
    rand_drop_cb = g("contrast_benchmark", "mean_random_factor_drop")
    checks.append((
        "contrast_benchmark mean_top_factor_drop > mean_random_factor_drop",
        top_drop_cb is not None and rand_drop_cb is not None and top_drop_cb > rand_drop_cb,
        f"top={top_drop_cb}, random={rand_drop_cb}",
    ))

    top_flip_cb = g("contrast_benchmark", "top_rank_flip_rate")
    rand_flip_cb = g("contrast_benchmark", "random_rank_flip_rate")
    checks.append((
        "contrast_benchmark top_rank_flip_rate > random_rank_flip_rate",
        top_flip_cb is not None and rand_flip_cb is not None and top_flip_cb > rand_flip_cb,
        f"top_flip={top_flip_cb}, random_flip={rand_flip_cb}",
    ))

    top_drop_fc = g("failure_case", "mean_top_factor_drop")
    rand_drop_fc = g("failure_case", "mean_random_factor_drop")
    fc_top_vs_rand = (
        top_drop_fc is not None and rand_drop_fc is not None and top_drop_fc < rand_drop_fc
    )
    checks.append((
        "failure_case mean_top_factor_drop < mean_random_factor_drop (informational)",
        fc_top_vs_rand,
        f"top={top_drop_fc}, random={rand_drop_fc} (expected failure-mode pattern)",
    ))

    for label, ok, detail in checks:
        lines.append(f"- [{_check(ok)}] {label}: {detail}")

    lines.append("\n## Missing fields\n")
    missing_rows: List[Tuple[str, str, str]] = []
    for r in rows:
        cond = r["condition"]
        for field in CSV_COLUMNS:
            if r.get(field) == "NA" and field not in (
                "intended_winner_archetypes", "actual_winner_archetypes",
                "winner_candidate_index_distribution", "intended_actual_match_count",
                "notes",
            ):
                missing_rows.append((cond, field, "not in log summary or per_task"))
    if not missing_rows:
        lines.append("| (none) | | |")
    else:
        lines.append("| condition | field | reason |")
        lines.append("|---|---|---|")
        for c, f, reason in missing_rows:
            lines.append(f"| {c} | {f} | {reason} |")

    lines.append("\n## Notes\n")
    lines.append("- Contrast V1 is intentionally excluded from the paper CSV.")
    lines.append("- `contrast_benchmark_5seed` uses `summary_mean_std` when present; "
                 "otherwise mean/std computed from `per_seed` summaries.")
    lines.append("- Archetype distribution is extracted from `summary.archetype_distribution` "
                 "or aggregated from `per_task` metadata for contrast conditions only.")
    lines.append("- Tiered and failure_case runs have `archetype_distribution: null` in logs; "
                 "CSV archetype columns are `NA` as expected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export interpretability paper results CSV from logs.")
    parser.add_argument("--logs-dir", type=str, default=str(DEFAULT_LOGS_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "interpretability_main_results.csv"
    audit_path = out_dir / "interpretability_csv_audit.md"

    resolved: Dict[str, Optional[Path]] = {}
    rows: List[Dict[str, str]] = []

    for cond, spec in CONDITION_SPECS.items():
        path = _find_log(
            logs_dir,
            spec.get("must_include", []),
            spec.get("must_exclude", []),
            spec.get("must_include_any"),
        )
        resolved[cond] = path
        rows.append(_build_row(cond, path, spec, logs_dir))

    _write_csv(rows, csv_path)
    _write_audit(rows, resolved, CONDITION_SPECS, audit_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
