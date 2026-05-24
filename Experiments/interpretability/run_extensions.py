#!/usr/bin/env python3
"""Run all §7 follow-up interpretability experiments.

Usage (from repo root):
  cp .env.example .env   # paste OPENAI_API_KEY=sk-or-v1-...
  python Experiments/interpretability/run_extensions.py --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
RUN = _HERE / "run_interpretability.py"
GEN_THETA = _HERE / "generate_theta_contrast_testset.py"
TEAM = _HERE / "team_interpretability.py"
PLOT_EXT = _HERE / "plot_extensions.py"

sys.path.insert(0, str(_HERE))
from env_loader import load_dotenv, check_openrouter_key  # noqa: E402


def _py(*args: str) -> int:
    cmd = [sys.executable, str(RUN), *args]
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO))


def _run_script(path: Path) -> int:
    cmd = [sys.executable, str(path)]
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO))


def suite_theta_contrast(require_api_key: bool) -> int:
    rc = _run_script(GEN_THETA)
    if rc != 0:
        return rc
    theta_path = _REPO / "simulator" / "20_Tasks_ThetaContrast.json"
    extra = ["--require-api-key"] if require_api_key else []
    return _py(
        "--testset", str(theta_path),
        "--n-rounds", "30", "--pool-size", "5",
        "--feedback", "gt",
        "--tag", "ext-theta-contrast",
        *extra,
    )


def suite_skill_feedback(require_api_key: bool) -> int:
    extra = ["--require-api-key"] if require_api_key else []
    for fb, tag in (("sim", "ext-sim-baseline"), ("skill", "ext-skill-feedback")):
        rc = _py(
            "--n-rounds", "30", "--pool-size", "5",
            "--num-tasks", "10",
            "--feedback", fb,
            "--tag", tag,
            *extra,
        )
        if rc != 0:
            return rc
    return 0


def suite_bge_encoder(require_api_key: bool) -> int:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        print("[extensions] SKIP bge_encoder — pip install sentence-transformers")
        return 0
    extra = ["--require-api-key"] if require_api_key else []
    return _py(
        "--encoder", "sbert",
        "--n-rounds", "30", "--pool-size", "5",
        "--num-tasks", "5",
        "--feedback", "gt",
        "--tag", "ext-bge-attention",
        *extra,
    )


def suite_multi_seed(require_api_key: bool) -> int:
    extra = ["--require-api-key"] if require_api_key else []
    return _py(
        "--n-rounds", "30", "--pool-size", "5",
        "--num-tasks", "5",
        "--feedback", "gt",
        "--n-seeds", "5",
        "--seed", "42",
        "--tag", "ext-multi-seed",
        *extra,
    )


def suite_team_shapley(require_api_key: bool) -> int:
    del require_api_key
    cmd = [sys.executable, str(TEAM)]
    print("\n>>>", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO))


def suite_dream_large(require_api_key: bool) -> int:
    if require_api_key and not check_openrouter_key():
        print("[extensions] ERROR: dream_large needs OPENAI_API_KEY in .env")
        return 1
    extra = ["--require-api-key"] if require_api_key else []
    return _py(
        "--n-rounds", "10", "--pool-size", "5",
        "--feedback", "sim",
        "--dreaming",
        "--dream-base-url", "openrouter",
        "--dream-model", "deepseek/deepseek-chat-v3-0324",
        "--dream-n-turns", "2",
        "--tag", "ext-dream-large-real",
        *extra,
    )


SUITES = {
    "theta_contrast": suite_theta_contrast,
    "skill_feedback": suite_skill_feedback,
    "bge_encoder": suite_bge_encoder,
    "multi_seed": suite_multi_seed,
    "team_shapley": suite_team_shapley,
    "dream_large": suite_dream_large,
}


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run interpretability extensions")
    parser.add_argument(
        "--suite", default="",
        help=f"Comma-separated: {','.join(SUITES)} or use --all",
    )
    parser.add_argument("--all", action="store_true", help="Run all suites")
    parser.add_argument(
        "--require-api-key", action="store_true",
        help="Fail if dreaming requested without OPENAI_API_KEY",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Run plot_extensions.py after experiments",
    )
    args = parser.parse_args()

    if args.all:
        names = list(SUITES.keys())
    elif args.suite:
        names = [s.strip() for s in args.suite.split(",") if s.strip()]
    else:
        parser.print_help()
        return 0

    for name in names:
        if name not in SUITES:
            print(f"Unknown suite: {name}")
            return 1
        rc = SUITES[name](args.require_api_key)
        if rc != 0:
            print(f"[extensions] FAILED suite={name} rc={rc}")
            return rc

    if args.plot and PLOT_EXT.is_file():
        subprocess.call([sys.executable, str(PLOT_EXT)], cwd=str(_REPO))

    print("\n[extensions] All requested suites completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
