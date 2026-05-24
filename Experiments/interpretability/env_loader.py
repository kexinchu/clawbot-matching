"""Load optional .env and validate LLM credentials for dreaming experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """Parse a simple KEY=VALUE .env file into os.environ (no extra deps)."""
    env_path = path or (_REPO_ROOT / ".env")
    loaded: Dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        if key:
            loaded[key] = value
    return loaded


def check_openrouter_key() -> bool:
    """Return True if OPENAI_API_KEY looks configured."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    return len(key) > 10


def dream_config_from_env() -> Dict[str, str]:
    """Read dreaming defaults from env (after load_dotenv)."""
    return {
        "base_url": os.environ.get("DREAM_BASE_URL", "openrouter"),
        "model": os.environ.get(
            "DREAM_MODEL", "deepseek/deepseek-chat-v3-0324",
        ),
        "n_turns": os.environ.get("DREAM_N_TURNS", "2"),
    }


def ensure_dreaming_credentials() -> None:
    """Raise with a helpful message if real dreaming was requested but no key."""
    if not check_openrouter_key():
        raise RuntimeError(
            "OPENAI_API_KEY is not set. For OpenRouter + DeepSeek dreaming:\n"
            "  1. cp .env.example .env\n"
            "  2. paste your OpenRouter key into OPENAI_API_KEY\n"
            "  3. re-run (or: export OPENAI_API_KEY=sk-or-v1-...)\n"
            "See Experiments/weight_interpretability.md §6."
        )
