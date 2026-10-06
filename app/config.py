"""Configuration with an injectable local date for reproducible evaluation."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Read only this project's .env; existing environment values take priority."""
    load_dotenv(ROOT / ".env", override=False)


def local_today() -> date:
    return datetime.now(timezone(timedelta(hours=7))).date()


@dataclass(frozen=True)
class Settings:
    today: date
    max_model_calls: int = 18
    max_tool_calls: int = 12
    max_replans: int = 2
    repeat_limit: int = 3
    stall_limit: int = 5
    consecutive_error_limit: int = 3
