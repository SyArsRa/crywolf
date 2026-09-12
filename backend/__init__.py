"""Cry Wolf.

Loading `.env` happens here, on package import, because it has to happen before
anything reads `os.environ` -- and `llm.py` reads the key at Observer
construction time, which can be deep inside a request. Putting it in the package
init means every entry point gets it for free: the server, the feeder, and
Lane B's `run_observer` / `test_observer` scripts alike.

Real environment variables win over `.env` (that's `load_dotenv`'s default), so
an explicit `export` still overrides the file for a one-off run.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv not installed; env vars still work
        return
    # Explicit path, not discovery: the .env belongs to the repo, and this must
    # behave the same whatever directory you happen to run from.
    load_dotenv(REPO_ROOT / ".env")


_load_dotenv()
