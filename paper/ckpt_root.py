"""Where the analysis scripts look for trained checkpoints.

Checkpoints are not distributed; REPRODUCE.md section 3 gives the training
recipe for every run name these scripts reference. Train them into a single
directory laid out as

    <root>/<run-name>/agent_<step>.msgpack

and point the scripts at it. The default root is ./checkpoints relative to the
repository, so `save_dir=checkpoints/<run-name>/` in the training commands puts
them where the analysis expects.

Override with the LP_CKPT_ROOT environment variable; an s3:// URI also works,
since the loaders accept one.

    LP_CKPT_ROOT=/data/lp-checkpoints python paper/forced_opening_adv.py
"""
import os
import pathlib

DEFAULT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent / "checkpoints")


def ckpt_root() -> str:
    """Root directory (or s3:// prefix) holding <run-name>/agent_<step>.msgpack."""
    return os.environ.get("LP_CKPT_ROOT", DEFAULT_ROOT).rstrip("/")


def agent_path(run: str, step: int = 2_000_000) -> str:
    """Full path to one trained agent checkpoint."""
    return f"{ckpt_root()}/{run}/agent_{step}.msgpack"


def require(run: str, step: int = 2_000_000) -> str:
    """agent_path(), but fail with a useful message instead of a stack trace."""
    p = agent_path(run, step)
    if not p.startswith("s3://") and not os.path.exists(p):
        raise SystemExit(
            f"missing checkpoint: {p}\n"
            f"  '{run}' is not distributed -- train it first. REPRODUCE.md section 3\n"
            f"  lists the recipe for every run name, and section 4 the experiment\n"
            f"  it belongs to. Set LP_CKPT_ROOT to use a different directory."
        )
    return p
