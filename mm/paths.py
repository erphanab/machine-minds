"""
Where weights and datasets live. v6 keeps its own models/ and data/ (empty at the start) and reads v5's when it has
none of its own, so v6 never writes into v5. MM_FALLBACK_ROOT overrides the fallback project (default ../v5).
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_ROOT = os.environ.get("MM_FALLBACK_ROOT", os.path.join(os.path.dirname(ROOT), "v5"))


def _first(kind, name):
    for base in (ROOT, FALLBACK_ROOT):
        p = os.path.join(base, kind, name)
        if os.path.exists(p):
            return p
    return os.path.join(ROOT, kind, name)


def model_dir(run):
    """models/<run>: v6's own, else the fallback project's, else v6's (to be created)."""
    return _first("models", run)


def data_dir(name):
    """data/<name>, looked up like model_dir."""
    return _first("data", name)
