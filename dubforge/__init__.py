"""DubForge: an expressive, modular anime dubbing pipeline (Japanese -> English).

Every stage reads and writes files in a per-video work folder, so you can stop,
inspect or hand-edit anything (especially script.json), and resume.
"""

__version__ = "0.1.0"


def json_default(o):
    """Let json.dumps write numpy numbers/bools (np.float32, np.bool_, arrays) that models hand back."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
