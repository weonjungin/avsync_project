from pathlib import Path
import yaml

def project_root():
    return Path(__file__).resolve().parents[1]

def load_cfg(path: str):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = project_root() / p
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def abspath(p: str | None):
    if p is None:
        return None
    p = Path(p)
    if p.is_absolute():
        return str(p)
    return str((project_root() / p).resolve())
