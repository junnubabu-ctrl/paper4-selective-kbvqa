from pathlib import Path as _Path
import sys as _sys
_ROOT=_Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in _sys.path: _sys.path.insert(0, str(_ROOT / "src"))
from paper4_kbvqa.utils.environment import write_environment
if __name__ == "__main__":
    print(write_environment("results/environment.json"))
