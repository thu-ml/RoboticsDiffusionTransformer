import sys
from pathlib import Path

# The repository is a set of top-level packages (models, scripts, train, ...), so the
# tests need the repository root on the import path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
