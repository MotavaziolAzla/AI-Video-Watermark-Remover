import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
E2FGVI_DIR = BASE_DIR / "models" / "E2FGVI"

# Clear any cached 'model' module
for mod in list(sys.modules.keys()):
    if mod.startswith("model"):
        del sys.modules[mod]

sys.path.insert(0, str(E2FGVI_DIR))
from model.e2fgvi_hq import InpaintGenerator
print("InpaintGenerator imported OK from E2FGVI:", InpaintGenerator)
