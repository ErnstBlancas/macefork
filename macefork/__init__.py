import os
import warnings

from .__version__ import __version__

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# Hide FutureWarnings (torch.jit on Python 3.14, MACE v1.0 removal notices).
# Set here so it applies before torch, e3nn or les are imported.
warnings.filterwarnings("ignore", category=FutureWarning)
