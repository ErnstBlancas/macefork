"""A `pickle_module` for `torch.load` that reads models pickled by upstream mace.

Whole-model files (foundation models, checkpoints written by `mace-torch`)
record their classes as `mace.modules.models.ScaleShiftMACE` and so on. This
fork lives under `macefork`, so without remapping those files either fail to
load or, when upstream mace is installed alongside, silently come back as
upstream objects. The unpickler below rewrites `mace` / `mace.*` to
`macefork` / `macefork.*` and leaves every other module untouched.
"""

import io
import pickle
from pickle import *  # noqa: F401,F403  # pylint: disable=wildcard-import,unused-wildcard-import

_LEGACY = "mace"
_CURRENT = "macefork"


def _remap(module: str) -> str:
    if module == _LEGACY or module.startswith(_LEGACY + "."):
        return _CURRENT + module[len(_LEGACY) :]
    return module


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_remap(module), name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def loads(data, /, **kwargs):
    return Unpickler(io.BytesIO(data), **kwargs).load()
