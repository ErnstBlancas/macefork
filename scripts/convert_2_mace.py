"""Make a model saved by macefork loadable by upstream mace.

A whole-model file is a zip archive whose data.pkl records each class by module
path, e.g. `macefork.modules.models.ScaleShiftMACE`. Upstream mace has the same
classes under `mace.*`, so this rewrites those module paths and copies every
other entry unchanged. The inverse of macefork/tools/legacy_pickle.py.

    python convert_2_mace.py finetune.model [finetune_mace.model]
"""

import argparse
import pickletools
import zipfile
from pathlib import Path

OLD = "macefork"
NEW = "mace"


def rename_modules(data: bytes) -> bytes:
    """Rewrite `macefork.*` class references in a pickle stream to `mace.*`."""
    out = bytearray()
    ops = list(pickletools.genops(data))
    for i, (op, arg, pos) in enumerate(ops):
        end = ops[i + 1][2] if i + 1 < len(ops) else len(data)
        chunk = data[pos:end]
        if op.name == "GLOBAL":  # torch saves with protocol 2: "c<module>\n<name>\n"
            module, name = arg.split(" ", 1)
            if module == OLD or module.startswith(OLD + "."):
                module = NEW + module[len(OLD) :]
                chunk = f"c{module}\n{name}\n".encode()
        out += chunk
    if OLD.encode() in out:
        raise ValueError(f"'{OLD}' still referenced after conversion, cannot convert")
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="model file saved by macefork")
    parser.add_argument(
        "output", nargs="?", help="converted model (default: <model>_mace.model)"
    )
    args = parser.parse_args()

    model = Path(args.model)
    output = Path(args.output or model.with_name(model.stem + "_mace.model"))
    if output.resolve() == model.resolve():
        raise SystemExit("output must differ from the input model")

    with zipfile.ZipFile(model) as src, zipfile.ZipFile(output, "w") as dst:
        for entry in src.infolist():
            data = src.read(entry)
            if entry.filename.endswith("/data.pkl"):
                data = rename_modules(data)
            dst.writestr(entry, data)
    print(f"written {output}")


if __name__ == "__main__":
    main()
