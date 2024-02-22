import os
import sys
import traceback
from pathlib import Path
from typing import Callable, Generic, TypeVar

from spandrel import ModelDescriptor, ModelLoader
import torch
from safetensors.torch import save_file

CURRENT_DIR = Path(__file__).resolve().parent

T = TypeVar("T")


class Lazy(Generic[T]):
    def __init__(self, func: Callable[[], T]):
        self.func = func
        self._value = None

    @property
    def value(self):
        if self._value is None:
            self._value = self.func()
        return self._value


def get_files_to_convert():
    dir = CURRENT_DIR / "to-convert"
    files = os.listdir(dir)
    return [
        dir / f
        for f in files
        if f.lower().endswith((".ckpt", ".pth", ".pt", ".safetensors"))
    ]


def save_pth(model: ModelDescriptor, path: Path):
    torch.save(model.model.state_dict(), path)


def save_safetensors(model: ModelDescriptor, path: Path):
    save_file(model.model.state_dict(), path)


def save(path: Path):
    out_dir = CURRENT_DIR / "out"
    out_dir.mkdir(exist_ok=True)

    model = Lazy(lambda: ModelLoader().load_from_file(path))

    def save_file(out_path: Path, save_func: Callable[[ModelDescriptor, Path], None]):
        if not out_path.exists():
            print(f"Saving {out_path.name}...")
            try:
                save_func(model.value, out_path)
            except Exception:
                print(traceback.format_exc())
        else:
            print(f"{out_path.name} already exists, skipping.")

    save_file(out_dir / (path.stem + ".pth"), save_pth)
    save_file(out_dir / (path.stem + ".safetensors"), save_safetensors)


def main():
    print("Searching for files to convert...")
    files = get_files_to_convert()
    print(f"Found {len(files)} files to convert.")

    for i, file in enumerate(files):
        print(f"Converting {file.name} ({i+1}/{len(files)})")
        save(file)

    print("Done.")
    input("Press enter to exit.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An error occurred:")
        print(traceback.format_exc())
        input("Press enter to exit.")
        sys.exit(1)
