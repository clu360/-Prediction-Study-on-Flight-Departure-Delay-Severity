"""Request complete local bytes from the Windows cloud provider, then verify SHA.

Uses Microsoft's CfHydratePlaceholder API (no logical file-content changes):
https://learn.microsoft.com/windows/win32/api/cfapi/nf-cfapi-cfhydrateplaceholder
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import ctypes
from ctypes import wintypes
import os

from flight_delay.experiments.config import file_hash
from flight_delay.experiments.results import read_json


def hydrate(path):
    if os.name != "nt":
        return
    # A verbatim current directory can leave '..' unnormalized. Pass the
    # ordinary normalized absolute filename to the Windows cloud API.
    filename = os.path.normpath(str(path.resolve()).removeprefix("\\\\?\\"))
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateFileW(filename, 0x80, 7, None, 3, 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        api = ctypes.WinDLL("CldApi.dll", use_last_error=True)
        api.CfHydratePlaceholder.argtypes = [
            wintypes.HANDLE,
            ctypes.c_longlong,
            ctypes.c_longlong,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        api.CfHydratePlaceholder.restype = ctypes.c_long
        result = api.CfHydratePlaceholder(handle, 0, -1, 0, None)
        if result and (os.stat(filename).st_file_attributes & 0x400):
            raise OSError(
                f"Cloud hydration failed for {path.name}: {result & 0xFFFFFFFF:#x}"
            )
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("../Data/Raw Datasets 2025")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    entries = read_json(root / "data/processed/audit/input_manifest.json")["inputs"]

    def verify(entry):
        path = args.raw_dir / entry["filename"]
        hydrate(path)
        if file_hash(path) != entry["sha256"]:
            raise ValueError(
                f"Raw source differs from original validated bytes: {path.name}"
            )
        return path.name

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(verify, entry) for entry in entries]):
            print(
                "Hydrated and original SHA-256 verified:", future.result(), flush=True
            )


if __name__ == "__main__":
    main()
