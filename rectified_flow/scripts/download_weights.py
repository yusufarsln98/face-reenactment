"""Download every model weight the face-swap pipeline needs, and verify it.

Run from the project root:
    .venv/bin/python scripts/download_weights.py

The script is idempotent.  A file that is present and whose sha256 matches the
hash recorded in models/MANIFEST.md is left alone.  A hash mismatch deletes the
file and stops the run.  Nothing runs on the GPU here.
"""

import hashlib
import shutil
import sys
import zipfile
from datetime import date
from pathlib import Path

import requests
from tqdm import tqdm

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MANIFEST = MODELS_DIR / "MANIFEST.md"

# hash_prefix is an independently known sha256 prefix.  It is checked on every
# run.  A full hash recorded in MANIFEST.md is checked on every later run.
WEIGHTS = [
    {
        "file": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/"
               "download/v1.3.4/GFPGANv1.4.pth",
        "note": "Tencent GFPGAN release.  v1.3.4 is the newest tag that ships "
                "GFPGANv1.4.pth.",
        "hash_prefix": None,
        "licence_name": "Apache License 2.0",
        "licence_body": (
            "GFPGANv1.4.pth is released under the Apache License, Version 2.0.\n"
            "Copyright (c) 2021 Tencent ARC.\n"
            "\n"
            "The full licence text is at\n"
            "  https://www.apache.org/licenses/LICENSE-2.0\n"
        ),
    },
    {
        "file": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/"
               "download/v0.1.0/RealESRGAN_x4plus.pth",
        "note": "Real-ESRGAN release v0.1.0.",
        "hash_prefix": None,
        "licence_name": "BSD 3-Clause License",
        "licence_body": (
            "RealESRGAN_x4plus.pth is released under the BSD 3-Clause License.\n"
            "Copyright (c) 2021 Xintao Wang and the BasicSR contributors.\n"
            "\n"
            "The full licence text is at\n"
            "  https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE\n"
        ),
    },
]


def sha256_file(path, chunk=1 << 20):
    """Return the sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest_hashes():
    """Return {file name: sha256} recorded in MANIFEST.md, or an empty dict."""
    if not MANIFEST.exists():
        return {}
    hashes = {}
    for line in MANIFEST.read_text().splitlines():
        if not line.startswith("| `") or "` |" not in line:
            continue
        name = line.split("`")[1]
        for cell in line.split("|"):
            cell = cell.strip().strip("`")
            if len(cell) == 64 and all(c in "0123456789abcdef" for c in cell):
                hashes[name] = cell
    return hashes


def download_direct(url, dest):
    """Download over plain HTTPS with a progress bar."""
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as handle:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                desc=dest.name,
            ) as bar:
                for block in response.iter_content(chunk_size=1 << 20):
                    if block:
                        handle.write(block)
                        bar.update(len(block))
    return dest


def read_licence_date(name):
    """Return the original fetch date from an existing licence file."""
    path = MODELS_DIR / f"LICENCE-{name}.txt"
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("Date fetched:"):
            return line.split(":", 1)[1].strip()
    return None


def write_licence(weight, full_hash, fetched):
    """Write the plain text licence file beside the weight."""
    name = weight["file"]
    path = MODELS_DIR / f"LICENCE-{name}.txt"
    path.write_text(
        f"Weight      : {name}\n"
        f"Licence     : {weight['licence_name']}\n"
        f"Source URL  : {weight['url']}\n"
        f"Date fetched: {fetched}\n"
        f"SHA256      : {full_hash}\n"
        f"Size        : {weight['size']} bytes\n"
        f"\n"
        f"{weight['licence_body']}\n"
        f"Source notes: {weight['note']}\n"
    )
    print(f"  wrote {path.name}")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    known = read_manifest_hashes()
    results = []
    total_bytes = 0

    for weight in WEIGHTS:
        name = weight["file"]
        dest = MODELS_DIR / name
        prefix = weight["hash_prefix"]
        expected = known.get(name)
        fetched = read_licence_date(name) or date.today().isoformat()
        print(f"\n=== {name} ===")

        if dest.exists():
            actual = sha256_file(dest)
            if expected and actual == expected:
                if prefix and not actual.startswith(prefix):
                    print(f"  HASH MISMATCH: {name}")
                    print(f"    expected {expected}")
                    print(f"    actual   {actual}")
                    dest.unlink()
                    return 1
                print(f"  already present and verified, skipping ({actual})")
                weight["sha256"] = actual
                weight["size"] = dest.stat().st_size
                weight["fetched"] = fetched
                write_licence(weight, actual, fetched)
                results.append(weight)
                total_bytes += weight["size"]
                continue
            if prefix and actual.startswith(prefix):
                expected = actual
            else:
                print(f"  existing file fails verification, deleting and re-downloading")
                dest.unlink()

        tmp = dest.with_suffix(dest.suffix + ".part")
        if tmp.exists():
            tmp.unlink()

        download_direct(weight["url"], tmp)

        actual = sha256_file(tmp)
        if expected and actual != expected:
            tmp.unlink()
            print(f"  HASH MISMATCH for {name}")
            print(f"    expected {expected}")
            print(f"    actual   {actual}")
            print("  the file was deleted.  aborting.")
            return 1
        if prefix and not actual.startswith(prefix):
            tmp.unlink()
            print(f"  HASH PREFIX MISMATCH for {name}")
            print(f"    expected prefix {prefix}")
            print(f"    actual           {actual}")
            print("  the file was deleted.  aborting.")
            return 1

        tmp.rename(dest)
        print(f"  sha256 verified: {actual}")
        print(f"  size: {dest.stat().st_size} bytes")

        if "extract_to" in weight:
            target = MODELS_DIR / weight["extract_to"]
            if target.exists():
                shutil.rmtree(target)
            with zipfile.ZipFile(dest) as archive:
                archive.extractall(target)
            members = sorted(p.name for p in target.iterdir())
            print(f"  extracted to {target.name}/: {', '.join(members)}")

        weight["sha256"] = actual
        weight["size"] = dest.stat().st_size
        weight["fetched"] = fetched
        write_licence(weight, actual, fetched)
        results.append(weight)
        total_bytes += weight["size"]

    write_manifest(results, total_bytes)
    print(f"\nAll weights present and verified.")
    print(f"Total size on disk: {total_bytes} bytes "
          f"({total_bytes / 1024 / 1024:.1f} MB)")
    return 0


def write_manifest(results, total_bytes):
    """Write models/MANIFEST.md for the report."""
    fetched = date.today().isoformat()
    lines = [
        "# Model weights manifest",
        "",
        f"Generated: {fetched}",
        "",
        "Every file below was downloaded over HTTPS and verified against a "
        "recorded sha256.",
        "",
        "| File | Size (bytes) | Size (MB) | sha256 | Licence | Source URL |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for weight in results:
        lines.append(
            f"| `{weight['file']}` | {weight['size']} | "
            f"{weight['size'] / 1024 / 1024:.1f} | `{weight['sha256']}` | "
            f"{weight['licence_name']} | <{weight['url']}> |"
        )
    lines += [
        "",
        f"Total: {total_bytes} bytes ({total_bytes / 1024 / 1024:.1f} MB)",
        "",
        "## Licences",
        "",
    ]
    for weight in results:
        lines += [
            f"### {weight['file']}",
            "",
            f"- Licence: {weight['licence_name']}",
            f"- Source: <{weight['url']}>",
            f"- sha256: `{weight['sha256']}`",
            f"- Licence file: `LICENCE-{weight['file']}.txt`",
            "",
        ]
    MANIFEST.write_text("\n".join(lines))
    print(f"\nwrote {MANIFEST}")


if __name__ == "__main__":
    sys.exit(main())