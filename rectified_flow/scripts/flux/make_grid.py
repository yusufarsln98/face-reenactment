"""Build a labelled 4x3 comparison grid from images that already exist on disk.

The grid never runs the model. It only reads image files that a previous run
wrote, places them in a fixed layout, and writes one file per cell size.

Layout, 4 columns by 3 rows:

    column 1  the video frame as the model saw it      faceswap_N_body_input
    column 2  the same frame, LoRA absent              faceswap_N_lora_off
    column 3  the same frame, LoRA attached            faceswap_N_lora_on
    column 4  the reference selfie, repeated per row   the original photo

Row order follows --rows, which lists source frame numbers.

The target column reads the original selfie, not the 256 px copy that the
pipeline saved, so it stays sharp at every cell size.

Every output filename contains "faceswap", as the project rule requires.

Run from the project root:

    fluxenv/bin/python scripts/flux/make_grid.py --cell 512
"""

from __future__ import annotations

import argparse
import os
import sys

from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BACKGROUND = (18, 18, 20)
TEXT = (240, 240, 240)
MUTED = (152, 152, 160)
RULE = (72, 72, 80)
HIGHLIGHT = (255, 206, 84)

GUTTER_RATIO = 0.30
PAD_RATIO = 0.045
HEADER_RATIO = 0.15


def metrics(cell):
    """Return gutter, padding, header height and margins for a square cell."""
    gutter = max(int(round(cell * GUTTER_RATIO)), 40)
    pad = max(int(round(cell * PAD_RATIO)), 4)
    header = max(int(round(cell * HEADER_RATIO)), 40)
    margin = pad * 2
    return gutter, pad, header, margin


def canvas_size(cell):
    """Return the exact canvas size for a square cell of `cell` pixels."""
    gutter, pad, header, margin = metrics(cell)
    width = cell * 4 + gutter + pad * 5 + margin
    height = cell * 3 + header + pad * 4 + margin
    return width, height, gutter, pad, header, margin


def fit_font(path, size):
    if not os.path.isfile(path):
        return ImageFont.load_default()
    return ImageFont.truetype(path, max(int(size), 9))


def centred(draw, x0, cell, y, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (cell - (box[2] - box[0])) // 2, y), text, font=font, fill=fill)


def build(rows, out_path, cell):
    """Write one grid file. Returns the exact canvas size."""
    width, height, gutter, pad, header, margin = canvas_size(cell)

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    f_head = fit_font(FONT_BOLD, cell * 0.050)
    f_sub = fit_font(FONT_REGULAR, cell * 0.032)
    f_row = fit_font(FONT_BOLD, cell * 0.044)
    f_seed = fit_font(FONT_REGULAR, cell * 0.030)

    columns = [
        ("INPUT", "video frame as the model saw it"),
        ("LoRA OFF", "adapter absent"),
        ("LoRA ON", "head swap applied"),
        ("TARGET", "reference selfie"),
    ]

    head_top = margin // 2
    rule_y = margin + header - pad // 2
    for index, (title, subtitle) in enumerate(columns):
        x0 = margin + gutter + pad + index * (cell + pad)
        centred(draw, x0, cell, head_top, title, f_head, TEXT)
        centred(draw, x0, cell, head_top + cell * 0.056, subtitle, f_sub, MUTED)
    draw.line([(margin, rule_y), (width - margin, rule_y)], fill=RULE, width=max(cell // 200, 1))

    for row, entry in enumerate(rows):
        top = margin + header + pad + row * (cell + pad)

        # Two separate lines. A single call with a newline would draw the
        # backslash and the n as literal characters.
        draw.text((margin, top + cell * 0.30), "FRAME", font=f_row, fill=TEXT)
        draw.text(
            (margin, top + cell * 0.30 + cell * 0.075),
            entry["label"],
            font=f_row,
            fill=TEXT,
        )
        draw.text(
            (margin, top + cell * 0.30 + cell * 0.155),
            entry["note"],
            font=f_seed,
            fill=MUTED,
        )
        draw.line(
            [(margin, top), (margin, top + cell)],
            fill=RULE,
            width=max(cell // 200, 1),
        )

        for column, path in enumerate(entry["images"]):
            x0 = margin + gutter + pad + column * (cell + pad)
            panel = Image.open(path).convert("RGB").resize(
                (cell, cell), Image.Resampling.LANCZOS
            )
            canvas.paste(panel, (x0, top))
            if column == 2:
                box = max(cell // 110, 2)
                draw.rectangle(
                    [x0 - box, top - box, x0 + cell + box, top + cell + box],
                    outline=HIGHLIGHT,
                    width=box,
                )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return canvas.size


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-dir", default="out/flux/faceswap_512x512")
    ap.add_argument("--target", default="IMG_20260817_111005.jpg")
    ap.add_argument("--rows", default="0,288,576")
    ap.add_argument(
        "--cell",
        action="append",
        default=None,
        help="cell size, repeatable. Accepts 512, 512x512 or 512.",
    )
    ap.add_argument("--out-dir", default="out/flux/grid")
    args = ap.parse_args()

    source_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.source_dir))
    target = os.path.abspath(os.path.join(PROJECT_ROOT, args.target))
    out_dir = os.path.abspath(os.path.join(PROJECT_ROOT, args.out_dir))

    if not os.path.isdir(source_dir):
        print(f"ERROR: no such directory: {source_dir}", file=sys.stderr)
        return 1
    if not os.path.isfile(target):
        print(f"ERROR: missing target photo: {target}", file=sys.stderr)
        return 1

    rows = []
    for index, token in enumerate(args.rows.split(",")):
        token = token.strip()
        stem = os.path.join(source_dir, f"faceswap_{index}_")
        needed = [stem + "body_input.png", stem + "lora_off.png", stem + "lora_on.png"]
        for path in needed:
            if not os.path.isfile(path):
                print(f"ERROR: row {index} is missing {path}", file=sys.stderr)
                return 1
        rows.append(
            {"label": token, "note": f"seed {index}", "images": needed + [target]}
        )

    cells = []
    for entry in args.cell or ["512x512"]:
        parts = [p for p in entry.replace("x", " ").replace(",", " ").split() if p]
        if len(parts) == 1:
            cells.append((int(parts[0]), int(parts[0])))
        elif len(parts) == 2:
            cells.append((int(parts[0]), int(parts[1])))
        else:
            print(f"ERROR: cannot read a cell size from {entry!r}", file=sys.stderr)
            return 1

    # Name the files after the source set, so the three runs never collide.
    tag = os.path.basename(source_dir.rstrip("/"))
    if tag.startswith("faceswap_"):
        tag = tag[len("faceswap_"):]

    print(f"[grid] source_dir={source_dir}")
    print(f"[grid] target={target}")
    print(f"[grid] rows={args.rows}")

    for width, height in cells:
        if width != height:
            print(f"ERROR: only square cells are supported, got {width}x{height}",
                  file=sys.stderr)
            return 1
        cell = width
        out_path = os.path.join(out_dir, f"faceswap_grid_{tag}_cell{cell}.png")
        size = build(rows, out_path, cell)
        print(
            f"[grid] {out_path}  canvas={size[0]}x{size[1]}  "
            f"cell={cell}x{cell}  {os.path.getsize(out_path)} bytes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
