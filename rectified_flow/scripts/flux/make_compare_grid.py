"""Build a labelled 5x1 comparison grid per frame, from images on disk.

The grid never runs the model. It only reads image files that a previous run
wrote, places them in a fixed layout, and writes one PNG per frame. It loads no
weight and starts no inference.

Layout, 5 columns by 1 row:

    column 1  the body crop the model saw          faceswap_body_N
    column 2  the frame, v1 LoRA                   faceswap_v1_frame_N
    column 3  the frame, v1.1 LoRA, prompt A       faceswap_v11_pa_frame_N
    column 4  the frame, v1.1 LoRA, prompt B       faceswap_v11_pb_frame_N
    column 5  the reference portrait               the original photo

The row label holds the frame number, so each file is self-describing.

Column 5 reads the original portrait, not the 256 px copy that the pipeline
saved. That photo is scaled to fit the cell and is never cropped, so the head
stays whole whatever the aspect ratio of the source file.

Every output filename contains "faceswap", as the project rule requires.

Run from the project root:

    fluxenv/bin/python scripts/flux/make_compare_grid.py
"""

from __future__ import annotations

import os
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

BACKGROUND = (18, 18, 20)
TEXT = (240, 240, 240)
MUTED = (152, 152, 160)
RULE = (72, 72, 80)

GUTTER_RATIO = 0.30
PAD_RATIO = 0.045
HEADER_RATIO = 0.15

# The cell size of every output file.
CELL = 512

# The three source frames, in the order the rows appear.
FRAMES = ("0175", "0222", "0270")

# Input directories, relative to the project root.
INPUT_DIR = "out/faceswap_version_compare/inputs"
V1_DIR = "out/faceswap_version_compare/v1_fluxklein_4b"
V11_DIR = "out/faceswap_version_compare/v1_1_fluxklein_4b"

# Output directory and the name of each file it holds.
OUT_DIR = "out/faceswap_version_compare/grid"

# Input file names. The frame is a four digit number.
INPUT_NAME = "faceswap_body_{frame}.png"
V1_NAME = "faceswap_v1_frame_{frame}.png"
V11_NAME = "faceswap_v11_p{letter}_frame_{frame}.png"

# The reference portrait. Read only.
REFERENCE_NAME = "IMG_20260817_111005.jpg"

# Output file name. It holds "faceswap", as the project rule requires.
OUT_NAME = "faceswap_compare_grid_{frame}.png"

# Column titles and subtitles, left to right.
COLUMNS = (
    ("INPUT", "body crop the model saw"),
    ("V1 LORA", "recorded v1 adapter"),
    ("V1.1 PROMPT A", "new adapter, prompt a"),
    ("V1.1 PROMPT B", "new adapter, prompt b"),
    ("REFERENCE", "original portrait, fit not cropped"),
)

# The muted line under the frame number in the row label.
NOTE = "recorded video frame"


def under_root(*parts):
    """Return an absolute path below the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


def metrics(cell):
    """Return gutter, padding, header height and margins for a square cell."""
    gutter = max(int(round(cell * GUTTER_RATIO)), 40)
    pad = max(int(round(cell * PAD_RATIO)), 4)
    header = max(int(round(cell * HEADER_RATIO)), 40)
    margin = pad * 2
    return gutter, pad, header, margin


def canvas_size(cell, columns, rows):
    """Return the exact canvas size for a square cell of `cell` pixels."""
    gutter, pad, header, margin = metrics(cell)
    width = cell * columns + gutter + pad * (columns + 1) + margin
    height = header + margin + rows * (cell + pad) + pad
    return width, height, gutter, pad, header, margin


def fit_font(path, size):
    if not os.path.isfile(path):
        return ImageFont.load_default()
    return ImageFont.truetype(path, max(int(size), 9))


def centred(draw, x0, cell, y, text, font, fill):
    box = draw.textbbox((0, 0), text, font=font)
    draw.text((x0 + (cell - (box[2] - box[0])) // 2, y), text, font=font, fill=fill)


def fit_cell(path, cell):
    """Scale one image to fit a square cell. The image is never cropped."""
    with Image.open(path) as handle:
        source = handle.convert("RGB")
    fitted = ImageOps.contain(source, (cell, cell), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (cell, cell), BACKGROUND)
    panel.paste(fitted, ((cell - fitted.width) // 2, (cell - fitted.height) // 2))
    return panel


def row_images(frame):
    """Return the five image paths of one frame, in column order."""
    return [
        under_root(INPUT_DIR, INPUT_NAME.format(frame=frame)),
        under_root(V1_DIR, V1_NAME.format(frame=frame)),
        under_root(V11_DIR, V11_NAME.format(letter="a", frame=frame)),
        under_root(V11_DIR, V11_NAME.format(letter="b", frame=frame)),
        under_root(REFERENCE_NAME),
    ]


def build(entry, out_path, cell):
    """Write one grid file. Returns the exact canvas size."""
    columns = len(COLUMNS)
    width, height, gutter, pad, header, margin = canvas_size(cell, columns, 1)

    canvas = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(canvas)

    f_head = fit_font(FONT_BOLD, cell * 0.050)
    f_sub = fit_font(FONT_REGULAR, cell * 0.032)
    f_row = fit_font(FONT_BOLD, cell * 0.044)
    f_note = fit_font(FONT_REGULAR, cell * 0.030)

    head_top = margin // 2
    rule_y = margin + header - pad // 2
    for index, (title, subtitle) in enumerate(COLUMNS):
        x0 = margin + gutter + pad + index * (cell + pad)
        centred(draw, x0, cell, head_top, title, f_head, TEXT)
        centred(draw, x0, cell, head_top + cell * 0.056, subtitle, f_sub, MUTED)
    draw.line(
        [(margin, rule_y), (width - margin, rule_y)],
        fill=RULE,
        width=max(cell // 200, 1),
    )

    top = margin + header + pad

    # Two separate lines. A single call with a newline would draw the
    # backslash and the n as literal characters.
    draw.text((margin, top + cell * 0.30), "FRAME", font=f_row, fill=TEXT)
    draw.text(
        (margin, top + cell * 0.30 + cell * 0.075),
        entry["frame"],
        font=f_row,
        fill=TEXT,
    )
    draw.text(
        (margin, top + cell * 0.30 + cell * 0.155),
        entry["note"],
        font=f_note,
        fill=MUTED,
    )
    draw.line(
        [(margin, top), (margin, top + cell)],
        fill=RULE,
        width=max(cell // 200, 1),
    )

    for column, path in enumerate(entry["images"]):
        x0 = margin + gutter + pad + column * (cell + pad)
        canvas.paste(fit_cell(path, cell), (x0, top))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return canvas.size


def main():
    entries = []
    for frame in FRAMES:
        images = row_images(frame)
        missing = [path for path in images if not os.path.isfile(path)]
        if missing:
            for path in missing:
                print(f"ERROR: frame {frame} is missing {path}", file=sys.stderr)
            return 1
        entries.append({"frame": frame, "note": NOTE, "images": images})

    expected = canvas_size(CELL, len(COLUMNS), 1)[:2]

    print(f"[compare-grid] frames={','.join(FRAMES)}")
    print(
        f"[compare-grid] cell={CELL}x{CELL}  columns={len(COLUMNS)}  "
        f"canvas={expected[0]}x{expected[1]}"
    )

    for entry in entries:
        out_path = under_root(OUT_DIR, OUT_NAME.format(frame=entry["frame"]))
        size = build(entry, out_path, CELL)
        if size != expected:
            print(
                f"ERROR: {out_path} drew {size[0]}x{size[1]}, "
                f"expected {expected[0]}x{expected[1]}",
                file=sys.stderr,
            )
            return 1

        # Read the file back, so the printed size is the size on disk.
        with Image.open(out_path) as check:
            written = check.size
        if written != expected:
            print(
                f"ERROR: {out_path} wrote {written[0]}x{written[1]}, "
                f"expected {expected[0]}x{expected[1]}",
                file=sys.stderr,
            )
            return 1

        print(
            f"[compare-grid] {out_path}  {written[0]}x{written[1]}  "
            f"{os.path.getsize(out_path)} bytes"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
