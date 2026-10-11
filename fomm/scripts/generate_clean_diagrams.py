#!/usr/bin/env python3
import urllib.request
import os

OUTPUT_DIRS = [
    "notes/assets",
    "docs/slides/figures"
]

for d in OUTPUT_DIRS:
    os.makedirs(d, exist_ok=True)

# Minimalist, monochrome, zero-color, zero-complex-math diagrams
DIAGRAMS = {
    "diagram_standard_gan": """
digraph G {
    graph [rankdir=LR, dpi=300, bgcolor="white", fontname="Helvetica", pad="0.2", nodesep=0.45, ranksep=0.6];
    node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#111827", fontname="Helvetica", fontsize=11, margin="0.25,0.18", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, penwidth=1.4, color="#111827", arrowsize=0.85];

    subgraph cluster_gen {
        label = "GENERATOR";
        color = "#9ca3af";
        style = "dashed,rounded";
        bgcolor = "#f9fafb";
        fontcolor = "#111827";
        fontname = "Helvetica-Bold";
        fontsize = 11;
        margin = 14;

        Z [label="Random Noise (z)", fillcolor="#ffffff"];
        G [label="Generator (G)\n(Creates fake image)", fillcolor="#f3f4f6"];
    }

    subgraph cluster_disc {
        label = "DISCRIMINATOR";
        color = "#9ca3af";
        style = "dashed,rounded";
        bgcolor = "#f9fafb";
        fontcolor = "#111827";
        fontname = "Helvetica-Bold";
        fontsize = 11;
        margin = 14;

        Real [label="Real Face Image", fillcolor="#ffffff"];
        Fake [label="Generated Face", fillcolor="#ffffff"];
        D    [label="Discriminator (D)\n(Evaluator)", fillcolor="#f3f4f6"];
        Out  [label="Decision\n(Real or Fake)", fillcolor="#ffffff", penwidth=2.0];
    }

    Z -> G;
    G -> Fake;
    Real -> D;
    Fake -> D;
    D -> Out;
    D -> G [style="dashed", label="  Training Feedback  ", constraint=false];
}
""",

    "diagram_facial_motion_order": """
digraph G {
    graph [rankdir=LR, dpi=300, bgcolor="white", fontname="Helvetica", pad="0.2", nodesep=0.4, ranksep=0.65];
    node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#111827", fontname="Helvetica", fontsize=11, margin="0.25,0.18", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, penwidth=1.4, color="#111827", arrowsize=0.85];

    Root [label="Facial Keypoint (x, y)\n(Motion reference point)", fillcolor="#f3f4f6"];

    Order0 [label="0th-Order: Translation Only\n- Moves (x, y) coordinates\n- Stiff, robotic motion", fillcolor="#ffffff"];

    Order1 [label="1st-Order: Local Deformation (FOMM)\n- Models motion around points\n- Captures realistic facial stretch", fillcolor="#f3f4f6"];

    Rot   [label="Rotation\n(Head turns & tilts)", fillcolor="#ffffff"];
    Scale [label="Scaling\n(Mouth opening & closing)", fillcolor="#ffffff"];
    Shear [label="Shearing\n(Skin stretching & squinting)", fillcolor="#ffffff"];

    Root -> Order0;
    Root -> Order1;

    Order1 -> Rot;
    Order1 -> Scale;
    Order1 -> Shear;
}
""",

    "diagram_fomm_pipeline": """
digraph G {
    graph [rankdir=LR, dpi=300, bgcolor="white", fontname="Helvetica", pad="0.2", nodesep=0.35, ranksep=0.55];
    node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#111827", fontname="Helvetica", fontsize=11, margin="0.24,0.16", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, penwidth=1.4, color="#111827", arrowsize=0.85];

    In [label="Inputs\nSource Photo & Driving Video", fillcolor="#f3f4f6"];

    KP [label="1. Keypoint Detector\n(Extracts 10 motion points)", fillcolor="#ffffff"];

    Dense [label="2. Dense Motion Network\n(Predicts movement field)", fillcolor="#ffffff"];

    Flow [label="Optical Flow Grid (T)\n(Dense movement vectors)", fillcolor="#ffffff"];
    Occ  [label="Occlusion Mask (O)\n(Detects hidden areas)", fillcolor="#f3f4f6"];

    Warp [label="3. Feature Warping\n(Warps source face)", fillcolor="#ffffff"];

    Gen [label="4. Inpainting Generator\n(Fills in teeth & mouth)", fillcolor="#f3f4f6"];

    Out [label="Synthesized Output\n(Final reenacted video frame)", fillcolor="#ffffff", penwidth=2.0];

    In -> KP;
    KP -> Dense;
    Dense -> Flow;
    Dense -> Occ;

    Flow -> Warp;
    Warp -> Gen;
    Occ -> Gen;
    Gen -> Out;
}
""",

    "diagram_occlusion_inpainting": """
digraph G {
    graph [rankdir=LR, dpi=300, bgcolor="white", fontname="Helvetica", pad="0.2", nodesep=0.4, ranksep=0.6];
    node [shape=box, style="rounded,filled", fillcolor="#ffffff", color="#111827", fontname="Helvetica", fontsize=11, margin="0.25,0.16", penwidth=1.5];
    edge [fontname="Helvetica", fontsize=10, penwidth=1.4, color="#111827", arrowsize=0.85];

    subgraph cluster_inputs {
        label = "INPUTS";
        color = "#9ca3af";
        style = "dashed,rounded";
        bgcolor = "#f9fafb";
        fontcolor = "#111827";
        fontname = "Helvetica-Bold";
        fontsize = 11;
        margin = 14;

        Warped [label="Warped Source Features\n(Visible face parts)", fillcolor="#ffffff"];
        Occ    [label="Occlusion Mask (O)\n(Flags unseen parts, e.g. open mouth)", fillcolor="#f3f4f6"];
        Prior  [label="Learned Face Patterns\n(Knowledge of what teeth/eyes look like)", fillcolor="#ffffff"];
    }

    Gen [label="Inpainting Generator\n(Combines visible parts +\ninpaints unseen regions)", fillcolor="#f3f4f6"];

    Out [label="Synthesized Output\n(Final photorealistic face)", fillcolor="#ffffff", penwidth=2.0];

    Warped -> Gen;
    Occ -> Gen;
    Prior -> Gen;
    Gen -> Out;
}
"""
}

def generate_diagram(name, dot_source):
    print(f"Generating {name}...")
    headers = {"Content-Type": "text/plain; charset=utf-8", "User-Agent": "Mozilla/5.0"}
    
    req_png = urllib.request.Request("https://kroki.io/graphviz/png", data=dot_source.encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req_png) as resp:
        png_data = resp.read()

    req_svg = urllib.request.Request("https://kroki.io/graphviz/svg", data=dot_source.encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req_svg) as resp:
        svg_data = resp.read()

    for out_dir in OUTPUT_DIRS:
        png_path = os.path.join(out_dir, f"{name}.png")
        svg_path = os.path.join(out_dir, f"{name}.svg")
        with open(png_path, "wb") as f:
            f.write(png_data)
        with open(svg_path, "wb") as f:
            f.write(svg_data)
        print(f"  Saved {png_path} ({len(png_data)} bytes)")

for name, dot in DIAGRAMS.items():
    generate_diagram(name, dot)

print("All monochrome diagrams successfully generated!")
