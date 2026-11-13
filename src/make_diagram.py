"""Draw the pipeline diagram (baseline and proposed design labelled, human-centered mechanism marked).

    python -m src.make_diagram      # writes report/figures/pipeline.pdf and pipeline.png

Pure drawing: no data, no model calls. Values (m, tau, k, model, classifier, calibration) are read from the frozen config, so the
figure cannot drift from the code. Colours are the Okabe-Ito blue and orange (colour-blind safe); shape and caption also encode the
design, so the figure still reads in greyscale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

INK, MUTED = "#1B1F23", "#4A5560"
GREY_FC, GREY_EC = "#F3F5F7", "#6B7682"
BLUE, BLUE_DARK, BLUE_TINT, BLUE_BOX = "#0072B2", "#005A8D", "#F2F8FC", "#DCEBF7"
ORANGE_DARK, ORANGE_TINT, ORANGE_BOX = "#9A5C00", "#FFF9EE", "#FCE8C3"
NL = "\n"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, x, y, w, h, title, body, fc, ec, badge=None, badge_fc=MUTED, lw=1.0, title_fs=7.3, body_fs=6.1, bold_body=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.10", fc=fc, ec=ec, lw=lw, zorder=2))
    top = y + h - 0.2
    ax.text(x + w / 2, top, title, ha="center", va="top", fontsize=title_fs, weight="bold", color=INK, zorder=4)
    body_top = top - 0.30
    ax.text(x + w / 2, (y + body_top) / 2 + 0.02, body, ha="center", va="center", fontsize=body_fs, color=INK, linespacing=1.35,
            weight="bold" if bold_body else "normal", zorder=4)
    if badge is not None:
        ax.add_patch(Circle((x + 0.02, y + h - 0.02), 0.23, fc=badge_fc, ec="white", lw=1.2, zorder=5))
        ax.text(x + 0.02, y + h - 0.02, str(badge), ha="center", va="center", fontsize=6.6, weight="bold", color="white", zorder=6)


def line(ax, xs, ys, color=GREY_EC, lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="butt", zorder=1)


def arrow(ax, p, q, color=GREY_EC, lw=1.2):
    ax.annotate("", xy=q, xytext=p, zorder=3,
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", color=color, lw=lw, shrinkA=0, shrinkB=0))


def lane(ax, x, w, edge, tint, ls="-"):
    ax.add_patch(FancyBboxPatch((x, 0.05), w, 3.55, boxstyle="round,pad=0,rounding_size=0.12", fc=tint, ec=edge, lw=1.5, ls=ls, zorder=0))


def footer(ax, x, w, color, text):
    ax.add_patch(Rectangle((x + 0.12, 0.13), w - 0.24, 0.5, fc=color, ec="none", zorder=2))
    ax.text(x + w / 2, 0.38, text, ha="center", va="center", fontsize=7.0, weight="bold", color="white", zorder=4)


def main() -> int:
    cfg = load_config()
    k, m, tau = cfg["retrieval"]["k"], cfg["state"]["m"], cfg["abstention"]["tau"]
    clf = cfg["classifier"]["type"].replace("_", " ")
    cal = cfg["calibration"]["type"]
    model = cfg["model"]["tag"]

    fig, ax = plt.subplots(figsize=(7.6, 5.9))
    ax.set_xlim(-0.2, 12.2)
    ax.set_ylim(-0.1, 8.5)
    ax.axis("off")

    # ---- section label ---------------------------------------------------------------------------------
    ax.text(0.0, 8.2, "SHARED PIPELINE", fontsize=7.4, weight="bold", color=MUTED, ha="left", va="center")
    ax.text(2.25, 8.2, "identical for the baseline and the proposed design; model-level only, no users", fontsize=6.6, color=MUTED,
            ha="left", va="center")
    line(ax, [0.0, 12.0], [7.98, 7.98], color="#C9D0D6", lw=0.8)

    # ---- row A: shared stages --------------------------------------------------------------------------
    y, h, w = 6.15, 1.5, 2.15
    xs = [0.0, 2.45, 4.9, 7.35, 9.8]
    stages = [
        (None, "Claim", "AVeriTeC claim" + NL + "with its claim date"),
        (1, "Decompose", model + NL + "5 true-frame +" + NL + "5 false-frame" + NL + "yes/no questions"),
        (2, "Retrieve", "BM25, sentence level," + NL + "over the claim's" + NL + "knowledge store;" + NL + f"top-{k} per question"),
        (3, "Answer", "Yes / No / Unverified" + NL + "from the snippets only" + NL + "(question frame hidden)"),
        (4, "Six counts", "(Yes, No, Unverified)" + NL + "x (true, false frame)"),
    ]
    for x, (badge, title, body) in zip(xs, stages):
        box(ax, x, y, w, h, title, body, GREY_FC, GREY_EC, badge=badge)
    for a_, b_ in zip(xs[:-1], xs[1:]):
        arrow(ax, (a_ + w, y + h / 2), (b_, y + h / 2))

    # ---- row B: classifier (shared) and the evidence-state rule (proposed mechanism) -----------------
    yb, hb = 4.2, 1.3
    box(ax, 0.3, yb, 6.7, hb, "Numeric classifier", clf.capitalize() + " + " + cal + " calibration" + NL +
        "-> verdict v (S / R / C / N) and calibrated confidence c", GREY_FC, GREY_EC, badge=5)
    box(ax, 8.35, yb, 3.65, hb, "Evidence state", f"rule with m = {m}:" + NL + "Supports / Refutes /" + NL + "Mixed / Insufficient",
        ORANGE_BOX, ORANGE_DARK, badge=6, badge_fc=ORANGE_DARK, lw=1.8, bold_body=True, body_fs=6.2)
    # counts -> classifier (elbow) and counts -> state (straight)
    line(ax, [10.0, 10.0], [y, 5.85])
    line(ax, [10.0, 3.65], [5.85, 5.85])
    arrow(ax, (3.65, 5.85), (3.65, yb + hb))
    arrow(ax, (11.2, y), (11.2, yb + hb), color=ORANGE_DARK, lw=1.6)

    # ---- lanes -------------------------------------------------------------------------------------------
    lane(ax, 0.0, 5.0, BLUE, BLUE_TINT, ls=(0, (5, 3)))
    lane(ax, 5.3, 6.7, ORANGE_DARK, ORANGE_TINT)
    box(ax, 0.2, 1.9, 4.6, 1.45, "Verdict shown for every claim", "never withheld," + NL + "no evidence state shown", BLUE_BOX, BLUE,
        body_fs=6.6)
    box(ax, 0.2, 0.74, 4.6, 1.0, "Evaluated with", "accuracy, macro-F1, Brier, ECE;" + NL + "unsafe error = wrong decisive S / R verdict",
        "white", BLUE, body_fs=6.0, title_fs=6.8)
    footer(ax, 0.0, 5.0, BLUE_DARK, "BASELINE  |  P0: forced verdict")

    box(ax, 5.5, 1.9, 6.3, 1.45, "Human-centered mechanism: abstain and explain",
        f"Withhold the verdict if confidence c < tau = {tau:g}" + NL + "OR the state is Mixed / Insufficient;" + NL +
        "otherwise show the verdict together with its state", ORANGE_BOX, ORANGE_DARK, lw=1.8, body_fs=6.4, bold_body=True)
    box(ax, 5.5, 0.74, 6.3, 1.0, "Evaluated with",
        "coverage, selective risk, unsafe-error rate, AURC;" + NL + "state information value; mechanism check (P2 vs P4a)",
        "white", ORANGE_DARK, body_fs=6.0, title_fs=6.8)
    footer(ax, 5.3, 6.7, ORANGE_DARK, "PROPOSED  |  P3: state + calibrated abstention")

    # ---- arrows into the lanes (straight, orthogonal) ------------------------------------------------
    arrow(ax, (2.6, yb), (2.6, 3.35), color=BLUE, lw=1.6)
    arrow(ax, (6.2, yb), (6.2, 3.35), color=ORANGE_DARK, lw=1.6)
    arrow(ax, (10.0, yb), (10.0, 3.35), color=ORANGE_DARK, lw=1.6)
    ax.text(2.75, 3.88, "v", fontsize=7, color=BLUE_DARK, weight="bold", ha="left", va="center")
    ax.text(6.35, 3.88, "v, c", fontsize=7, color=ORANGE_DARK, weight="bold", ha="left", va="center")
    ax.text(10.15, 3.88, "state", fontsize=7, color=ORANGE_DARK, weight="bold", ha="left", va="center")

    out = ROOT / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pipeline.pdf", bbox_inches="tight", pad_inches=0.05, metadata={"CreationDate": None})
    fig.savefig(out / "pipeline.png", dpi=220, bbox_inches="tight", pad_inches=0.05)
    print("wrote", out / "pipeline.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
