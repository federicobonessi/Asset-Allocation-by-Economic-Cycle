"""
Asset Allocation by Economic Cycle
=====================================
A macro-driven framework that maps optimal asset allocation
across the four phases of the economic cycle:
Expansion, Peak, Contraction, and Recovery.

Simulates portfolio performance historically by phase,
scores current positioning, and produces a full allocation report.

Author: Federico Bonessi | The Meridian Playbook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
DARK_BG = "#0d1117"
GOLD    = "#c9a84c"
WHITE   = "#e6edf3"
GREY    = "#30363d"
MID     = "#8b949e"
RED     = "#f85149"
GREEN   = "#3fb950"
BLUE    = "#58a6ff"
ORANGE  = "#ffa657"
PURPLE  = "#a371f7"

PHASE_COLORS = {
    "Expansion":   "#3fb950",
    "Peak":        "#ffa657",
    "Contraction": "#f85149",
    "Recovery":    "#58a6ff",
}

# ─────────────────────────────────────────────
# ASSET UNIVERSE
# ─────────────────────────────────────────────

ASSETS = [
    "Global Equities",
    "Emerging Markets",
    "Real Estate",
    "Commodities",
    "Investment Grade Bonds",
    "High Yield Bonds",
    "Government Bonds",
    "Gold",
    "Cash",
    "Private Equity",
]

# ─────────────────────────────────────────────
# CYCLE FRAMEWORK
# ─────────────────────────────────────────────

# Optimal allocation (%) per phase — research-based
# Rows = assets, Cols = [Expansion, Peak, Contraction, Recovery]
ALLOCATIONS = pd.DataFrame({
    "Expansion":   [30, 10, 10,  5,  5,  8,  2,  5,  2, 23],
    "Peak":        [20,  5,  8, 12,  8,  5,  5, 10,  7, 20],
    "Contraction": [ 8,  2,  3,  3, 20,  2, 28, 15, 15,  4],
    "Recovery":    [25,  8,  8,  7,  8,  8,  5,  8,  3, 20],
}, index=ASSETS)

# Verify all columns sum to 100
assert all(ALLOCATIONS.sum() == 100), "Allocations must sum to 100 per phase"

# ─────────────────────────────────────────────
# MACRO INDICATORS PER PHASE
# ─────────────────────────────────────────────

MACRO = {
    "Expansion": {
        "GDP Growth":       "Accelerating (2.5%+)",
        "Inflation":        "Rising moderately",
        "Central Bank":     "Neutral to tightening",
        "Credit Spreads":   "Tightening",
        "Yield Curve":      "Steepening",
        "Corporate Earnings":"Strong, beating estimates",
        "signal_color":     GREEN,
    },
    "Peak": {
        "GDP Growth":       "Slowing from peak",
        "Inflation":        "High, persistent",
        "Central Bank":     "Actively tightening",
        "Credit Spreads":   "Widening",
        "Yield Curve":      "Flattening / inverting",
        "Corporate Earnings":"Margins under pressure",
        "signal_color":     ORANGE,
    },
    "Contraction": {
        "GDP Growth":       "Negative (recession)",
        "Inflation":        "Falling",
        "Central Bank":     "Cutting rates",
        "Credit Spreads":   "Sharply wide",
        "Yield Curve":      "Deeply inverted",
        "Corporate Earnings":"Declining, misses",
        "signal_color":     RED,
    },
    "Recovery": {
        "GDP Growth":       "Turning positive",
        "Inflation":        "Low, stable",
        "Central Bank":     "Accommodative",
        "Credit Spreads":   "Beginning to tighten",
        "Yield Curve":      "Re-steepening",
        "Corporate Earnings":"Recovering, surprises",
        "signal_color":     BLUE,
    },
}

# ─────────────────────────────────────────────
# HISTORICAL PERFORMANCE BY PHASE (simulated)
# Based on stylised academic literature averages
# ─────────────────────────────────────────────

PERF = pd.DataFrame({
    "Expansion":   [14.2,  18.5,  9.3,  11.2,   3.1,   9.8,   1.2,   2.1,   3.8,  22.0],
    "Peak":        [ 5.1,   2.3,  4.2,  14.8,   4.5,   3.2,   4.8,   8.9,   4.2,   8.5],
    "Contraction": [-18.2, -25.1, -12.3, -8.5,   8.2,  -9.8,  15.3,  14.2,   3.5, -22.0],
    "Recovery":    [22.1,  28.4,  12.5,   8.2,   5.5,  14.2,   2.1,   3.5,   2.8,  18.5],
}, index=ASSETS)  # annualised returns (%)

# Volatility by phase
VOL = pd.DataFrame({
    "Expansion":   [14.0, 20.0, 12.0, 18.0,  4.0,  9.0,  3.5,  12.0,  0.5, 18.0],
    "Peak":        [16.0, 22.0, 14.0, 22.0,  5.0, 11.0,  4.0,  14.0,  0.5, 20.0],
    "Contraction": [24.0, 30.0, 20.0, 25.0,  6.0, 18.0,  5.0,  18.0,  0.5, 28.0],
    "Recovery":    [18.0, 24.0, 15.0, 20.0,  5.0, 12.0,  4.0,  14.0,  0.5, 22.0],
}, index=ASSETS)

# ─────────────────────────────────────────────
# CURRENT PHASE ASSESSMENT
# ─────────────────────────────────────────────

CURRENT_PHASE = "Recovery"   # <-- change this to model different phases

CURRENT_PORTFOLIO = pd.Series({
    "Global Equities":         28,
    "Emerging Markets":         6,
    "Real Estate":              8,
    "Commodities":              4,
    "Investment Grade Bonds":  10,
    "High Yield Bonds":         6,
    "Government Bonds":         8,
    "Gold":                     7,
    "Cash":                     5,
    "Private Equity":          18,
})

# ─────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────

def compute_portfolio_metrics(weights_pct: pd.Series, phase: str) -> dict:
    """Compute expected return, vol, Sharpe for a portfolio in a given phase."""
    w   = weights_pct / 100
    ret = (w * PERF[phase]).sum()
    vol = np.sqrt((w ** 2 * VOL[phase] ** 2).sum())  # simplified (no corr)
    sharpe = ret / vol if vol > 0 else 0
    return {"return": ret, "vol": vol, "sharpe": sharpe}


def alignment_score(current: pd.Series, optimal: pd.Series) -> float:
    """
    Score 0-100 measuring how aligned the current portfolio is
    with the optimal allocation for the current phase.
    Lower deviation = higher score.
    """
    diff = (current - optimal).abs().sum()   # max possible = 200 (fully misaligned)
    return round(max(0, 100 - diff / 2), 1)


def compute_transition_shifts(from_phase: str, to_phase: str) -> pd.Series:
    """Returns the weight delta moving from one phase allocation to another."""
    return ALLOCATIONS[to_phase] - ALLOCATIONS[from_phase]


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def make_report(current_phase, current_portfolio):
    phases = ["Expansion", "Peak", "Contraction", "Recovery"]

    fig = plt.figure(figsize=(22, 28), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.48, wspace=0.35)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.spines[:].set_color(GREY)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(WHITE)
        if title:
            ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold", pad=10)

    pct_f = FuncFormatter(lambda x, _: f"{x:.0f}%")
    pct_f2 = FuncFormatter(lambda x, _: f"{x:.0f}%")

    opt = ALLOCATIONS[current_phase]
    score = alignment_score(current_portfolio, opt)
    cur_metrics = compute_portfolio_metrics(current_portfolio, current_phase)
    opt_metrics = compute_portfolio_metrics(opt, current_phase)

    # ── TITLE
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(DARK_BG); ax0.axis("off")
    ax0.text(0.5, 0.82, "ASSET ALLOCATION BY ECONOMIC CYCLE",
             ha="center", color=GOLD, fontsize=22, fontweight="bold",
             transform=ax0.transAxes)
    ax0.text(0.5, 0.54,
             f"Macro-Driven Framework  |  Current Phase: {current_phase.upper()}  |  "
             f"Alignment Score: {score}/100",
             ha="center", color=WHITE, fontsize=11, transform=ax0.transAxes)
    phase_color = PHASE_COLORS[current_phase]
    ax0.text(0.5, 0.26,
             f"Expected Return: {cur_metrics['return']:.1f}%  |  "
             f"Volatility: {cur_metrics['vol']:.1f}%  |  "
             f"Sharpe: {cur_metrics['sharpe']:.2f}",
             ha="center", color=phase_color, fontsize=10, transform=ax0.transAxes)
    ax0.axhline(0.08, color=GOLD, linewidth=0.8, xmin=0.1, xmax=0.9)

    # ── 1. CYCLE WHEEL (4-quadrant diagram)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(DARK_BG)
    ax1.set_xlim(-1.4, 1.4); ax1.set_ylim(-1.4, 1.4)
    ax1.set_aspect("equal"); ax1.axis("off")
    ax1.set_title("The Economic Cycle", color=GOLD, fontsize=11,
                  fontweight="bold", pad=10)

    quadrants = [
        ("Recovery",    0,   90,  0.55,  0.55),
        ("Expansion",  90,  180, -0.55,  0.55),
        ("Peak",       180, 270, -0.55, -0.55),
        ("Contraction",270, 360,  0.55, -0.55),
    ]
    from matplotlib.patches import Wedge
    for qname, a1, a2, tx, ty in quadrants:
        is_current = (qname == current_phase)
        alpha = 0.9 if is_current else 0.35
        w = Wedge((0, 0), 1.1, a1, a2,
                  facecolor=PHASE_COLORS[qname], alpha=alpha,
                  edgecolor=DARK_BG, linewidth=2)
        ax1.add_patch(w)
        ax1.text(tx, ty, qname, ha="center", va="center",
                 color=WHITE if is_current else MID,
                 fontsize=9, fontweight="bold" if is_current else "normal")
        if is_current:
            ax1.text(tx, ty - 0.22, "◄ NOW", ha="center", va="center",
                     color=PHASE_COLORS[qname], fontsize=7, fontweight="bold")

    # Centre circle
    centre = plt.Circle((0, 0), 0.28, color=DARK_BG, zorder=5)
    ax1.add_patch(centre)
    ax1.text(0, 0.06, "CYCLE", ha="center", va="center",
             color=GOLD, fontsize=8, fontweight="bold", zorder=6)
    ax1.text(0, -0.08, "CLOCK", ha="center", va="center",
             color=GOLD, fontsize=8, fontweight="bold", zorder=6)

    # Arrows around the circle
    arrow_angles = [45, 135, 225, 315]
    for angle in arrow_angles:
        rad = np.radians(angle)
        ax1.annotate("", xy=(1.28 * np.cos(rad + 0.15), 1.28 * np.sin(rad + 0.15)),
                     xytext=(1.28 * np.cos(rad - 0.15), 1.28 * np.sin(rad - 0.15)),
                     arrowprops=dict(arrowstyle="->", color=MID, lw=1.2))

    # ── 2. OPTIMAL ALLOCATION — current phase (pie)
    ax2 = fig.add_subplot(gs[1, 1])
    pie_colors = [GOLD, BLUE, GREEN, RED, ORANGE, PURPLE,
                  "#ffa657", WHITE, MID, "#c9a84c"]
    short = [a.replace("Investment Grade ", "IG ").replace("Government ", "Govt ")
             for a in ASSETS]
    wedges, texts, ats = ax2.pie(
        opt, labels=short, autopct="%1.0f%%",
        colors=pie_colors,
        textprops={"color": WHITE, "fontsize": 6.5},
        wedgeprops={"edgecolor": DARK_BG, "linewidth": 1.2},
        startangle=90
    )
    for at in ats: at.set_color(DARK_BG); at.set_fontsize(6.5)
    style_ax(ax2, f"Optimal Allocation\n{current_phase} Phase")

    # ── 3. CURRENT vs OPTIMAL — bar chart
    ax3 = fig.add_subplot(gs[1, 2])
    x  = np.arange(len(ASSETS))
    w  = 0.38
    short2 = [a.split(" ")[0][:10] for a in ASSETS]
    ax3.bar(x - w/2, current_portfolio.values, width=w,
            color=BLUE, alpha=0.8, label="Current")
    ax3.bar(x + w/2, opt.values, width=w,
            color=GOLD, alpha=0.8, label=f"Optimal ({current_phase})")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short2, rotation=45, ha="right", fontsize=7)
    ax3.yaxis.set_major_formatter(pct_f)
    ax3.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax3, "Current vs Optimal Allocation")

    # ── 4. PERFORMANCE HEATMAP (assets x phases)
    ax4 = fig.add_subplot(gs[2, :2])
    data = PERF.values
    im   = ax4.imshow(data.T, cmap="RdYlGn", aspect="auto",
                      vmin=-25, vmax=25)
    ax4.set_xticks(range(len(ASSETS)))
    ax4.set_xticklabels(short2, rotation=45, ha="right", fontsize=8)
    ax4.set_yticks(range(len(phases)))
    ax4.set_yticklabels(phases, fontsize=9)
    for i in range(len(ASSETS)):
        for j in range(len(phases)):
            val = data[i, j]
            ax4.text(i, j, f"{val:.0f}%", ha="center", va="center",
                     fontsize=7.5,
                     color="black" if abs(val) > 12 else WHITE,
                     fontweight="bold" if abs(val) > 12 else "normal")
    cb = plt.colorbar(im, ax=ax4, shrink=0.8)
    cb.set_label("Ann. Return (%)", color=WHITE, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE)
    # Highlight current phase row
    phase_idx = phases.index(current_phase)
    for i in range(len(ASSETS)):
        ax4.add_patch(plt.Rectangle(
            (i - 0.5, phase_idx - 0.5), 1, 1,
            fill=False, edgecolor=GOLD, linewidth=1.5))
    style_ax(ax4, "Historical Asset Performance by Cycle Phase (Ann. Return %)")

    # ── 5. PHASE METRICS comparison
    ax5 = fig.add_subplot(gs[2, 2])
    metric_labels = ["Exp. Return", "Volatility", "Sharpe x10"]
    cur_vals = [cur_metrics["return"], cur_metrics["vol"],
                cur_metrics["sharpe"] * 10]
    opt_vals = [opt_metrics["return"], opt_metrics["vol"],
                opt_metrics["sharpe"] * 10]
    x2 = np.arange(len(metric_labels))
    ax5.bar(x2 - 0.2, cur_vals, width=0.35, color=BLUE,  alpha=0.8, label="Current")
    ax5.bar(x2 + 0.2, opt_vals, width=0.35, color=GOLD,  alpha=0.8, label="Optimal")
    ax5.set_xticks(x2)
    ax5.set_xticklabels(metric_labels, fontsize=9)
    ax5.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax5, "Portfolio Metrics\nCurrent vs Optimal")

    # ── 6. MACRO INDICATORS TABLE
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_facecolor(DARK_BG); ax6.axis("off")
    ax6.set_title("Macro Indicator Framework by Phase", color=GOLD,
                  fontsize=11, fontweight="bold", pad=8)

    indicators = ["GDP Growth", "Inflation", "Central Bank",
                  "Credit Spreads", "Yield Curve", "Corporate Earnings"]
    col_w = 1 / (len(phases) + 1)

    # Header
    ax6.text(0.01, 0.94, "Indicator", transform=ax6.transAxes,
             color=GOLD, fontsize=9, fontweight="bold")
    for j, phase in enumerate(phases):
        ax6.text(0.01 + (j + 1) * col_w, 0.94, phase,
                 transform=ax6.transAxes,
                 color=PHASE_COLORS[phase], fontsize=9, fontweight="bold")
    ax6.plot([0.01, 0.99], [0.90, 0.90], color=GOLD, linewidth=0.5,
             transform=ax6.transAxes)

    for i, ind in enumerate(indicators):
        y = 0.82 - i * 0.13
        ax6.text(0.01, y, ind, transform=ax6.transAxes,
                 color=WHITE, fontsize=8, fontweight="bold")
        for j, phase in enumerate(phases):
            val = MACRO[phase][ind]
            is_cur = (phase == current_phase)
            ax6.text(0.01 + (j + 1) * col_w, y, val,
                     transform=ax6.transAxes,
                     color=PHASE_COLORS[phase] if is_cur else MID,
                     fontsize=7.5,
                     fontweight="bold" if is_cur else "normal")

    # ── 7. REBALANCING SIGNALS
    ax7 = fig.add_subplot(gs[4, :2])
    ax7.set_facecolor(DARK_BG)
    shifts = opt - current_portfolio
    shifts_sorted = shifts.sort_values()
    cols_bar = [GREEN if v > 0 else RED for v in shifts_sorted.values]
    short3 = [a.replace("Investment Grade ", "IG ").replace("Government ", "Govt ")
              for a in shifts_sorted.index]
    y_pos = np.arange(len(short3))
    bars = ax7.barh(y_pos, shifts_sorted.values, color=cols_bar, alpha=0.85)
    ax7.set_yticks(y_pos)
    ax7.set_yticklabels(short3, fontsize=8)
    ax7.axvline(0, color=WHITE, linewidth=0.8)
    ax7.xaxis.set_major_formatter(pct_f2)
    ax7.set_xlabel("Weight Change (%)", color=WHITE, fontsize=9)
    for bar, val in zip(bars, shifts_sorted.values):
        if abs(val) > 0.5:
            ax7.text(
                val + (0.3 if val > 0 else -0.3),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.0f}%", va="center",
                color=GREEN if val > 0 else RED, fontsize=8
            )
    style_ax(ax7, f"Rebalancing Signals: Current to {current_phase} Optimal")

    # ── 8. ALIGNMENT SCORE GAUGE
    ax8 = fig.add_subplot(gs[4, 2])
    ax8.set_facecolor(DARK_BG); ax8.axis("off")
    ax8.set_title("Alignment Score", color=GOLD, fontsize=11,
                  fontweight="bold", pad=10)

    # Draw gauge
    from matplotlib.patches import Arc
    score_color = GREEN if score >= 70 else ORANGE if score >= 40 else RED
    theta = score / 100 * 180

    bg_arc = Arc((0, 0), 1.6, 1.6, angle=0, theta1=0, theta2=180,
                 color=GREY, linewidth=14, zorder=1)
    score_arc = Arc((0, 0), 1.6, 1.6, angle=0,
                    theta1=180 - theta, theta2=180,
                    color=score_color, linewidth=14, zorder=2)
    ax8.add_patch(bg_arc)
    ax8.add_patch(score_arc)
    ax8.set_xlim(-1.2, 1.2); ax8.set_ylim(-0.5, 1.1)
    ax8.text(0, 0.1, f"{score}", ha="center", va="center",
             color=score_color, fontsize=36, fontweight="bold")
    ax8.text(0, -0.18, "/ 100", ha="center", va="center",
             color=MID, fontsize=12)
    lbl = "WELL ALIGNED" if score >= 70 else "PARTIALLY ALIGNED" if score >= 40 else "MISALIGNED"
    ax8.text(0, -0.38, lbl, ha="center", va="center",
             color=score_color, fontsize=9, fontweight="bold")
    ax8.text(-0.85, -0.05, "0", ha="center", color=MID, fontsize=8)
    ax8.text(0.85, -0.05, "100", ha="center", color=MID, fontsize=8)

    # FOOTER
    fig.text(0.5, 0.005,
             "The Meridian Playbook  |  Research on Capital Allocation & Financial Systems"
             "  |  themeridianplaybook.com",
             ha="center", color=GREY, fontsize=8)

    import os; os.makedirs("outputs", exist_ok=True)
    out = "outputs/cycle_allocation_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    ✓ Report saved → {out}\n")
    return out


# ─────────────────────────────────────────────
# PRINT CONSOLE SUMMARY
# ─────────────────────────────────────────────

def print_summary(current_phase, current_portfolio):
    opt    = ALLOCATIONS[current_phase]
    score  = alignment_score(current_portfolio, opt)
    cur_m  = compute_portfolio_metrics(current_portfolio, current_phase)
    opt_m  = compute_portfolio_metrics(opt, current_phase)
    shifts = opt - current_portfolio

    print("=" * 65)
    print(f"  CYCLE ALLOCATION FRAMEWORK")
    print(f"  Current Phase: {current_phase.upper()}")
    print("=" * 65)
    print(f"\n  Alignment Score:  {score}/100")
    print(f"\n  CURRENT PORTFOLIO ({current_phase})")
    print(f"  Expected Return:  {cur_m['return']:.1f}%")
    print(f"  Volatility:       {cur_m['vol']:.1f}%")
    print(f"  Sharpe Ratio:     {cur_m['sharpe']:.2f}")
    print(f"\n  OPTIMAL ALLOCATION ({current_phase})")
    print(f"  Expected Return:  {opt_m['return']:.1f}%")
    print(f"  Volatility:       {opt_m['vol']:.1f}%")
    print(f"  Sharpe Ratio:     {opt_m['sharpe']:.2f}")
    print(f"\n  TOP REBALANCING SIGNALS")
    for asset, delta in shifts.sort_values().items():
        if abs(delta) >= 3:
            sign = "+" if delta > 0 else ""
            print(f"  {asset:<30} {sign}{delta:.0f}%")
    print("=" * 65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   ASSET ALLOCATION BY ECONOMIC CYCLE         ║")
    print("║   The Meridian Playbook                      ║")
    print("╚══════════════════════════════════════════════╝\n")

    print(f"📐  Analysing phase: {CURRENT_PHASE}...")
    print_summary(CURRENT_PHASE, CURRENT_PORTFOLIO)

    print("\n📊  Generating report...")
    make_report(CURRENT_PHASE, CURRENT_PORTFOLIO)

    print("✅  Analysis complete.")
    print("    Open outputs/cycle_allocation_report.png\n")


if __name__ == "__main__":
    main()
