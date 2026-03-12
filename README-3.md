# Asset Allocation by Economic Cycle

**A macro-driven framework that maps optimal asset allocation across the four phases of the economic cycle: Expansion, Peak, Contraction, and Recovery.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com) — a research project on capital allocation, portfolio strategy and global financial systems.

---

## What It Does

This tool answers a question every CIO and private banker asks every quarter: *given where we are in the economic cycle, how should capital be allocated?*

It provides:

- **Optimal allocation per phase** — research-based weights across 10 asset classes
- **Historical performance heatmap** — annualised returns by asset and phase
- **Alignment score** — how aligned your current portfolio is with the optimal for the current phase
- **Rebalancing signals** — which positions to increase or reduce
- **Macro indicator framework** — the economic signals that define each phase
- **Cycle wheel** — visual representation of current positioning in the cycle

---

## The Four Phases

| Phase | GDP | Inflation | Central Bank | Key Assets |
|-------|-----|-----------|-------------|------------|
| **Expansion** | Accelerating | Rising moderately | Neutral to tightening | Equities, Private Equity |
| **Peak** | Slowing | High, persistent | Actively tightening | Commodities, Gold |
| **Contraction** | Negative | Falling | Cutting rates | Govt Bonds, Gold, Cash |
| **Recovery** | Turning positive | Low, stable | Accommodative | Equities, High Yield |

---

## Output

The tool generates a single high-resolution report (`outputs/cycle_allocation_report.png`) with eight panels:

1. **Cycle Wheel** — four-phase diagram with current phase highlighted
2. **Optimal Allocation** — pie chart for the current phase
3. **Current vs Optimal** — bar chart comparison
4. **Performance Heatmap** — historical returns across all assets and phases
5. **Portfolio Metrics** — expected return, volatility, Sharpe ratio
6. **Macro Indicator Framework** — full table of economic signals per phase
7. **Rebalancing Signals** — which assets to increase or reduce
8. **Alignment Score Gauge** — 0–100 score of current positioning

---

## Configuration

```python
# Set the current economic phase
CURRENT_PHASE = "Recovery"   # Expansion | Peak | Contraction | Recovery

# Define your current portfolio
CURRENT_PORTFOLIO = pd.Series({
    "Global Equities":         28,
    "Emerging Markets":         6,
    "Real Estate":              8,
    ...
})
```

---

## Installation

```bash
git clone https://github.com/your-username/cycle-allocation.git
cd cycle-allocation
pip install -r requirements.txt
python src/cycle_allocation.py
```

---

## The Trilogy

This project is the fourth in a series of UHNW wealth management tools:

| Project | Focus |
|---------|-------|
| [Portfolio Optimizer](https://github.com/your-username/portfolio-analyzer) | Efficient frontier, Sharpe optimization |
| [Risk Scoring Model](https://github.com/your-username/risk-scoring-model) | Multi-dimensional UHNW risk assessment |
| [Wealth Projection Tool](https://github.com/your-username/wealth-projection) | Long-term wealth preservation scenarios |
| **Cycle Allocation** | Macro-driven allocation by economic phase |

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*
*[LinkedIn](https://www.linkedin.com/in/federico-bonessi/) | [The Meridian Playbook](https://themeridianplaybook.com)*
