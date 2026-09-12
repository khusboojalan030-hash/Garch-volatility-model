# GARCH(1,1) Volatility Modelling on NIFTY 50

A project that actually proves and models volatility clustering in
Indian equity index data, instead of just assuming it - built while
going deeper into the market risk / volatility modelling side of FRM.

## What it teaches (and what the script actually shows)

- **Why constant volatility is wrong** - compares a flat full-sample
  std dev against a 30-day rolling std dev on real data. If volatility
  were constant, the rolling number would barely move. It doesn't stay
  still - that's the whole point.
- **The ARCH LM test** - before fitting any GARCH model, you should
  check that ARCH effects (volatility clustering) actually exist in
  the data. The script runs this test and reports the p-value.
- **GARCH(1,1) parameters** - omega, alpha, and beta, and what each one
  actually controls in plain terms (not just the formula).
- **Persistence (alpha + beta)** - what it means for how long a
  volatility shock sticks around before decaying back to normal.
- **Dynamic VaR** - using the GARCH model's next-day volatility
  forecast to produce a VaR estimate that updates with current market
  conditions, instead of a flat historical number.

## How it's built

1. Download 5 years of NIFTY 50 data (`^NSEI` on yfinance) and compute
   log returns
2. Run the ARCH LM test to confirm volatility clustering is present
   before bothering to fit a model
3. Fit GARCH(1,1) using the `arch` library, inspect omega/alpha/beta
4. Extract and plot conditional volatility next to raw returns, to see
   the clustering visually
5. Use the fitted model's 1-step-ahead forecast to produce a dynamic
   1-day VaR estimate

## Understanding the parameters

| Parameter | What it controls |
|---|---|
| **omega (ω)** | Baseline/long-run variance level - the "floor" volatility reverts to |
| **alpha (α)** | How much yesterday's shock (squared return surprise) feeds into today's volatility - reactivity to news |
| **beta (β)** | How much yesterday's volatility itself carries into today - "stickiness" of volatility |
| **alpha + beta** | Persistence - how long a volatility spike takes to decay back to normal. Close to 1 = shocks last a long time; well below 1 = they fade fast |

## Why GARCH VaR beats a flat historical VaR

A plain historical VaR uses one number for the whole sample period -
it doesn't know if you're currently in a calm market or a turbulent
one. GARCH's conditional volatility does know this: it updates every
day based on recent shocks and recent volatility. So the VaR forecast
this script produces actually reacts to current conditions - it goes
up automatically during volatile stretches and comes back down during
calm ones.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python garch_model.py
```

To model a different index/stock, change the ticker at the top of the
script:
```python
TICKER = "^NSEI"   # NIFTY 50 - swap for any yfinance ticker
YEARS = 5
VAR_CONFIDENCE = 0.95
```

## Project structure

```
garch-volatility-model/
├── garch_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Possible extensions

- Try EGARCH or GJR-GARCH to capture the leverage effect (volatility
  reacting more to negative shocks than positive ones)
- Backtest the GARCH VaR forecast against realized returns
- Compare GARCH VaR against the historical/parametric VaR from my
  other project
- Extend to a multi-day volatility forecast
