"""
GARCH(1,1) Volatility Modelling on NIFTY 50
---------------------------------------------
Built this to actually see volatility clustering in real data instead
of just reading about it - "today's volatility depends on yesterday's"
sounds simple until you try to prove it and then forecast with it.

Steps:
1. pull 5y of NIFTY 50 data, compute log returns
2. run the ARCH LM test to confirm volatility clustering is actually there
3. fit a GARCH(1,1) model with the arch library
4. plot the conditional volatility to see the clustering visually
5. use the fitted model to forecast next-day volatility and turn that
   into a 1-day VaR estimate
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model
from scipy import stats

TICKER = "^NSEI"          # NIFTY 50 index on yfinance
YEARS = 5
VAR_CONFIDENCE = 0.95      # confidence level for the VaR forecast


def get_returns(ticker, years):
    print(f"downloading {years}y of data for {ticker}...")
    prices = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)["Close"]
    prices = prices.dropna()
    # log returns, scaled by 100 - arch_model likes returns in % form,
    # optimizer converges better and the parameter values are easier to read
    log_returns = np.log(prices / prices.shift(1)).dropna() * 100
    log_returns = log_returns.squeeze()
    print(f"got {len(log_returns)} days of returns, {prices.index[0].date()} -> {prices.index[-1].date()}")
    return log_returns


def check_volatility_clustering(returns):
    """
    Before fitting a GARCH model you should actually check that
    volatility clustering exists - otherwise you're fitting a model
    for an effect that isn't there. The ARCH LM test does this: it
    regresses squared returns on their own lags and tests whether
    past squared returns predict current squared returns. A small
    p-value (< 0.05) means yes, ARCH effects are present.
    """
    print("\n--- ARCH LM test ---")
    lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(returns, nlags=5)
    print(f"LM statistic: {lm_stat:.4f}")
    print(f"LM p-value:   {lm_pvalue:.6f}")
    if lm_pvalue < 0.05:
        print("p-value < 0.05 -> we reject the null of 'no ARCH effects'.")
        print("Volatility clustering IS present, GARCH is a reasonable model to fit.")
    else:
        print("p-value >= 0.05 -> can't reject 'no ARCH effects'.")
        print("Not much evidence of clustering here - a GARCH model may not add much.")
    return lm_pvalue


def show_naive_vs_reality(returns):
    """
    Quick illustration of why a constant-volatility assumption is wrong:
    compare the 30-day rolling std dev to the single, full-sample std dev.
    If volatility were really constant, the rolling number would just be
    flat noise around the full-sample number. In practice it swings around
    a lot - that swinging IS the clustering.
    """
    rolling_vol = returns.rolling(30).std()
    full_sample_vol = returns.std()
    print("\n--- constant vol vs reality ---")
    print(f"full-sample (constant) daily vol assumption: {full_sample_vol:.3f}%")
    print(f"30-day rolling vol ranges from {rolling_vol.min():.3f}% to {rolling_vol.max():.3f}%")
    print("if volatility were actually constant, that range would be tiny.")
    print("it isn't - that's volatility clustering: calm periods and turbulent")
    print("periods cluster together instead of being random noise day to day.")
    return rolling_vol


def fit_garch(returns):
    print("\n--- fitting GARCH(1,1) ---")
    model = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")
    print(fitted.summary())
    return fitted


def explain_parameters(fitted):
    params = fitted.params
    omega = params["omega"]
    alpha = params["alpha[1]"]
    beta = params["beta[1]"]
    persistence = alpha + beta

    print("\n--- what the parameters mean ---")
    print(f"omega = {omega:.5f}  -> baseline/long-run variance level (constant term)")
    print(f"alpha = {alpha:.5f}  -> how much yesterday's SHOCK (squared return surprise)")
    print("          feeds into today's volatility - bigger alpha = more reactive")
    print("          to sudden news/shocks")
    print(f"beta  = {beta:.5f}  -> how much yesterday's VOLATILITY carries into today")
    print("          bigger beta = volatility is stickier/more persistent by itself")
    print(f"\npersistence (alpha + beta) = {persistence:.5f}")
    if persistence < 1:
        half_life = np.log(0.5) / np.log(persistence) if persistence > 0 else np.nan
        print(f"since alpha + beta < 1, volatility shocks DIE OUT over time")
        print(f"(mean-reverting). half-life of a shock ~ {half_life:.1f} days -")
        print("meaning it takes about that many days for a volatility spike to")
        print("decay halfway back to its long-run average.")
    else:
        print("alpha + beta >= 1 - volatility shocks are NOT mean-reverting,")
        print("they'd persist indefinitely (this is unusual and worth double")
        print("checking the data/model if it happens).")
    return persistence


def plot_conditional_volatility(returns, fitted, rolling_vol):
    cond_vol = fitted.conditional_volatility

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(returns.index, returns, color="steelblue", linewidth=0.6)
    axes[0].set_title("Daily Log Returns (%)")
    axes[0].axhline(0, color="black", linewidth=0.5)

    axes[1].plot(cond_vol.index, cond_vol, color="darkred", label="GARCH(1,1) conditional volatility")
    axes[1].plot(rolling_vol.index, rolling_vol, color="gray", alpha=0.5, label="30-day rolling std dev")
    axes[1].set_title("Conditional Volatility - Clustering Visible Here")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("garch_volatility.png", dpi=150)
    print("\nsaved chart to garch_volatility.png")


def garch_var_forecast(fitted, confidence):
    """
    Use the GARCH model's 1-step-ahead volatility forecast to build a
    dynamic VaR estimate. Unlike a flat historical VaR, this VaR updates
    with current market conditions - if we're in a turbulent period,
    the forecasted vol (and therefore VaR) goes up automatically.
    """
    forecast = fitted.forecast(horizon=1, reindex=False)
    forecast_variance = forecast.variance.values[-1, 0]
    forecast_vol = np.sqrt(forecast_variance)  # still in % daily terms

    z = stats.norm.ppf(1 - confidence)
    # mean return assumed ~0 for a 1-day horizon, keeps this focused on volatility
    var_forecast_pct = z * forecast_vol

    print(f"\n--- 1-day GARCH VaR forecast ---")
    print(f"forecasted next-day volatility: {forecast_vol:.3f}%")
    print(f"{confidence:.0%} 1-day VaR forecast: {var_forecast_pct:.3f}%")
    print("(this is dynamic - rerun on a different day and it moves with")
    print("current volatility conditions, unlike a static historical VaR)")
    return forecast_vol, var_forecast_pct


def main():
    returns = get_returns(TICKER, YEARS)
    check_volatility_clustering(returns)
    rolling_vol = show_naive_vs_reality(returns)
    fitted = fit_garch(returns)
    explain_parameters(fitted)
    plot_conditional_volatility(returns, fitted, rolling_vol)
    garch_var_forecast(fitted, VAR_CONFIDENCE)


if __name__ == "__main__":
    main()
