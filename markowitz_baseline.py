import numpy as np
import cvxpy as cp

from data_loader import TICKERS, SECTOR_NAMES


def solve_min_variance(sigma, long_only=True):
    """Global minimum-variance portfolio (no return target)."""
    n = sigma.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value, prob.value


def solve_target_return(mu, sigma, target_return, long_only=True):
    """Minimize variance subject to hitting a target expected return."""
    n = sigma.shape[0]
    w = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    constraints = [cp.sum(w) == 1, mu @ w >= target_return]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if w.value is None:
        return None, None
    return w.value, prob.value


def compute_efficient_frontier(mu, sigma, n_points=50, long_only=True):
    min_var_w, _ = solve_min_variance(sigma, long_only)
    min_var_return = mu @ min_var_w

    max_return = mu.max() if long_only else mu.max()  # long-only cap = best single asset
    targets = np.linspace(min_var_return, max_return * 0.999, n_points)

    frontier_returns, frontier_risks, frontier_weights = [], [], []
    for t in targets:
        w, var = solve_target_return(mu, sigma, t, long_only)
        if w is None:
            continue
        frontier_returns.append(mu @ w)
        frontier_risks.append(np.sqrt(var))
        frontier_weights.append(w)

    return (
        np.array(frontier_returns),
        np.array(frontier_risks),
        frontier_weights,
    )


def solve_max_sharpe(mu, sigma, risk_free_rate=0.0, long_only=True):
    n = sigma.shape[0]
    excess = mu - risk_free_rate
    y = cp.Variable(n)
    objective = cp.Minimize(cp.quad_form(y, cp.psd_wrap(sigma)))
    constraints = [excess @ y == 1]
    if long_only:
        constraints.append(y >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if y.value is None:
        return None, None, None

    w = y.value / np.sum(y.value)
    port_return = mu @ w
    port_risk = np.sqrt(w @ sigma @ w)
    sharpe = (port_return - risk_free_rate) / port_risk
    return w, port_return, port_risk, sharpe


def portfolio_stats(w, mu, sigma, risk_free_rate=0.0):
    ret = mu @ w
    risk = np.sqrt(max(w @ sigma @ w, 0))
    sharpe = (ret - risk_free_rate) / risk if risk > 1e-12 else np.nan
    return ret, risk, sharpe


if __name__ == "__main__":
    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")

    print("=" * 70)
    print("CLASSICAL MARKOWITZ BASELINE (continuous weights, long-only)")
    print("=" * 70)

    # Minimum variance portfolio
    w_minvar, var_minvar = solve_min_variance(sigma)
    ret_minvar, risk_minvar, sharpe_minvar = portfolio_stats(w_minvar, mu, sigma)
    print(f"\n--- Minimum Variance Portfolio ---")
    print(f"Expected return: {ret_minvar:.4f}  |  Risk (vol): {risk_minvar:.4f}  |  Sharpe: {sharpe_minvar:.4f}")
    for t, wi in zip(TICKERS, w_minvar):
        if wi > 1e-4:
            print(f"  {t:5s}: {wi:.4f}")

    # Max Sharpe portfolio
    w_sharpe, ret_sharpe, risk_sharpe, sharpe_val = solve_max_sharpe(mu, sigma)
    print(f"\n--- Maximum Sharpe Ratio Portfolio ---")
    print(f"Expected return: {ret_sharpe:.4f}  |  Risk (vol): {risk_sharpe:.4f}  |  Sharpe: {sharpe_val:.4f}")
    for t, wi in zip(TICKERS, w_sharpe):
        if wi > 1e-4:
            print(f"  {t:5s}: {wi:.4f}")

    # Efficient frontier
    frontier_ret, frontier_risk, frontier_weights = compute_efficient_frontier(mu, sigma, n_points=50)
    print(f"\n--- Efficient Frontier ---")
    print(f"Computed {len(frontier_ret)} points.")
    print(f"Return range: [{frontier_ret.min():.4f}, {frontier_ret.max():.4f}]")
    print(f"Risk range:   [{frontier_risk.min():.4f}, {frontier_risk.max():.4f}]")

    np.save("results/frontier_returns.npy", frontier_ret)
    np.save("results/frontier_risks.npy", frontier_risk)
    np.save("results/w_minvar.npy", w_minvar)
    np.save("results/w_maxsharpe.npy", w_sharpe)
    np.save("results/frontier_weights.npy", np.array(frontier_weights))

    print("\nSaved frontier + key portfolios to results/")
