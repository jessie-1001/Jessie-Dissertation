def print_risk_comparison(copula_metrics, historical_metrics, alpha=0.05):
    level = int((1 - alpha) * 100)
    print(f"=== Portfolio Risk Measures Comparison (α={alpha:.2f}, {level}% CI) ===")
    print(f"{'Method':<20} {'VaR':>12} {'ES':>12}")
    print(f"{'Copula Simulated':<20} {copula_metrics['VaR']:>12.5f} {copula_metrics['ES']:>12.5f}")
    print(f"{'Historical':<20} {historical_metrics['VaR']:>12.5f} {historical_metrics['ES']:>12.5f}")
