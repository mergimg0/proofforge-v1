"""Analysis 12: Sigmoid scaling law fit to pass rate curve."""

import numpy as np
from scipy.optimize import curve_fit


def sigmoid(x, L, k, x0, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b


def analysis_sigmoid_fit(checkpoints):
    print("\n" + "=" * 72)
    print("ANALYSIS 12: Sigmoid scaling law fit")
    print("=" * 72)

    steps_arr = np.array([c[0] for c in checkpoints], dtype=float)
    pass_rates = []
    for step, records_by_thm in checkpoints:
        total = sum(len(recs) for recs in records_by_thm.values())
        passing = sum(sum(1 for r in recs if r["success"]) for recs in records_by_thm.values())
        pass_rates.append(passing / total if total > 0 else 0.0)
    rates_arr = np.array(pass_rates) * 100

    print(f"\nInput data:")
    for s, r in zip(steps_arr, rates_arr):
        print(f"  Step {s:6.0f}: {r:6.2f}%")

    try:
        p0 = [80.0, 0.05, 50.0, 2.0]
        bounds = ([0, 0.001, 0, -10], [100, 1.0, 300, 30])
        popt, pcov = curve_fit(sigmoid, steps_arr, rates_arr, p0=p0, bounds=bounds, maxfev=10000)
        L, k, x0, b = popt
        perr = np.sqrt(np.diag(pcov))

        residuals = rates_arr - sigmoid(steps_arr, *popt)
        rmse = np.sqrt(np.mean(residuals**2))

        print(f"\nSigmoid fit: y = {L:.2f} / (1 + exp(-{k:.4f} * (x - {x0:.2f}))) + {b:.2f}")
        print(f"  Asymptote (L + b): {L + b:.2f}%")
        print(f"  Growth rate (k): {k:.4f}")
        print(f"  Inflection point (x0): step {x0:.1f}")
        print(f"  RMSE: {rmse:.2f}pp")
        print(f"  Parameter uncertainties: L+-{perr[0]:.2f}, k+-{perr[1]:.4f}, x0+-{perr[2]:.2f}, b+-{perr[3]:.2f}")

        print(f"\nExtrapolations:")
        for target_step in [200, 250, 300, 500]:
            predicted = sigmoid(target_step, *popt)
            print(f"  Step {target_step}: {predicted:.2f}%")

        for target_pct in [80, 85, 90, 95]:
            target_val = target_pct - b
            if target_val <= 0 or target_val >= L:
                print(f"  {target_pct}% unreachable (asymptote = {L + b:.2f}%)")
            else:
                x_target = x0 - np.log(L / target_val - 1) / k
                print(f"  {target_pct}%: ~step {x_target:.0f}")

    except Exception as e:
        print(f"\nSigmoid fit failed: {e}")
        if len(steps_arr) >= 3:
            coeffs = np.polyfit(steps_arr[-3:], rates_arr[-3:], 1)
            for target_step in [200, 250, 300]:
                predicted = np.polyval(coeffs, target_step)
                print(f"  Step {target_step}: {predicted:.2f}% (linear)")
