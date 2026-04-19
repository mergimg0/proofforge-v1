"""Analysis 7: Trajectory length ratio (low-T / high-T)."""

import numpy as np


def analysis_length_ratio(checkpoints, all_theorems):
    print("\n" + "=" * 72)
    print("ANALYSIS 7: Trajectory length ratio (low-T / high-T)")
    print("=" * 72)

    print(f"\n{'Checkpoint':<12}  {'N_multi_T':<12}  {'Mean_ratio_0.3/1.2':<20}  {'Mean_len_T=0.3':<16}  {'Mean_len_T=1.2':<16}")
    print("-" * 80)

    for step, records_by_thm in checkpoints:
        ratios = []
        lens_low = []
        lens_high = []
        for tid in all_theorems:
            recs = records_by_thm.get(tid, [])
            temp_lengths: dict[float, list[int]] = {}
            for rec in recs:
                if rec["success"]:
                    t = round(rec["temperature"], 1)
                    temp_lengths.setdefault(t, []).append(rec["n_tokens"])
            if 0.3 in temp_lengths and 1.2 in temp_lengths:
                mean_low = np.mean(temp_lengths[0.3])
                mean_high = np.mean(temp_lengths[1.2])
                if mean_high > 0:
                    ratios.append(mean_low / mean_high)
                    lens_low.append(mean_low)
                    lens_high.append(mean_high)

        if ratios:
            mean_ratio = np.mean(ratios)
            mean_l = np.mean(lens_low)
            mean_h = np.mean(lens_high)
            print(f"C{step:04d}        {len(ratios):<12d}  {mean_ratio:<20.3f}  {mean_l:<16.1f}  {mean_h:<16.1f}")
        else:
            print(f"C{step:04d}        {'0':<12}  {'N/A':<20}  {'N/A':<16}  {'N/A':<16}")

    print(f"\nPer-theorem length ratio at C0150 (theorems with both T=0.3 and T=1.2 success):")
    step_150 = None
    for step, records_by_thm in checkpoints:
        if step == 150:
            step_150 = records_by_thm
    if step_150:
        for tid in sorted(all_theorems):
            recs = step_150.get(tid, [])
            temp_lengths: dict[float, list[int]] = {}
            for rec in recs:
                if rec["success"]:
                    t = round(rec["temperature"], 1)
                    temp_lengths.setdefault(t, []).append(rec["n_tokens"])
            if 0.3 in temp_lengths and 1.2 in temp_lengths:
                mean_low = np.mean(temp_lengths[0.3])
                mean_high = np.mean(temp_lengths[1.2])
                ratio = mean_low / mean_high if mean_high > 0 else float("inf")
                tid_disp = tid[:55]
                direction = "low-T shorter" if ratio < 1 else "low-T longer"
                print(f"  {tid_disp:<57} {mean_low:.0f}/{mean_high:.0f} = {ratio:.2f} ({direction})")
