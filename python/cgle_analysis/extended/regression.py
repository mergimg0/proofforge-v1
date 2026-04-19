"""Analysis 10: Full regression analysis across checkpoints."""


def analysis_regression(checkpoints, all_theorems):
    print("\n" + "=" * 72)
    print("ANALYSIS 10: Full regression analysis")
    print("=" * 72)

    solved = {}
    for tid in all_theorems:
        solved[tid] = []
        for step, records_by_thm in checkpoints:
            recs = records_by_thm.get(tid, [])
            solved[tid].append(any(r["success"] for r in recs))

    print(f"\n{'Checkpoint':<12}  {'Solved':<8}  {'New':<6}  {'Lost':<6}  {'Retained':<10}  {'Regression_rate':<16}  {'Cumulative_ever':<16}")
    print("-" * 86)

    ever_solved: set[str] = set()
    prev_solved: set[str] = set()
    for idx, (step, _) in enumerate(checkpoints):
        curr_solved = {tid for tid in all_theorems if solved[tid][idx]}
        new = curr_solved - prev_solved
        lost = prev_solved - curr_solved
        retained = prev_solved & curr_solved
        ever_solved |= curr_solved

        reg_rate = len(lost) / len(prev_solved) if prev_solved else 0.0
        print(f"C{step:04d}        {len(curr_solved):<8d}  {len(new):<6d}  {len(lost):<6d}  {len(retained):<10d}  {reg_rate:.3f}            {len(ever_solved)}")

        prev_solved = curr_solved

    print(f"\nTheorems that regressed (solved then lost) at any checkpoint:")
    regression_events = []
    for tid in all_theorems:
        for idx in range(1, len(checkpoints)):
            if solved[tid][idx - 1] and not solved[tid][idx]:
                regression_events.append((tid, checkpoints[idx - 1][0], checkpoints[idx][0]))

    if regression_events:
        for tid, from_step, to_step in regression_events:
            tid_disp = tid[:55]
            final_solved = solved[tid][-1]
            status = "RECOVERED" if final_solved else "STILL LOST"
            print(f"  {tid_disp:<57}  lost C{from_step:04d}->C{to_step:04d}  [{status}]")
    else:
        print("  None!")

    print(f"\nConsolidation trajectory:")
    for idx in range(1, len(checkpoints)):
        step = checkpoints[idx][0]
        curr = {tid for tid in all_theorems if solved[tid][idx]}
        prev = {tid for tid in all_theorems if solved[tid][idx - 1]}
        if prev:
            retention = len(curr & prev) / len(prev)
            print(f"  C{checkpoints[idx-1][0]:04d}->C{step:04d}: {retention:.1%} retention ({len(curr & prev)}/{len(prev)} retained)")
