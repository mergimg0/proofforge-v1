"""Analysis 4: Temperature breadth (learning depth) across checkpoints."""

def analysis_temperature_breadth(checkpoints, all_theorems):
    print("\n" + "=" * 72)
    print("ANALYSIS 4: Temperature breadth (learning depth)")
    print("=" * 72)

    steps = [c[0] for c in checkpoints]
    breadth = {}
    for step, records_by_thm in checkpoints:
        for tid in all_theorems:
            solving_temps = set()
            for rec in records_by_thm.get(tid, []):
                if rec["success"]:
                    t = round(rec["temperature"], 1)
                    solving_temps.add(t)
            breadth.setdefault(tid, []).append((step, len(solving_temps)))

    print(f"\n{'Checkpoint':<12}  {'Fully(3T)':<12}  {'Mostly(2T)':<12}  {'Fragile(1T)':<12}  {'Unsolved(0T)':<12}  {'Depth_ratio':<12}")
    print("-" * 80)

    depth_ratios = []
    for idx, step in enumerate(steps):
        fully = mostly = fragile = unsolved = 0
        for tid in all_theorems:
            n_temps = breadth[tid][idx][1]
            if n_temps == 3:
                fully += 1
            elif n_temps == 2:
                mostly += 1
            elif n_temps == 1:
                fragile += 1
            else:
                unsolved += 1
        total_solved = fully + mostly + fragile
        ratio = fully / total_solved if total_solved > 0 else 0.0
        depth_ratios.append(ratio)
        print(f"C{step:04d}        {fully:<12d}  {mostly:<12d}  {fragile:<12d}  {unsolved:<12d}  {ratio:.3f}")

    print(f"\nTheorems with monotonically increasing depth:")
    mono_count = 0
    for tid in all_theorems:
        vals = [b[1] for b in breadth[tid]]
        nonzero = [(i, v) for i, v in enumerate(vals) if v > 0]
        if len(nonzero) >= 2:
            is_mono = all(nonzero[i+1][1] >= nonzero[i][1] for i in range(len(nonzero)-1))
            if is_mono and nonzero[-1][1] > nonzero[0][1]:
                first_step = steps[nonzero[0][0]]
                last_step = steps[nonzero[-1][0]]
                first_depth = nonzero[0][1]
                last_depth = nonzero[-1][1]
                tid_disp = tid[:55]
                print(f"  {tid_disp:<57} {first_depth}T@C{first_step:04d} -> {last_depth}T@C{last_step:04d}")
                mono_count += 1
    print(f"\n  {mono_count}/{len(all_theorems)} theorems show monotonic depth increase")

    return breadth, depth_ratios
