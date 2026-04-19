import os

import pandas as pd
from pandas.tests.series.methods.test_rank import results

from PSO.Lib_Swarm import Swarm
from PSO.utils import sample_control_by_time, min_max_array, sample_diversity, sample_stability, sample_number_in_bounds
from fitness_function.FitnessFunction import EVALUATION_SET


def run_tvac():
    results_data = []
    for func in EVALUATION_SET:
        fun = func()
        swarm = Swarm(
            fun
        )

        for i in range(5000):
            w, c1, c2 = sample_control_by_time(i, 5000)
            swarm.set_control(c1, c2, w)
            swarm.step(i)

            results_data.append({
                "run": -1,
                "n_t": -1,
                "step_number": i,
                "function_name": func.__name__,
                "inertia": w,
                "c1": c1,
                "c2": c2,
                "stable": sample_stability(w, c1, c2),
                "particles_in_bounds": sample_number_in_bounds(swarm.swarm.position, fun.bounds[0], fun.bounds[1]),
                "avg_velocity": swarm.swarm.velocity.mean(),
                "swarm_diversity": sample_diversity(swarm.swarm.position)[0],
                "best_fitness": swarm.swarm.best_cost,
            })

        print('The best cost found by our swarm is: {:.4f}'.format(swarm.swarm.best_cost))
        print('The best position found by our swarm is: {}'.format(swarm.swarm.best_pos))

        del swarm

    df = pd.DataFrame(results_data)
    output_dir = "../results/pso_paper_results/lib"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{output_dir}/tvac.json"

    # Using 'orient' to keep the JSON structure manageable
    df.to_json(output_filename, index=False)
    print(f"\nEvaluation complete. Total records: {len(df)}")
    print(f"Results saved to {output_filename}")


if __name__ == "__main__":
    run_tvac()
