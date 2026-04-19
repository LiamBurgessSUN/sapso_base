import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from rl_env_wrapper import SAPSOEnv
from fitness_function.FitnessFunction import EVALUATION_SET
from results.plots.plotly_plots import plot_swarm


def run_single_evaluation(model, func_class, num_particles=30, dim=30, patience=200, auto=False, seed=17,
                          nt: int = 125):
    """
    Simulates a single function run and returns the internal swarm metrics.
    """
    env = SAPSOEnv(
        num_particles=num_particles,
        dim=dim,
        max_steps=5000,
        n_t=nt,
        stagnation_patience=patience,
        fitness_function_class=func_class,
        auto=auto,
        seed=seed
    )

    assert env.n_t == nt
    assert env.dim == dim

    obs, _ = env.reset()
    done = False

    while not done:
        # Deterministic policy for evaluation
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.swarm, env.current_step, env.results_data


def evaluate_sac_sapso(model_path="policies/sac_sapso_policy_nt_<type>.zip", num_particles=30, dim=30, nt=125,
                       auto=False, seed=17, plot=True):
    """
    Benchmarks the trained SAC policy across the entire Evaluation Set (7 functions).
    Generates a summary table and convergence plots for each.
    """
    if auto:
        ntype = "auto"
    else:
        ntype = str(nt)

    model_path = model_path.replace("<type>", ntype)
    print(f"📂 Loading policy from {model_path}...")
    try:
        model = SAC.load(model_path)
    except FileNotFoundError:
        print(f"❌ Model not found at {model_path}. Train the model first using RL.py.")
        return

    results = []

    print(f"\n🚀 Benchmarking SAC-SAPSO against {len(EVALUATION_SET)} Test Functions...")
    print("-" * 60)

    fitness_log = []

    data = []

    for func_class in EVALUATION_SET:
        func_name = func_class().__class__.__name__
        print(f"Testing: {func_name}...", end=" ", flush=True)

        # Propagating seed directly to evaluation run
        swarm, steps, new_data = run_single_evaluation(model, func_class, num_particles, dim, 200, auto, seed, nt)

        results.append({
            "Function": func_name,
            "Best Fitness": f"{swarm.best_fitness:.4e}",
            "Iterations": steps,
            "Stagnated": "Yes" if swarm.is_stagnated() else "No"
        })
        fitness_log.append(swarm.best_fitness)
        print(f"Done. (Best: {swarm.best_fitness:.2e})")

        # Visualization for each test function
        if plot:
            plot_swarm(swarm, func_name)

        data += new_data

    # 5. Display Summary Table (Ref: Table 6 in Paper)
    df = pd.DataFrame(results)
    print("\n📊 EVALUATION SUMMARY TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    print(f"Mean Fitness: {np.mean(fitness_log):.4e}")
    print(f"Std Fitness: {np.std(fitness_log):.4e}")

    save_path = f"sac_sapso_policy_nt_{ntype}"

    # pd.DataFrame(data).to_json(f"./test_results/{save_path}.json")
    return data


if __name__ == "__main__":
    n_t = 125
    SEED = 17
    AUTO = True
    evaluate_sac_sapso(nt=n_t, auto=AUTO, seed=SEED)
