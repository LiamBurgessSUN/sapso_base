import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import SAC

from fitness_function.FitnessFunction import EVALUATION_SET
from RL.rl_env_wrapper import SAPSOEnv
from PSO.Swarm import Swarm

# Research Configuration (Ref: Section 4.2)
NUM_TRIALS = 30
MODEL_PATH = "../policies/sac_sapso_policy_nt_125.zip"
NUM_PARTICLES = 30
DIM = 30
MAX_STEPS = 5000


def run_rl_trial(model, func_class, trial_seed):
    """Executes one independent trial using the SAC policy."""
    env = SAPSOEnv(
        num_particles=NUM_PARTICLES,
        dim=DIM,
        max_steps=MAX_STEPS,
        seed=trial_seed,
        fitness_function_class=func_class
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.swarm.best_fitness


def run_constant_baseline_trial(func_class, trial_seed):
    """Fixed Parameters (Clerc & Kennedy)."""
    ff = func_class()
    swarm = Swarm(
        number_particles=NUM_PARTICLES,
        fitness_function=ff,
        expected_iterations=MAX_STEPS,
        seed=trial_seed
    )
    swarm.set_control_parameters(1.496180, 1.496180, 0.729844)

    for i in range(MAX_STEPS):
        swarm.step(i)
        if swarm.is_stagnated(): break
    return swarm.best_fitness


def run_tvac_baseline_trial(func_class, trial_seed):
    """Time-Variant Acceleration Coefficients (Eq. 3)."""
    ff = func_class()
    swarm = Swarm(
        number_particles=NUM_PARTICLES,
        fitness_function=ff,
        expected_iterations=MAX_STEPS,
        seed=trial_seed
    )

    for i in range(MAX_STEPS):
        # Apply Eq. 3 Logic
        w = 0.4 * ((i - MAX_STEPS) / MAX_STEPS) ** 2 + 0.4
        c1 = -3 * (i / MAX_STEPS) + 3.5
        c2 = 3 * (i / MAX_STEPS) + 0.5
        swarm.set_control_parameters(c1, c2, w)

        swarm.step(i)
        if swarm.is_stagnated(): break
    return swarm.best_fitness


def perform_statistical_analysis():
    print(f"📂 Loading SAC Policy: {MODEL_PATH}")
    try:
        model = SAC.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    all_stats = []

    for func_class in EVALUATION_SET:
        func_name = func_class().__class__.__name__
        print(f"\n📊 Significance Testing: {func_name}")

        rl_data, const_data, tvac_data = [], [], []

        for i in range(NUM_TRIALS):
            # Seed propagation unique for each trial but synced across methods
            seed = 1000 + i
            rl_data.append(run_rl_trial(model, func_class, seed))
            const_data.append(run_constant_baseline_trial(func_class, seed))
            tvac_data.append(run_tvac_baseline_trial(func_class, seed))
            print(f"  Trial {i + 1}/{NUM_TRIALS}...", end="\r")

        # Welch's T-Test against both baselines
        t_const, p_const = stats.ttest_ind(rl_data, const_data, equal_var=False)
        t_tvac, p_tvac = stats.ttest_ind(rl_data, tvac_data, equal_var=False)

        # Means and Std Devs
        m_rl, s_rl = np.mean(rl_data), np.std(rl_data)
        m_const = np.mean(const_data)
        m_tvac = np.mean(tvac_data)

        # Improvement calculation (vs Constant)
        imp_const = ((m_const - m_rl) / abs(m_const)) * 100 if m_const != 0 else 0

        all_stats.append({
            "Function": func_name,
            "RL Mean (σ)": f"{m_rl:.2e} ({s_rl:.1e})",
            "Imp % (vs Const)": f"{imp_const:.1f}%",
            "P-Val (Const)": f"{p_const:.2e}",
            "P-Val (TVAC)": f"{p_tvac:.2e}",
            "Sig (Both)": "✅ YES" if (p_const < 0.05 and p_tvac < 0.05) else "⚠️ VARIES"
        })

    df = pd.DataFrame(all_stats)
    print("\n\n" + "=" * 95)
    print("📈 MULTI-BASELINE SIGNIFICANCE RESULTS (30 Trials/Function)")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95)
    print("Interpretation: P-Value < 0.05 confirms RL strategy is significantly different from baseline.")


if __name__ == "__main__":
    perform_statistical_analysis()