from rl_eval import evaluate_sac_sapso
import pandas as pd

if __name__ == "__main__":
    SEED = 17
    AUTO = False
    N_T = 10
    results = pd.DataFrame()

    for i in range(0, 30):
        print(f"Iteration {i}")
        df = pd.DataFrame(evaluate_sac_sapso(seed=SEED + i, auto=AUTO, nt=N_T, plot=False))
        df["run"] = i

        results = pd.concat([results, df])

    if AUTO:
        ntype = "auto"
    else:
        ntype = str(N_T)
    save_path = f"sac_sapso_policy_nt_{ntype}"
    results.reset_index(inplace=True)
    results.to_json(f"./trial_results/{save_path}.json")
