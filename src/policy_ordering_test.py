import sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ciceroscm" / "src"))

from src.utils.config_utils import load_yaml_config

def temp_return(T, gamma):
    T = np.asarray(T, dtype=float)
    w = gamma ** np.arange(len(T))
    return -np.sum(w * (T ** 4))

def main():
    policy_ordering_config = load_yaml_config("policy_ordering_test.yaml", "policy_ordering_test")
    run_dir = policy_ordering_config["run_dir"]
    gamma = policy_ordering_config["gamma"]
    policy_consistency_results_dir = Path(run_dir) / "policy_consistency_results.npy"

    policy_data = np.load(policy_consistency_results_dir, allow_pickle=True)

    result = {
        idx: [ep["T_scm"].tolist(), ep["T_net"].tolist()]
        for idx, ep in enumerate(policy_data)
    }

    idxs = sorted(result.keys())
    R_scm = np.array([temp_return(result[i][0], gamma) for i in idxs])
    R_net = np.array([temp_return(result[i][1], gamma) for i in idxs])

    tau, _ = kendalltau(R_scm, R_net)       
    rho, _ = spearmanr(R_scm, R_net)        

    print(f"Kendall's tau: {tau:.3f}")
    print(f"Spearman's rho: {rho:.6f}")

    summary_path = Path(run_dir) / "ordering_test_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"Kendall's tau: {tau:.3f}",
                f"Spearman's rho: {rho:.6f}",
            ]
        )
    )

if __name__ == "__main__":
    main()