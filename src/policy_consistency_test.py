import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ciceroscm" / "src"))

from src.utils.config_utils import load_yaml_config
from src.utils.cicero_utils import load_cicero_inputs
from src.marl_env_step import CICEROSCMEngine


def compute_scm_for_random_episodes(run_dir, n_samples=3, seed=None):
    """
    Sample up to `n_samples` .npz logs from `run_dir/marl_logs` and
    return their surrogate vs. CICERO-SCM temperature trajectories.
    """
    run_dir = Path(run_dir)
    logs_dir = run_dir / "marl_logs"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"{logs_dir} not found")

    episode_files = sorted(logs_dir.glob("*.npz"))
    if not episode_files:
        raise FileNotFoundError(f"No .npz logs in {logs_dir}")

    rng = np.random.default_rng(seed)
    picks = rng.choice(
        episode_files,
        size=min(n_samples, len(episode_files)),
        replace=False,
    )

    # Load CICERO-SCM data once so we can spin up a fresh engine per episode
    data_root = run_dir.parents[1]  # …/data/<timestamp>
    cfg_path = data_root / "configs" / "cicero_scm.yaml"
    cicero_cfg = load_yaml_config(cfg_path)
    gaspam_data, conc_data, em_data, nat_ch4_data, nat_n2o_data = load_cicero_inputs(cicero_cfg)
    historical = em_data.loc[:cicero_cfg["em_data_policy"]]
    engine_kwargs = dict(
        historical_emissions=historical,
        gaspam_data=gaspam_data,
        conc_data=conc_data,
        nat_ch4_data=nat_ch4_data,
        nat_n2o_data=nat_n2o_data,
        pamset_udm=cicero_cfg["cfg"][0]["pamset_udm"],
        pamset_emiconc=cicero_cfg["cfg"][0]["pamset_emiconc"],
        em_data_start=cicero_cfg["em_data_start"],
        em_data_policy=cicero_cfg["em_data_policy"],
        udir=cicero_cfg["test_data_dir"],
        idtm=24,
    )

    results = []
    for npz_path in tqdm(
        picks,
        desc="CICERO-SCM episodes",
        unit="episode",
    ):
        with np.load(npz_path, allow_pickle=True) as episode:
            years = episode["years"]
            E_global = episode["E_global"]
            T_net = episode["T"]
            gas_names = list(episode["gas_names"])

        engine = CICEROSCMEngine(**engine_kwargs, scenname=npz_path.stem)
        T_scm = np.array([engine.step(e) for e in E_global])

        results.append(
            {
                "episode_path": npz_path,
                "years": years,
                "gas_names": gas_names,
                "T_net": T_net,
                "T_scm": T_scm,
            }
        )
    return results


def main():
    marl_config = load_yaml_config("policy_consistency_test.yaml", "policy_consistency_test")
    run_dir = marl_config["run_dir"]
    n_samples = marl_config["n_samples"]
    random_seed = marl_config["random_seed"]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    policy_consistency_root = Path(run_dir) / "policy_consistency_check"
    policy_consistency_root.mkdir(parents=True, exist_ok=True)

    sample_run_dir = policy_consistency_root / f"{timestamp}_{n_samples}"
    sample_run_dir.mkdir(parents=True, exist_ok=True)

    results = compute_scm_for_random_episodes(run_dir, n_samples=n_samples, seed=random_seed)

    output_path = sample_run_dir / "policy_consistency_results.npy"
    np.save(output_path, results)
    print(f"Saved results for {len(results)} episodes to {output_path}")

    # Compute global RMSE and R^2 across all samples
    T_net_all = np.concatenate([sample["T_net"] for sample in results])
    T_scm_all = np.concatenate([sample["T_scm"] for sample in results])

    rmse = np.sqrt(mean_squared_error(T_net_all, T_scm_all))
    r2 = r2_score(T_net_all, T_scm_all)
    print(f"Global RMSE across {len(results)} episodes: {rmse:.6f} K")
    print(f"Global R^2 across {len(results)} episodes: {r2:.6f}")

    summary_path = sample_run_dir / "summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"timestamp: {timestamp}",
                f"run_dir: {run_dir}",
                f"requested_n_samples: {n_samples}",
                f"actual_episodes: {len(results)}",
                f"random_seed: {random_seed}",
                f"rmse_K: {rmse:.6f}",
                f"r2: {r2:.6f}",
            ]
        )
    )
    print(f"Wrote summary metrics to {summary_path}")

if __name__ == "__main__":
    main()
