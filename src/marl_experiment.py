import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ray.*")
warnings.filterwarnings("ignore", message=".*Logger.*deprecated.*", category=DeprecationWarning)

sys.path.insert(0, os.path.join(os.getcwd(), '../ciceroscm/', 'src'))  # Make ciceroscm importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from marl_env import ClimateMARL  # noqa: E402
from src.utils.marl_utils import load_marl_setup, _to_jsonable  # noqa: E402
from src.utils.config_utils import load_yaml_config


class ClimateMarlExperiment:
    def __init__(self, env_config, emission_data, economics_config, actions_config, training_cfg):
        self.env_config = env_config
        self.emission_data = emission_data
        self.economics_config = economics_config
        self.actions_config = actions_config
        self.training_cfg = training_cfg
        self.lever_names = list(actions_config["lever_names"])
        self.action_dim = len(actions_config["action_sizes"])
        self.adaptation_idx = self.action_dim - 1

    def rollout_fixed_actions(self, seed=None):
        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )

        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

        done = False
        t = 0
        trajectory = []
        per_agent_return = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}
        total_return = 0.0
        info_dict = {}

        base_action = np.zeros(self.action_dim, dtype=np.int64)
        if "energy" in self.lever_names:
            energy_idx = self.lever_names.index("energy")
            energy_levels = self.actions_config["lever_levels"]["energy"]
            energy_choice = 2 if len(energy_levels) > 1 else 0
            energy_choice = min(energy_choice, len(energy_levels) - 1)
            base_action[energy_idx] = energy_choice

        while not done:
            actions = {agent_id: base_action.copy() for agent_id in obs.keys()}
            step_log = {agent_id: base_action.tolist() for agent_id in obs.keys()}

            trajectory.append(step_log)
            obs, rewards, terminated, truncated, info_dict = env.step(actions)

            for aid, r in rewards.items():
                per_agent_return[aid] += float(r)
                total_return += float(r)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            t += 1

        return trajectory, per_agent_return, total_return, t, obs, info_dict

    def rollout_greedy_actions(self, algo, policy_mapping_fn, seed=None):
        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )

        obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

        done = False
        t = 0
        trajectory = []
        per_agent_return = {f"country_{i}": 0.0 for i in range(self.env_config["N"])}

        rnn_state = {}
        for agent_id in obs.keys():
            pol_id = policy_mapping_fn(agent_id)
            pol = algo.get_policy(pol_id)
            rnn_state[agent_id] = pol.get_initial_state()

        while not done:
            actions, step_log = {}, {}

            for agent_id, agent_obs in obs.items():
                pol_id = policy_mapping_fn(agent_id)
                pol = algo.get_policy(pol_id)

                act, new_state, _ = pol.compute_single_action(
                    agent_obs,
                    state=rnn_state[agent_id],
                    explore=False,
                )
                rnn_state[agent_id] = new_state

                actions[agent_id] = np.asarray(act, dtype=np.int64)
                step_log[agent_id] = np.asarray(act).tolist()

            trajectory.append(step_log)
            obs, rewards, terminated, truncated, info_dict = env.step(actions)

            for aid, r in rewards.items():
                per_agent_return[aid] += float(r)

            done = terminated.get("__all__", False) or truncated.get("__all__", False)
            t += 1

        return trajectory, per_agent_return, t, obs, info_dict

    def print_greedy_summary(self, greedy_traj):
        if not greedy_traj:
            return {}

        lever_levels = {
            name: np.asarray(self.actions_config["lever_levels"][name], dtype=float)
            for name in self.lever_names
        }
        adaptation_levels = np.asarray(
            self.actions_config["adaptation_levels"], dtype=float
        )

        agents = [f"country_{i}" for i in range(self.env_config["N"])]
        T = len(greedy_traj)
        start_year = int(self.env_config["hist_end"]) + 1
        years = list(range(start_year, start_year + T))

        print(f"\nGreedy policy choices over {T} years ({years[0]}–{years[-1]}):")

        policy_logger = {}
        for ag in agents:
            idx_matrix = np.array(
                [np.asarray(step[ag], dtype=int) for step in greedy_traj], dtype=int
            )

            lever_series = {}
            for j, name in enumerate(self.lever_names):
                levels = lever_levels[name]
                selections = np.clip(idx_matrix[:, j], 0, len(levels) - 1)
                efforts = levels[selections]
                lever_series[name] = efforts.tolist()

            adapt_idx = np.clip(idx_matrix[:, self.adaptation_idx], 0, len(adaptation_levels) - 1)
            adapt_levels = adaptation_levels[adapt_idx]

            print(f"- {ag}:")
            for name in self.lever_names:
                series = lever_series[name]
                print(
                    f"  {name} effort (fraction): " + ", ".join(f"{v:.2f}" for v in series)
                )
            print(
                "  adaptation investment (fraction): "
                + ", ".join(f"{v:.2f}" for v in adapt_levels)
            )

            policy_logger[ag] = {
                "lever_effort_fraction": lever_series,
                "adaptation_investment_fraction": adapt_levels.tolist(),
            }

        return policy_logger

    def register_env_rl(self):
        def env_creator(cfg):
            return ClimateMARL(
                cfg["env_config"],
                cfg["emission_data"],
                cfg["economics_config"],
                cfg["actions_config"],
            )

        register_env("climate_marl", env_creator)

    def run_marl_experiment(self):
        self.register_env_rl()

        env = ClimateMARL(
            self.env_config,
            self.emission_data,
            self.economics_config,
            self.actions_config,
        )
        obs_sp = env.observation_space("country_0")
        act_sp = env.action_space("country_0")
        N = self.env_config["N"]

        policies = {f"country_{i}": (None, obs_sp, act_sp, {}) for i in range(N)}
        policy_mapping_fn = lambda agent_id, *_, **__: agent_id  # noqa: E731

        ppo_cfg = self.training_cfg.get("ppo", {})
        model_cfg = dict(ppo_cfg.get("model", {}))
        model_cfg.setdefault("use_lstm", True)
        model_cfg.setdefault("vf_share_layers", False)
        model_cfg.setdefault("lstm_cell_size", 64)
        model_cfg.setdefault("max_seq_len", self.env_config["horizon"])

        train_batch_multiplier = ppo_cfg.get("train_batch_multiplier", 8)
        minibatch_multiplier = ppo_cfg.get("minibatch_multiplier", 2)
        train_batch_size = train_batch_multiplier * N * self.env_config["horizon"]
        minibatch_size = minibatch_multiplier * N * self.env_config["horizon"]

        lr_schedule = ppo_cfg.get("lr_schedule")
        entropy_schedule = ppo_cfg.get("entropy_coeff_schedule")
        rollout_cfg = ppo_cfg.get("rollout", {})
        resources_cfg = ppo_cfg.get("resources", {})

        fragment_length = rollout_cfg.get("fragment_length", self.env_config["horizon"])
        if isinstance(fragment_length, str) and fragment_length.lower() == "horizon":
            fragment_length = self.env_config["horizon"]

        base_config = (
            PPOConfig()
            .framework(ppo_cfg.get("framework", "torch"))
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(
                env="climate_marl",
                env_config={
                    "env_config": self.env_config,
                    "emission_data": self.emission_data,
                    "economics_config": self.economics_config,
                    "actions_config": self.actions_config,
                },
            )
            .training(
                model=model_cfg,
                train_batch_size=train_batch_size,
                minibatch_size=minibatch_size,
                num_epochs=ppo_cfg.get("num_epochs", 5),
                gamma=ppo_cfg.get("gamma", 0.999),
                lr=ppo_cfg.get("lr", 2e-4),
                clip_param=ppo_cfg.get("clip_param", 0.3),
                entropy_coeff=ppo_cfg.get("entropy_coeff", 0.02),
                lr_schedule=lr_schedule,
                entropy_coeff_schedule=entropy_schedule,
            )
            .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
            .resources(num_gpus=resources_cfg.get("num_gpus", 0))
            .env_runners(
                rollout_fragment_length=fragment_length,
                batch_mode="complete_episodes",
                num_env_runners=rollout_cfg.get("num_env_runners", 1),
                num_envs_per_env_runner=rollout_cfg.get("num_envs_per_runner", 1),
                num_gpus_per_env_runner=rollout_cfg.get("num_gpus_per_env_runner", 0),
                sample_timeout_s=rollout_cfg.get("sample_timeout_s", 1200),
            )
        )

        base_config.seed = self.training_cfg.get("seed", 0)
        algo = base_config.build_algo()


        results_dir = Path(self.env_config["output_dir"])

        with open(results_dir / "env_config.json", "w") as f:
            json.dump(_to_jsonable(self.env_config), f, indent=4)
        with open(results_dir / "economics_config.json", "w") as f:
            json.dump(_to_jsonable(self.economics_config), f, indent=4)
        with open(results_dir / "actions_config.json", "w") as f:
            json.dump(_to_jsonable(self.actions_config), f, indent=4)
        with open(results_dir / "training_config.json", "w") as f:
            json.dump(_to_jsonable(self.training_cfg), f, indent=4)

        num_iterations = self.training_cfg.get("num_iterations", 150)
        eval_interval = max(1, self.training_cfg.get("greedy_eval_interval", 2))

        num_env_steps = 0
        per_agent_reward_logger_train = {}
        per_agent_reward_logger_greedy = {}
        per_agent_policy_logger_greedy = {}
        temperature_logger = {}
        training_time_stats = {}
        t0 = time.time()

        for i in range(num_iterations):
            # trajectory, per_agent_return, _, _, _, info_dict = self.rollout_fixed_actions()
            # print("[fixed] per-agent returns:", {k: round(v, 2) for k, v in per_agent_return.items()})
            # print("[fixed] temperature trajectory:", info_dict['country_0']['Temperature_trajectory'])
            # self.print_greedy_summary(trajectory)

            result = algo.train()
            per_agent_rewards = result["env_runners"]["policy_reward_mean"]
            num_env_steps += result["env_runners"]["episodes_timesteps_total"]

            per_agent_reward_logger_train[num_env_steps] = per_agent_rewards
            print(f"Per-agent eval rewards at iteration {i} after {num_env_steps} timesteps")
            for policy_id, mean_rew in per_agent_rewards.items():
                print(f"  {policy_id}: {mean_rew:.4f}")

            training_time_stats[num_env_steps] = time.time() - t0

            if i % eval_interval == 0:
                traj, agent_ret, _, _, info_dict = self.rollout_greedy_actions(algo, policy_mapping_fn)
                per_agent_reward_logger_greedy[num_env_steps] = agent_ret
                print("[greedy] per-agent returns:", {k: round(v, 2) for k, v in agent_ret.items()})
                print("[greedy] temperature trajectory:", info_dict['country_0']['Temperature_trajectory'])
                policy_logger = self.print_greedy_summary(traj)
                per_agent_policy_logger_greedy[num_env_steps] = policy_logger
                temperature_logger[num_env_steps] = info_dict['country_0']['Temperature_trajectory']

                intermediate = {
                    "train_reward": per_agent_reward_logger_train,
                    "greedy_reward": per_agent_reward_logger_greedy,
                    "temperature_trajectory": temperature_logger,
                    "greedy_policy": per_agent_policy_logger_greedy,
                    "training_time_stats": training_time_stats,
                }
                with open(results_dir / "marl_experiment_results_intermediate.json", "w") as f:
                    json.dump(intermediate, f, indent=4)

            print("=" * 60)

        final_results = {
            "train_reward": per_agent_reward_logger_train,
            "greedy_reward": per_agent_reward_logger_greedy,
            "temperature_trajectory": temperature_logger,
            "greedy_policy": per_agent_policy_logger_greedy,
            "training_time_stats": training_time_stats,
        }
        with open(results_dir / "marl_experiment_results.json", "w") as f:
            json.dump(final_results, f, indent=4)


def main():
    marl_config = load_yaml_config("marl.yaml", "marl")
    env_config, emission_data, economics_config, actions_config, training_cfg = load_marl_setup(marl_config)

    experiment = ClimateMarlExperiment(
        env_config,
        emission_data,
        economics_config,
        actions_config,
        training_cfg,
    )
    experiment.run_marl_experiment()


if __name__ == "__main__":
    main()
