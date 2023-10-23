# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
# This file has the CLEANRL_LICENSE
import argparse
import datetime
import logging
import os
import pathlib
import pprint
import random
import time
from collections import deque
from distutils.util import strtobool
from typing import Optional

import comet_ml
import gymnasium as gym
import hydra
import models.utils as utils
import numpy as np
import polars as pl
import stable_baselines3 as sb3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment import CrisisEnv, CurriculumCrisisEnv, SimilarityCrisisEnv
from omegaconf import DictConfig
from stable_baselines3.common.buffers import ReplayBuffer


@hydra.main(config_path="config", config_name="dqn", version_base=None)
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)


def train(cfg: DictConfig):
    run_name = f"CrisiEnv_DQN_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

    if cfg.logger.experiment_key:
        experiment = comet_ml.ExistingExperiment(**cfg.logger)
    else:
        experiment = comet_ml.Experiment(**cfg.logger)
        experiment.set_name(run_name)
        experiment.log_parameters(dict(cfg))

    # TRY NOT TO MODIFY: seeding
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(cfg.trainer.accelerator)

    envs = gym.vector.SyncVectorEnv([lambda: _create_env(cfg.environment.train, seed)])

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), **cfg.optimizer)
    # OneCycle
    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.optimizer.lr,
        total_steps=cfg.trainer.total_timesteps,
        pct_start=0.01,
        div_factor=10,
        final_div_factor=10000,
    )
    # Polynomial
    lr_scheduler = optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=cfg.trainer.total_timesteps,
    )
    # Constant
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _: cfg.optimizer.lr,
    )
    current_step = 0
    if cfg.checkpoint:
        logging.info(f"Loading checkpoint")
        state_dicts = torch.load(cfg.checkpoint, map_location=device)
        q_network.load_state_dict(state_dicts["model_state_dict"])
        optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
        current_step = 0  # int(pathlib.Path(cfg.checkpoint).stem)
        cfg.optimizer.lr = state_dicts["optimizer_state_dict"]["param_groups"][0]["lr"]

    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    experiment.set_model_graph(str(q_network))

    rb = ReplayBuffer(
        10000,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    top_k_rewards = [(-np.inf, None)]

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=seed)
    for global_step in range(current_step, cfg.trainer.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(
            cfg.policy.start_e,
            cfg.policy.end_e,
            cfg.policy.exploration_fraction * cfg.trainer.total_timesteps,
            global_step,
        )
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                logging.info(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                experiment.log_metric(
                    "episodic_return", info["episode"]["r"], global_step
                )
                experiment.log_metric(
                    "episodic_length", info["episode"]["l"], global_step
                )
                experiment.log_metric("epsilon", epsilon, global_step)
                # Summary stats
                scores = info["score"]
                useful_values = np.count_nonzero(scores) / len(scores) if scores else 0
                experiment.log_metric("useless_taken", 1 - useful_values, global_step)
                experiment.log_metric("summary_lenght", len(scores), global_step)
                experiment.log_metric(
                    "taken_percentage", len(scores) / info["episode"]["l"], global_step
                )

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.policy.learning_starts:
            if global_step % cfg.policy.train_frequency == 0:
                data = rb.sample(cfg.trainer.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = (
                        data.rewards.flatten()
                        + cfg.policy.gamma * target_max * (1 - data.dones.flatten())
                    )
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    experiment.log_metric("td_loss", loss, global_step)
                    experiment.log_metric(
                        "q_values", old_val.mean().item(), global_step
                    )
                    experiment.log_metric(
                        "max_q_value", old_val.abs().max().item(), global_step
                    )
                    logging.info(
                        f"SPS: {int(global_step / (time.time() - start_time))}"
                    )
                    experiment.log_metric(
                        "SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            # update target network
            if global_step % cfg.policy.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(
                    target_network.parameters(), q_network.parameters()
                ):
                    target_network_param.data.copy_(
                        cfg.policy.tau * q_network_param.data
                        + (1.0 - cfg.policy.tau) * target_network_param.data
                    )

        if global_step % cfg.trainer.save_frequency == 0:
            os.makedirs(f"weights/{run_name}", exist_ok=True)
            mean_eval_reward = np.mean(evaluate(cfg, q_network))
            experiment.log_metric("mean_eval_reward", mean_eval_reward, global_step)
            # If the mean reward is better than any of the previous top k rewards, remove the worst one and save the new one
            save_name = f"weights/{run_name}/{global_step}_{mean_eval_reward}.pth"
            if mean_eval_reward > top_k_rewards[0][0]:
                top_k_rewards.append((mean_eval_reward, save_name))
                top_k_rewards = sorted(top_k_rewards, key=lambda x: x[0], reverse=True)[
                    :3
                ]
            if save_name in [x[1] for x in top_k_rewards]:
                torch.save({"model_state_dict": q_network.state_dict()}, save_name)
            torch.save(
                {
                    "model_state_dict": q_network.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"weights/{run_name}/last_{global_step}.pth",
            )

    # Log the last weights to comet.ml
    weights = pathlib.Path(f"weights/{run_name}").glob("*.pth")
    last = max(weights, key=lambda x: float(x.stem))
    experiment.log_model("model", str(last))

    envs.close()
    experiment.end()


def evaluate(cfg: DictConfig, q_network: Optional[nn.Module] = None):
    envs = gym.vector.SyncVectorEnv([lambda: _create_env(cfg.environment.test, 42)])
    device = cfg.trainer.accelerator
    model = QNetwork(envs).to(device) if q_network is None else q_network
    if cfg.checkpoint:
        model.load_state_dict(
            torch.load(cfg.checkpoint, map_location=device)["model_state_dict"]
        )

    with torch.no_grad():
        obs, _ = envs.reset(seed=42)
        episodic_returns = []
        all_q_values = []
        action0_steps = []
        step = 0
        while len(episodic_returns) < len(envs.envs[0].events):
            if random.random() < cfg.policy.eval_epsilon:
                actions = np.array(
                    [envs.single_action_space.sample() for _ in range(envs.num_envs)]
                )
            else:
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()
                if actions == 0:
                    all_q_values += [
                        q_values[0, 0].cpu().item() - q_values[0, 1].cpu().item()
                    ]
                    action0_steps.append(step)
            next_obs, _, term, _, infos = envs.step(actions)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if "episode" not in info:
                        continue
                    print(
                        f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}"
                    )
                    episodic_returns += [info["episode"]["r"]]

                    # Write to file in TREC format
                    current_event = info["event"]
                    summary_folder = pathlib.Path(f"summaries/{current_event.id}")
                    summary_folder.mkdir(parents=True, exist_ok=True)
                    pl.DataFrame(
                        {
                            "text": info["text"],
                            "score": info["score"],
                            "q_value": all_q_values,
                            "step": action0_steps,
                        }
                    ).with_row_count().write_ndjson(
                        summary_folder / f"dqn_{current_event.date}.jsonl"
                    )
                    action0_steps.clear()
                    all_q_values.clear()
            obs = next_obs
            step += 1

    return episodic_returns


def _create_env(cfg, seed: int = 0):
    if cfg.mode in ("eval", "train"):
        env = SimilarityCrisisEnv(**cfg)
    else:
        env = CurriculumCrisisEnv(
            **cfg, curriculum_steps=[200000, 400000, 600000, 800000]
        )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env: gym.vector.SyncVectorEnv):
        super().__init__()
        self.network = utils.MLP(
            env.single_observation_space.shape,
            env.single_action_space.n,
            hidden_sizes=[770, 770],  # [512, 256, 128, 64],
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration if duration > 0 else 0
    return max(slope * t + start_e, end_e) if duration > 0 else end_e


if __name__ == "__main__":
    main()
