from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

from src.utils.io import load_json, save_json
from src.utils.plots import save_rl_curve


REJECT = 0
ACCEPT = 1


class ThresholdRLAgent:
    def __init__(
        self,
        num_conf_states: int = 10,
        num_nlp_states: int = 2,
        num_actions: int = 2,
        epsilon: float = 0.10,
        alpha: float = 0.10,
        gamma: float = 0.90,
    ) -> None:
        self.num_conf_states = num_conf_states
        self.num_nlp_states = num_nlp_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = np.zeros((num_conf_states, num_nlp_states, num_actions), dtype=np.float32)

    def get_state(self, confidence: float, nlp_match: int) -> Tuple[int, int]:
        conf_bucket = min(int(confidence * 10), 9)
        return conf_bucket, int(nlp_match)

    def choose_action(self, confidence: float, nlp_match: int, training: bool = True) -> int:
        conf_bucket, nlp_state = self.get_state(confidence, nlp_match)

        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        return int(np.argmax(self.q_table[conf_bucket, nlp_state]))

    def update(
        self,
        confidence: float,
        nlp_match: int,
        action: int,
        reward: float,
        next_confidence: float,
        next_nlp_match: int,
    ) -> None:
        state = self.get_state(confidence, nlp_match)
        next_state = self.get_state(next_confidence, next_nlp_match)

        current_q = self.q_table[state[0], state[1], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1]])

        self.q_table[state[0], state[1], action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )


def compute_reward(correct: int, action: int, confidence: float, genus_match: int) -> float:
    if action == ACCEPT:
        if correct == 1:
            if confidence >= 0.80 and genus_match == 1:
                return 2.0
            return 1.0
        return -2.5

    if correct == 0:
        return 0.75
    return -0.5


def moving_average(values: List[float], window: int = 30) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def train_agent_from_predictions(predictions: List[dict], episodes: int = 1000, seed: int = 42):
    np.random.seed(seed)
    agent = ThresholdRLAgent()

    rewards = []

    n = len(predictions)
    for _ in range(episodes):
        idx = np.random.randint(0, n)
        next_idx = np.random.randint(0, n)

        row = predictions[idx]
        next_row = predictions[next_idx]

        confidence = float(row["confidence"])
        genus_match = int(row["genus_match"])
        correct = int(row["correct"])

        action = agent.choose_action(confidence, genus_match, training=True)
        reward = compute_reward(correct, action, confidence, genus_match)

        next_confidence = float(next_row["confidence"])
        next_genus_match = int(next_row["genus_match"])

        agent.update(
            confidence,
            genus_match,
            action,
            reward,
            next_confidence,
            next_genus_match,
        )

        rewards.append(reward)

    return agent, rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="experiments/results/mobilenetv3_val_predictions.json",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()

    predictions = load_json(args.predictions_path)
    if not predictions:
        raise ValueError("Prediction file is empty.")

    seeds = [42, 123, 777]
    smoothed_runs = []
    agents = []

    for seed in seeds:
        agent, rewards = train_agent_from_predictions(predictions, episodes=args.episodes, seed=seed)
        agents.append(agent)
        smoothed_runs.append(moving_average(rewards, window=30))

    min_len = min(len(r) for r in smoothed_runs)
    smoothed_runs = np.array([r[:min_len] for r in smoothed_runs])

    mean_rewards = smoothed_runs.mean(axis=0)
    std_rewards = smoothed_runs.std(axis=0)

    results_dir = Path("experiments/results")
    logs_dir = Path("experiments/logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    save_rl_curve(mean_rewards, std_rewards, str(results_dir / "rl_learning_curve.png"))

    save_json(
        {
            "episodes": args.episodes,
            "seeds": seeds,
            "mean_final_reward": float(mean_rewards[-1]),
            "std_final_reward": float(std_rewards[-1]),
            "q_table": agents[0].q_table.tolist(),
        },
        logs_dir / "rl_qtable.json",
    )

    print("Saved RL Q-table to experiments/logs/rl_qtable.json")


if __name__ == "__main__":
    main()