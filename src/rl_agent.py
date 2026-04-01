from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

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
        self.q_table = np.zeros(
            (num_conf_states, num_nlp_states, num_actions),
            dtype=np.float32,
        )

    def get_state(self, confidence: float, nlp_match: int) -> Tuple[int, int]:
        conf_bucket = min(int(float(confidence) * 10), 9)
        return conf_bucket, int(nlp_match)

    def choose_action(self, confidence: float, nlp_match: int, training: bool = True) -> int:
        conf_bucket, nlp_state = self.get_state(confidence, nlp_match)
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.num_actions))
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
    """
    Reward design:
    - ACCEPT a correct prediction: positive reward
    - ACCEPT a wrong prediction: strong penalty
    - REJECT a wrong prediction: small positive reward
    - REJECT a correct prediction: small penalty
    """
    if action == ACCEPT:
        if correct == 1:
            if confidence >= 0.80 and genus_match == 1:
                return 2.0
            return 1.0
        return -2.5

    # REJECT
    if correct == 0:
        return 0.75
    return -0.5


def moving_average(values: List[float], window: int = 30) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def train_agent_from_predictions(
    predictions: List[dict],
    episodes: int = 1000,
    seed: int = 42,
) -> Tuple[ThresholdRLAgent, List[float], List[int], List[int]]:
    np.random.seed(seed)

    if not predictions:
        raise ValueError("Prediction file is empty.")

    agent = ThresholdRLAgent()
    rewards: List[float] = []
    successes: List[int] = []
    accepts: List[int] = []

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

        rewards.append(float(reward))
        accepts.append(1 if action == ACCEPT else 0)

        success = (
            (action == ACCEPT and correct == 1)
            or (action == REJECT and correct == 0)
        )
        successes.append(1 if success else 0)

    return agent, rewards, successes, accepts


def evaluate_policy(
    agent: ThresholdRLAgent,
    predictions: List[dict],
) -> Dict[str, float]:
    successes: List[int] = []
    accepts: List[int] = []
    rewards: List[float] = []

    for row in predictions:
        confidence = float(row["confidence"])
        genus_match = int(row["genus_match"])
        correct = int(row["correct"])

        action = agent.choose_action(confidence, genus_match, training=False)
        reward = compute_reward(correct, action, confidence, genus_match)

        success = (
            (action == ACCEPT and correct == 1)
            or (action == REJECT and correct == 0)
        )

        successes.append(1 if success else 0)
        accepts.append(1 if action == ACCEPT else 0)
        rewards.append(float(reward))

    return {
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "accept_rate": float(np.mean(accepts)) if accepts else 0.0,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="experiments/results/mobilenetv3_val_predictions.json",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--window", type=int, default=30)
    args = parser.parse_args()

    predictions = load_json(args.predictions_path)
    if not predictions:
        raise ValueError("Prediction file is empty.")

    seeds = [42, 123, 777]
    smoothed_runs: List[np.ndarray] = []
    per_seed_metrics: List[dict] = []
    agents: List[ThresholdRLAgent] = []

    for seed in seeds:
        agent, rewards, successes, accepts = train_agent_from_predictions(
            predictions=predictions,
            episodes=args.episodes,
            seed=seed,
        )
        agents.append(agent)

        smoothed = moving_average(rewards, window=args.window)
        smoothed_runs.append(smoothed)

        eval_metrics = evaluate_policy(agent, predictions)
        per_seed_metrics.append(
            {
                "seed": seed,
                "final_smoothed_reward": float(smoothed[-1]) if len(smoothed) else 0.0,
                "train_success_rate": float(np.mean(successes)) if successes else 0.0,
                "train_accept_rate": float(np.mean(accepts)) if accepts else 0.0,
                "eval_success_rate": eval_metrics["success_rate"],
                "eval_accept_rate": eval_metrics["accept_rate"],
                "eval_avg_reward": eval_metrics["avg_reward"],
            }
        )

    min_len = min(len(run) for run in smoothed_runs)
    smoothed_runs_arr = np.array([run[:min_len] for run in smoothed_runs], dtype=np.float32)

    mean_rewards = smoothed_runs_arr.mean(axis=0)
    std_rewards = smoothed_runs_arr.std(axis=0)

    # Use the best agent based on eval success rate
    best_idx = int(np.argmax([m["eval_success_rate"] for m in per_seed_metrics]))
    best_agent = agents[best_idx]

    results_dir = Path("experiments/results")
    logs_dir = Path("experiments/logs")
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    save_rl_curve(mean_rewards, std_rewards, str(results_dir / "rl_learning_curve.png"))

    rl_metrics = {
        "episodes": args.episodes,
        "window": args.window,
        "seeds": seeds,
        "best_seed": per_seed_metrics[best_idx]["seed"],
        "mean_final_reward": float(mean_rewards[-1]) if len(mean_rewards) else 0.0,
        "std_final_reward": float(std_rewards[-1]) if len(std_rewards) else 0.0,
        "mean_eval_success_rate": float(np.mean([m["eval_success_rate"] for m in per_seed_metrics])),
        "std_eval_success_rate": float(np.std([m["eval_success_rate"] for m in per_seed_metrics])),
        "mean_eval_accept_rate": float(np.mean([m["eval_accept_rate"] for m in per_seed_metrics])),
        "std_eval_accept_rate": float(np.std([m["eval_accept_rate"] for m in per_seed_metrics])),
        "per_seed_metrics": per_seed_metrics,
    }

    save_json(rl_metrics, results_dir / "rl_metrics.json")
    save_json(
        {
            **rl_metrics,
            "q_table": best_agent.q_table.tolist(),
        },
        logs_dir / "rl_qtable.json",
    )

    print("Saved RL curve to experiments/results/rl_learning_curve.png")
    print("Saved RL metrics to experiments/results/rl_metrics.json")
    print("Saved RL Q-table to experiments/logs/rl_qtable.json")
    print(f"Mean eval success rate: {rl_metrics['mean_eval_success_rate']:.4f}")


if __name__ == "__main__":
    main()