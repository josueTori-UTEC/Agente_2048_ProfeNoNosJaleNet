from __future__ import annotations

import argparse
import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from game_2048 import Game2048

def make_seeds(n: int, seed0: int) -> List[int]:
    rng = np.random.default_rng(seed0)
    return rng.integers(0, 2**31 - 1, size=n, dtype=np.int64).tolist()


def load_agent(agent_module: str, agent_class: str, seed: Optional[int]) -> object:
    mod = importlib.import_module(agent_module)
    cls = getattr(mod, agent_class)
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()

def evaluate_agent_scalar(
    agent: object,
    seeds: List[int],
    size: int = 4,
    max_steps: int = 5000,
) -> Dict[str, float]:
    scores = []
    max_tiles = []
    steps_list = []

    for sd in seeds:
        game = Game2048(size=size, seed=int(sd))
        score = 0
        steps = 0

        while True:
            legal = game.legal_actions()
            if not legal:
                break

            action = agent.act(game.board.copy(), legal)
            result = game.step(action)

            if result.info.get("moved", False):
                score += int(result.reward)

            steps += 1
            if result.done or steps >= max_steps:
                break

        scores.append(score)
        max_tiles.append(int(game.board.max()))
        steps_list.append(steps)

    scores = np.array(scores, dtype=np.float64)
    max_tiles = np.array(max_tiles, dtype=np.float64)
    steps_list = np.array(steps_list, dtype=np.float64)

    L = np.log1p(scores)                         # log(1 + score)
    T = np.log2(np.maximum(max_tiles, 1.0))      # log2(max_tile), safe
    K = np.log1p(steps_list)                     # log(1 + steps)

    L_mean = float(L.mean())
    L_med = float(np.median(L))
    T_mean = float(T.mean())
    K_mean = float(K.mean())

    final_score = 1000.0 * L_mean + 30.0 * T_mean + 10.0 * L_med - 2.0 * K_mean

    return {
        "final_score": float(final_score),
        "mean_log_score": L_mean,
        "median_log_score": L_med,
        "mean_log2_max_tile": T_mean,
        "mean_log_steps": K_mean,
        "episodes": float(len(seeds)),
        "mean_raw_score": float(scores.mean()),
        "median_raw_score": float(np.median(scores)),
        "mean_max_tile": float(max_tiles.mean()),
        "max_max_tile": float(max_tiles.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent-module", type=str, required=True, help="e.g., agent_submission")
    p.add_argument("--agent-class", type=str, default="Agent", help="default: Agent")
    p.add_argument("--size", type=int, default=4)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed0", type=int, default=67, help="seed to generate the evaluation seed list")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--agent-seed", type=int, default=None, help="seed passed to agent constructor (optional)")
    args = p.parse_args()

    seeds = make_seeds(args.episodes, args.seed0)
    agent = load_agent(args.agent_module, args.agent_class, seed=args.agent_seed)

    if not hasattr(agent, "act"):
        raise AttributeError("Agent must implement method: act(board, legal_actions) -> str")

    metrics = evaluate_agent_scalar(agent, seeds, size=args.size, max_steps=args.max_steps)

    print("Evaluation ==============")
    print(f"Agent: {args.agent_module}.{args.agent_class}")
    print(f"Episodes: {args.episodes} | Board: {args.size}x{args.size} | seed0: {args.seed0}")
    print(f"FinalScore: {metrics['final_score']:.6f}")
    print(f"MeanScore(raw): {metrics['mean_raw_score']:.2f} | MedianScore(raw): {metrics['median_raw_score']:.2f}")
    print(f"MeanMaxTile: {metrics['mean_max_tile']:.2f} | MaxMaxTile: {metrics['max_max_tile']:.0f}")
    print(
        "Components: "
        f"mean_log_score={metrics['mean_log_score']:.6f}, "
        f"median_log_score={metrics['median_log_score']:.6f}, "
        f"mean_log2_max_tile={metrics['mean_log2_max_tile']:.6f}, "
        f"mean_log_steps={metrics['mean_log_steps']:.6f}"
    )


if __name__ == "__main__":
    main()