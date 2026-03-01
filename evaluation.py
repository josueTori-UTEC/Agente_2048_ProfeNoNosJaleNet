import numpy as np
from game_2048 import Game2048

def evaluate_agent_scalar(agent, seeds, size=4, max_steps=5000):
    scores = []
    max_tiles = []
    steps_list = []

    for sd in seeds:
        game = Game2048(size=size, seed=sd)
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

    # Logs
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
        "episodes": int(len(seeds)),
    }
