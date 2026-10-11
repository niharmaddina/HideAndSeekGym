"""Train PPO on HideAndSeekEnv and evaluate it.

    python train.py --timesteps 50000 --seed 0 --out runs/seed0

Writes <out>/monitor.csv (episode returns and lengths, from stable-baselines3's Monitor wrapper),
<out>/model.zip and <out>/eval.json (mean and std return over --eval-episodes greedy episodes).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from agentsim import HideAndSeekEnv


def evaluate(model, env, episodes: int, seed: int) -> dict:
    returns = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return {"episodes": episodes, "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/seed0")
    ap.add_argument("--max-steps", type=int, default=500, help="episode length cap")
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_env(HideAndSeekEnv(max_steps=args.max_steps))
    env = Monitor(HideAndSeekEnv(max_steps=args.max_steps), filename=str(out / "monitor"))

    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, seed=args.seed, verbose=0)
    model.learn(total_timesteps=args.timesteps)
    model.save(out / "model")

    result = evaluate(model, HideAndSeekEnv(max_steps=args.max_steps), args.eval_episodes, args.seed)
    (out / "eval.json").write_text(json.dumps({"timesteps": args.timesteps, "seed": args.seed, **result}, indent=2))
    print(json.dumps(result))
    env.close()


if __name__ == "__main__":
    main()
