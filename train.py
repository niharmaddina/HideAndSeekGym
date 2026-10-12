"""Train PPO on HideAndSeekEnv and evaluate it.

    python train.py --timesteps 50000 --seed 0 --out runs/seed0
    python train.py --timesteps 300000 --seed 0 --seeker-speed 0.05 --normalize --out runs/moving005_seed0

Writes <out>/monitor.csv (episode returns and lengths, from stable-baselines3's Monitor wrapper),
<out>/model.zip and <out>/eval.json (mean and std return over --eval-episodes greedy episodes).
With --normalize the observations are normalised with VecNormalize (rewards are not); the running
statistics are saved to <out>/vecnormalize.pkl and reloaded for the evaluation, which is how the
saved model has to be used.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agentsim import HideAndSeekEnv


def make_env(args, monitor_path: str | None = None):
    env = HideAndSeekEnv(max_steps=args.max_steps, increment=args.increment, seeker_speed=args.seeker_speed)
    return Monitor(env, filename=monitor_path) if monitor_path else env


def load_normalizer(path: Path, args) -> VecNormalize:
    """Observation statistics saved by --normalize, frozen for evaluation."""
    normalizer = VecNormalize.load(str(path), DummyVecEnv([lambda: make_env(args)]))
    normalizer.training = False
    normalizer.norm_reward = False
    return normalizer


def evaluate(model, env, episodes: int, seed: int, normalizer: VecNormalize | None = None) -> dict:
    returns, lengths, caught = [], [], 0
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + i)
        done, total, steps = False, 0.0, 0
        while not done:
            model_obs = normalizer.normalize_obs(obs) if normalizer is not None else obs
            action, _ = model.predict(model_obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        caught += int(terminated)
    return {"episodes": episodes, "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)), "caught": caught}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/seed0")
    ap.add_argument("--max-steps", type=int, default=500, help="episode length cap")
    ap.add_argument("--increment", type=float, default=0.1, help="hider step size in metres")
    ap.add_argument("--seeker-speed", type=float, default=0.0,
                    help="metres the seeker moves toward the hider per step (0 = stationary, the original task)")
    ap.add_argument("--normalize", action="store_true",
                    help="normalise observations with VecNormalize; statistics go to <out>/vecnormalize.pkl")
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_env(make_env(args))
    env = DummyVecEnv([lambda: make_env(args, str(out / "monitor"))])
    if args.normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, seed=args.seed, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    train_seconds = time.time() - t0
    model.save(out / "model")

    normalizer = None
    if args.normalize:
        env.save(str(out / "vecnormalize.pkl"))
        normalizer = load_normalizer(out / "vecnormalize.pkl", args)

    result = evaluate(PPO.load(out / "model"), make_env(args), args.eval_episodes, args.seed, normalizer)
    config = {"timesteps": args.timesteps, "seed": args.seed, "max_steps": args.max_steps, "increment": args.increment,
              "seeker_speed": args.seeker_speed, "normalize": args.normalize, "train_seconds": round(train_seconds, 1)}
    (out / "eval.json").write_text(json.dumps({**config, **result}, indent=2))
    print(json.dumps({**config, **result}))
    env.close()


if __name__ == "__main__":
    main()
