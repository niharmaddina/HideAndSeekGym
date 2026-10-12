"""Reference policies on the evaluation seeds train.py uses (seed + 10000 + i, so 10000..10019 by default).

    python scripts/baselines.py                       # seeker_speed 0, 0.05 and 0.1
    python scripts/baselines.py --seeker-speed 0.2    # one setting

For each setting prints mean +/- std return, mean episode length and number of catches over
--episodes episodes for two policies:

  random     uniform random actions, drawn from the environment's own seeded RNG (so the numbers
             are reproducible; the 303 +/- 145 quoted in the first README tables used an unseeded
             sampler and is a few points lower by chance)
  heuristic  greedy move-away: the action that maximises the distance to the seeker right now,
             ignoring the seeker's reply
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentsim import HideAndSeekEnv, _MOVES  # noqa: E402


def random_policy(env: HideAndSeekEnv, obs: np.ndarray) -> int:
    return int(env.np_random.integers(env.action_space.n))


def greedy_move_away(env: HideAndSeekEnv, obs: np.ndarray) -> int:
    hider, seeker = obs[:3].astype(np.float64), obs[3:].astype(np.float64)
    best, best_distance = 0, -1.0
    for action, (axis, direction) in _MOVES.items():
        candidate = hider.copy()
        candidate[axis] += direction * env.increment
        candidate = np.clip(candidate, 0.0, env.stage_size)
        distance = float(np.linalg.norm(candidate - seeker))
        if distance > best_distance:
            best, best_distance = action, distance
    return best


def run(policy, env: HideAndSeekEnv, seeds) -> dict:
    returns, lengths, caught = [], [], 0
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        total, steps, done = 0.0, 0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(env, obs))
            total += reward
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        caught += int(terminated)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)), "caught": caught}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeker-speed", type=float, nargs="*", default=[0.0, 0.05, 0.1])
    ap.add_argument("--increment", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0, help="train.py --seed whose evaluation seeds to use")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    seeds = [args.seed + 10_000 + i for i in range(args.episodes)]
    print(f"{args.episodes} episodes, seeds {seeds[0]}..{seeds[-1]}, increment {args.increment}, max_steps {args.max_steps}\n")
    print("| seeker_speed | policy | return | episode length | caught |")
    print("|---|---|---|---|---|")
    for speed in args.seeker_speed:
        env = HideAndSeekEnv(seeker_speed=speed, increment=args.increment, max_steps=args.max_steps)
        for name, policy in (("random", random_policy), ("heuristic", greedy_move_away)):
            r = run(policy, env, seeds)
            print(f"| {speed:g} | {name} | {r['mean_return']:.0f} ± {r['std_return']:.0f} | {r['mean_length']:.0f} | {r['caught']}/{args.episodes} |")


if __name__ == "__main__":
    main()
