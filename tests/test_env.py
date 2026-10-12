import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from agentsim import HideAndSeekEnv

PLUS_Y, MINUS_Y, MINUS_X, PLUS_X, PLUS_Z, MINUS_Z = range(6)


def test_reset_is_seeded_and_bodies_start_apart():
    a, b = HideAndSeekEnv(), HideAndSeekEnv()
    obs_a, _ = a.reset(seed=7)
    obs_b, _ = b.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)
    obs_c, _ = a.reset(seed=8)
    assert not np.array_equal(obs_a, obs_c)
    for seed in range(50):
        obs, info = a.reset(seed=seed)
        assert a.observation_space.contains(obs)
        assert info["distance"] >= a.catch_radius
        assert info["distance"] == pytest.approx(np.linalg.norm(obs[:3] - obs[3:]), abs=1e-5)


def test_hider_is_clipped_to_the_stage():
    env = HideAndSeekEnv()
    obs, _ = env.reset(seed=11)                      # chosen so the walk below does not run into the seeker
    for _ in range(200):                             # 200 x 0.1 m = 20 m, twice the stage
        obs, _, terminated, _, _ = env.step(PLUS_X)
        assert not terminated
        assert env.observation_space.contains(obs)
    assert obs[0] == pytest.approx(env.stage_size)
    for _ in range(200):
        obs, _, terminated, _, _ = env.step(MINUS_Z)
        assert not terminated
        assert env.observation_space.contains(obs)
    assert obs[2] == pytest.approx(0.0)


def test_stationary_seeker_does_not_move():
    env = HideAndSeekEnv(seeker_speed=0.0)
    obs, _ = env.reset(seed=1)
    seeker = obs[3:].copy()
    for action in range(6):
        obs, _, _, _, _ = env.step(action)
        assert np.array_equal(obs[3:], seeker)


def test_chasing_seeker_moves_seeker_speed_toward_the_hider():
    env = HideAndSeekEnv(seeker_speed=0.1)
    obs, _ = env.reset(seed=2)
    seeker_before = obs[3:].astype(np.float64)
    obs, _, _, _, _ = env.step(PLUS_Z)
    hider_after, seeker_after = obs[:3].astype(np.float64), obs[3:].astype(np.float64)
    moved = seeker_after - seeker_before
    assert np.linalg.norm(moved) == pytest.approx(0.1, abs=1e-5)
    toward = hider_after - seeker_before
    assert np.allclose(moved / np.linalg.norm(moved), toward / np.linalg.norm(toward), atol=1e-4)
    assert env.observation_space.contains(obs)


def test_chasing_seeker_catches_the_hider_and_terminates_with_penalty():
    env = HideAndSeekEnv(seeker_speed=0.5, max_steps=500)
    env.reset(seed=3)
    for t in range(env.max_steps):
        obs, reward, terminated, truncated, info = env.step(PLUS_Y)
        if terminated:
            break
    assert terminated and not truncated
    assert t + 1 < env.max_steps
    assert info["caught"] is True
    assert info["distance"] < env.catch_radius
    assert reward < 0
    assert reward == pytest.approx(info["distance"] / env.stage_size - env.catch_penalty)


def test_reward_is_positive_distance_fraction_when_not_caught():
    env = HideAndSeekEnv()
    obs, _ = env.reset(seed=4)
    for action in range(6):
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert reward > 0
        assert reward == pytest.approx(info["distance"] / env.stage_size)
        assert reward == pytest.approx(np.linalg.norm(obs[:3] - obs[3:]) / env.stage_size, abs=1e-5)
        assert reward <= np.sqrt(3) + 1e-6                      # at most the cube diagonal / stage_size


def test_truncation_at_max_steps():
    env = HideAndSeekEnv(max_steps=25)
    env.reset(seed=11)
    for t in range(25):
        _, _, terminated, truncated, _ = env.step(PLUS_X)
        assert not terminated
        assert truncated == (t == 24)


@pytest.mark.parametrize("seeker_speed", [0.0, 0.05, 0.1])
def test_check_env(seeker_speed):
    check_env(HideAndSeekEnv(seeker_speed=seeker_speed))


def test_invalid_arguments():
    with pytest.raises(ValueError):
        HideAndSeekEnv(seeker_speed=-0.1)
    with pytest.raises(ValueError):
        HideAndSeekEnv(backend="nope")
    with pytest.raises(ValueError):
        HideAndSeekEnv(render_mode="human")          # needs the pybullet backend
