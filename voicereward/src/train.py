"""Compare three reward sources for CartPole: environment, voice-only, hybrid (env + voice)."""
import argparse, csv, random
import numpy as np
import gymnasium as gym
from agent import QAgent, discretize
from feedback import SimulatedTeacher, text_to_reward

MODES = ["env_reward", "voice_only", "hybrid"]


def run(mode, seed, episodes, talk_prob, asr_error, voice_weight=1.0):
    env = gym.make("CartPole-v1")
    agent = QAgent(seed=seed)
    teacher = SimulatedTeacher(talk_prob, asr_error, random.Random(seed))
    lengths = []
    for ep in range(episodes):
        eps = max(0.02, 1.0 * (0.995 ** ep))
        obs, _ = env.reset(seed=seed * 100000 + ep)
        s, steps, done = discretize(obs), 0, False
        while not done:
            a = agent.act(s, eps)
            obs2, env_r, terminated, truncated, _ = env.step(a)
            voice_r = text_to_reward(teacher.speak(obs, obs2, terminated))
            if mode == "env_reward":
                r = env_r
            elif mode == "voice_only":
                r = voice_r
            else:
                r = env_r + voice_weight * voice_r
            s2 = discretize(obs2)
            agent.update(s, a, r, s2, terminated)
            obs, s, steps = obs2, s2, steps + 1
            done = terminated or truncated
        lengths.append(steps)  # TRUE performance = steps the pole stayed up
    return lengths


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--talk_prob", type=float, default=0.3)
    p.add_argument("--asr_error", type=float, default=0.1)
    p.add_argument("--out", default="curves.csv")
    a = p.parse_args()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "seed", "episode", "steps_balanced"])
        for mode in MODES:
            for seed in range(a.seeds):
                L = run(mode, seed, a.episodes, a.talk_prob, a.asr_error)
                for i, v in enumerate(L):
                    w.writerow([mode, seed, i, v])
                print(mode, seed, "last-100 mean:", np.mean(L[-100:]), flush=True)
