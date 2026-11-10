"""Live demo: train the CartPole agent with YOUR voice (or typed words) as the reward.

Usage:
  python live_voice.py --typed            # type 'good' / 'bad' + Enter while it runs (no mic needed)
  python live_voice.py --mic              # speak into the microphone (needs sounddevice + faster-whisper)
  add --no-render to run without a window.

Each heard phrase becomes a reward (+1/-1) applied to the agent's next few steps.
"""
import argparse, threading, queue, time, sys
import numpy as np
import gymnasium as gym
from agent import QAgent, discretize
from feedback import text_to_reward

heard = queue.Queue()


def typed_listener():
    for line in sys.stdin:
        heard.put(line.strip())


def mic_listener(seconds=2, sr=16000):
    import sounddevice as sd
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny.en", compute_type="int8")
    while True:
        audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        segs, _ = model.transcribe(audio[:, 0], language="en")
        text = " ".join(s.text for s in segs).strip()
        if text:
            print(f'  heard: "{text}"', flush=True)
            heard.put(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--typed", action="store_true")
    p.add_argument("--mic", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--hold", type=int, default=10, help="steps a spoken reward stays active")
    a = p.parse_args()
    target = mic_listener if a.mic else typed_listener
    threading.Thread(target=target, daemon=True).start()

    env = gym.make("CartPole-v1", render_mode=None if a.no_render else "human")
    agent = QAgent()
    pending, hold_left = 0.0, 0
    for ep in range(a.episodes):
        eps = max(0.02, 0.995 ** ep)
        obs, _ = env.reset()
        s, steps, done = discretize(obs), 0, False
        while not done:
            while not heard.empty():
                pending = text_to_reward(heard.get()); hold_left = a.hold
                print(f"  reward {pending:+.0f} for next {a.hold} steps", flush=True)
            act = agent.act(s, eps)
            obs, env_r, term, trunc, _ = env.step(act)
            r = env_r + (pending if hold_left > 0 else 0.0)
            hold_left = max(0, hold_left - 1)
            s2 = discretize(obs)
            agent.update(s, act, r, s2, term)
            s, steps, done = s2, steps + 1, term or trunc
            if not a.no_render:
                time.sleep(0.01)
        print(f"episode {ep}: balanced {steps} steps", flush=True)


if __name__ == "__main__":
    main()
