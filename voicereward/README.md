# VoiceReward: Training a Reinforcement Learning Agent Through Spoken Human Feedback

**Course:** UCS760 - Edge AI and Robotics: Reinforcement Learning & Conversational AI

| Name | Roll No. |
|---|---|
| Aditya Bhatoa | 102323014 |
| Avtesh Singh | 102323016 |
| Meharpreet Singh | 102323017 |

## Status: project proposal / starter code
This repository accompanies our **idea submission**. It contains the proposal report and starter code.
**No experimental results are claimed yet.** The experiments listed below are planned work.

## Idea
RL agents normally need a hand-written reward function. We want a human to train the agent **by speaking**:

```
microphone -> ASR (Whisper) -> sentiment / keyword classifier -> reward (+1 / -1 / 0) -> Q-learning update -> agent acts
```

## Dataset / environment
- **Environment:** Gymnasium `CartPole-v1` (installed with `pip install gymnasium`, no download).
- **Feedback phrases:** `data/feedback_phrases.csv` - 20 phrases (10 positive, 10 negative) created by the team.
- **Optional, for ASR tests:** Google Speech Commands (Hugging Face `speech_commands`) and our own recorded clips.

## Planned work
1. Baseline Q-learning agent on CartPole with the environment reward
2. Feedback module: text -> reward (keyword lexicon, optional BERT)
3. Simulated teacher (speaks occasionally, simulated ASR errors) for repeatable experiments
4. Compare 3 reward settings over several seeds: environment-only, voice-only, hybrid
5. Live demo with typed and microphone feedback
6. Real-voice test of ASR + sentiment (optional)
7. Extensions: harder task (PPO/SAC), quantised ASR on an edge device (Jetson)

Evaluation: steps balanced, last-100-episode average, episodes to reach a 100-step average, mean/std over seeds, significance test.
We do not assume voice rewards will help; honest reporting of any outcome is part of the plan.

## Repository layout
```
src/feedback.py     speech text -> reward, simulated teacher
src/agent.py        tabular Q-learning agent
src/train.py        experiment runner for the 3 reward settings (starter code, not yet evaluated)
src/live_voice.py   live demo (typed or microphone feedback)
data/feedback_phrases.csv
report/             proposal report (PDF) and its generator
```

## How to run the starter code
```bash
pip install -r requirements.txt
cd src
python train.py --episodes 2000 --seeds 10 --out curves.csv
python live_voice.py --typed
```
