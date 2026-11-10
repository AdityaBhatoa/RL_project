"""Turn spoken feedback (as text) into a numeric reward, and simulate a human teacher.

Pipeline position:  microphone -> ASR (Whisper) -> text  -> [this file] -> reward in {-1, 0, +1}
"""
import random

POSITIVE = {"good", "great", "nice", "yes", "perfect", "well", "excellent",
            "right", "awesome", "correct", "better", "keep"}
NEGATIVE = {"bad", "no", "wrong", "stop", "terrible", "worse", "fall", "falling",
            "oops", "nope", "not", "incorrect", "fail"}

GOOD_PHRASES = ["good", "good job", "great", "nice", "yes", "perfect", "well done",
                "that's right", "excellent", "keep going"]
BAD_PHRASES = ["bad", "no", "wrong", "no no", "stop", "terrible", "oops",
               "that's wrong", "not like that", "no, other way"]

_clf = None


def text_to_reward(text, backend="lexicon"):
    """Sentiment analysis step: text -> +1 (praise), -1 (criticism), 0 (neutral/unknown)."""
    global _clf
    if not text:
        return 0.0
    if backend == "bert":  # optional, heavier; needs `pip install transformers torch`
        if _clf is None:
            from transformers import pipeline
            _clf = pipeline("sentiment-analysis",
                            model="distilbert-base-uncased-finetuned-sst-2-english")
        out = _clf(text)[0]
        return 1.0 if out["label"] == "POSITIVE" else -1.0
    for ch in ",!.?":
        text = text.replace(ch, " ")
    words = text.lower().split()
    score = sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)
    return float(max(-1, min(1, score)))


class SimulatedTeacher:
    """Stands in for a human watching CartPole and speaking feedback.

    - Speaks only with probability `talk_prob` per step (humans are not constantly talking).
    - Says "good"/"bad" depending on whether the pole got more or less upright.
    - `asr_error` = chance the speech recogniser hears a random wrong phrase.
    - Always says something negative when the pole falls (the human would say "no!").
    """

    def __init__(self, talk_prob=0.3, asr_error=0.1, rng=None):
        self.talk_prob = talk_prob
        self.asr_error = asr_error
        self.rng = rng or random.Random(0)

    def speak(self, prev_obs, obs, terminated):
        if terminated:
            phrase = self.rng.choice(BAD_PHRASES)
        elif self.rng.random() > self.talk_prob:
            return ""
        else:
            better = abs(obs[2]) + 0.3 * abs(obs[3]) <= abs(prev_obs[2]) + 0.3 * abs(prev_obs[3])
            phrase = self.rng.choice(GOOD_PHRASES if better else BAD_PHRASES)
        if self.rng.random() < self.asr_error:
            phrase = self.rng.choice(GOOD_PHRASES + BAD_PHRASES)
        return phrase
