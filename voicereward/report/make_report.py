import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

REPO = os.environ.get("REPO_URL", "the GitHub repository link submitted with this report")
ss = getSampleStyleSheet()
body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10.5, leading=15, spaceAfter=6, alignment=4)
bul = ParagraphStyle("bl", parent=body, leftIndent=14, bulletIndent=2, spaceAfter=3, alignment=0)
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14, spaceBefore=10, spaceAfter=6, keepWithNext=1)
title = ParagraphStyle("t", parent=ss["Title"], fontSize=20, leading=24)
center = ParagraphStyle("c", parent=ss["Normal"], alignment=1, fontSize=11, leading=15)
mono = ParagraphStyle("m", parent=ss["Code"], fontSize=9, leading=12, alignment=1, spaceAfter=8)
small = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, leading=12, textColor=colors.HexColor("#444444"))
cell = ParagraphStyle("cell", parent=ss["Normal"], fontSize=9.5, leading=12)

def tbl(data, widths):
    data = [[Paragraph(str(c), cell) for c in row] for row in data]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t

P = lambda t: Paragraph(t, body)
B = lambda items: [Paragraph(i, bul, bulletText="\u2022") for i in items]
s = [Spacer(1, 1.2 * cm),
     Paragraph("VoiceReward: Training a Reinforcement Learning Agent Through Spoken Human Feedback", title), Spacer(1, 0.3 * cm),
     Paragraph("UCS760 - Edge AI and Robotics: Reinforcement Learning &amp; Conversational AI<br/>Lab Project Proposal (Idea Submission) - November 2026", center), Spacer(1, 0.4 * cm),
     tbl([["Name", "Roll Number"], ["Aditya Bhatoa", "102323014"], ["Avtesh Singh", "102323016"], ["Meharpreet Singh", "102323017"]], [8 * cm, 5 * cm]),
     Spacer(1, 0.3 * cm)]

s += [Paragraph("1. Problem Statement", h1), P(
 "In reinforcement learning (RL) an agent learns by receiving rewards, but the reward function usually has to be written by hand. This is slow, needs expert knowledge, and is easy to get wrong. "
 "People, on the other hand, naturally teach by giving feedback such as \"good\", \"no\" or \"stop\". This project proposes to let a human train an RL agent <b>by speaking to it</b>, "
 "and to study how well spoken feedback works as a reward signal.")]

s += [Paragraph("2. Proposed Idea", h1), P(
 "Spoken feedback is converted to text by automatic speech recognition (ASR), classified as praise or criticism by a sentiment step, and mapped to a numeric reward that the agent learns from.")]
s += [Paragraph("microphone -> ASR (e.g. Whisper) -> sentiment / keyword classifier -> reward (+1 / -1 / 0) -> RL update -> agent acts -> human watches and speaks again", mono)]
s += [P("The project connects both halves of the course: reinforcement learning (Temporal-Difference learning, Q-learning, Gymnasium) and conversational AI (speech recognition and sentiment analysis).")]

s += [Paragraph("3. Dataset and Environment", h1),
 tbl([["Resource", "Purpose", "Link / location"],
      ["Gymnasium CartPole-v1 (environment, no download needed)", "Test environment for the RL agent. It produces the agent's states and episodes; performance is the number of steps the pole stays up (max 500).", "gymnasium.farama.org/environments/classic_control/cart_pole/"],
      ["Feedback phrase set (created by the team): 20 labelled phrases, 10 positive and 10 negative", "Defines the words a speaker may say and their reward label (+1 / -1). Used by a simulated teacher and to test the sentiment step.", "data/feedback_phrases.csv in our repository"],
      ["Google Speech Commands (optional, public)", "Real recorded short words (such as \"yes\", \"no\", \"stop\") to test the ASR step with real voices.", "Hugging Face dataset: speech_commands"]],
     [5.2 * cm, 6.6 * cm, 4.4 * cm]),
 Spacer(1, 4),
 Paragraph("Note: the project is mainly about interaction with an environment, so the main data is generated during training. We plan to also record a small set of our own voice feedback clips for testing.", small)]

s += [Paragraph("4. What We Plan To Do", h1)]
s += B([
 "<b>Phase 1 - Baseline agent:</b> tabular Q-learning on CartPole-v1 with the standard environment reward.",
 "<b>Phase 2 - Feedback module:</b> turn text into a reward (keyword lexicon first, optional BERT sentiment classifier later).",
 "<b>Phase 3 - Simulated teacher:</b> a scripted \"human\" that speaks occasionally and sometimes mishears (simulated ASR error), so experiments are repeatable.",
 "<b>Phase 4 - Compare three reward settings:</b> (a) environment reward only, (b) voice reward only, (c) hybrid (environment + voice). Run several random seeds and compare steps balanced and speed of learning.",
 "<b>Phase 5 - Live demo:</b> a person types or speaks feedback while the agent trains, using Whisper (faster-whisper) for ASR.",
 "<b>Phase 6 - Real-voice test (optional):</b> test the ASR + sentiment step on Speech Commands and our own recordings; vary noise and ASR error rate.",
 "<b>Phase 7 - Extensions (if time):</b> a harder task (continuous control with PPO/SAC), and running a quantised ASR model on an edge device such as an NVIDIA Jetson to measure latency and memory."])

s += [Paragraph("5. Evaluation Plan", h1)]
s += B([
 "Steps balanced per episode (true performance, independent of which reward was used for training).",
 "Average of the last 100 episodes, and episodes needed to reach a 100-step average.",
 "Mean and standard deviation over multiple random seeds, with a significance test (e.g. Welch t-test).",
 "Sensitivity to how often the teacher speaks and to the ASR error rate.",
 "For the speech step: accuracy of mapping spoken words to the correct reward sign."])

s += [Paragraph("6. Current Status", h1), P(
 "A starter implementation (Q-learning agent, feedback-to-reward module, simulated teacher, training script and a live typed/microphone demo script) has been written in the repository (" + REPO + "). "
 "<b>No experimental results are claimed in this proposal.</b> The experiments in Section 4 are planned work, and the code has not yet been formally evaluated.")]

s += [Paragraph("7. Expected Outcomes and Risks", h1)]
s += B([
 "<b>Expected deliverables:</b> working code, a feedback-phrase dataset, a comparison of the three reward settings, a live demo, and a final report.",
 "<b>We do not assume voice rewards will help.</b> Spoken feedback may be noisy, delayed or inconsistent, and it is possible that it does not improve learning. Reporting that honestly is also a valid result.",
 "<b>Risks:</b> delay between an action and the human's comment; ASR errors with accents and background noise; a keyword sentiment model that is too simple; CartPole is a small testbed."])

s += [Paragraph("References", h1)]
for r in ["[1] R. S. Sutton and A. G. Barto, <i>Reinforcement Learning: An Introduction</i>, 2nd ed., MIT Press, 2018.",
          "[2] W. B. Knox and P. Stone, \"Interactively shaping agents via human reinforcement: the TAMER framework,\" Proc. K-CAP, 2009.",
          "[3] A. Radford et al., \"Robust speech recognition via large-scale weak supervision\" (Whisper), 2022.",
          "[4] G. Brockman et al., \"OpenAI Gym,\" arXiv:1606.01540, 2016; Farama Foundation, Gymnasium.",
          "[5] P. Warden, \"Speech Commands: A dataset for limited-vocabulary speech recognition,\" arXiv:1804.03209, 2018."]:
    s += [Paragraph(r, small), Spacer(1, 3)]

def footer(c, d):
    c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
    c.drawCentredString(A4[0] / 2, 1.2 * cm, f"VoiceReward - UCS760 Lab Project Proposal - page {d.page}")

SimpleDocTemplate("VoiceReward_Proposal.pdf", pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                  title="VoiceReward Project Proposal", author="Aditya Bhatoa, Avtesh Singh, Meharpreet Singh").build(s, onFirstPage=footer, onLaterPages=footer)
