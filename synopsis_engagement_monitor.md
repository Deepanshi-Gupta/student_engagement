# Synopsis

## Real-Time Multimodal Student Engagement & Distress Monitor

---

## Abstract

In most classrooms today, a teacher stands in front of thirty students and has no reliable way of knowing how many of them are genuinely following the lesson. Some look attentive but are mentally elsewhere. Some are confused but will not say so. A few may be carrying emotional distress that is quietly eating into their ability to learn. This project proposes a real-time AI system that addresses exactly this problem — not by replacing the teacher's judgment, but by extending it.

The system uses a standard webcam and microphone to continuously analyse a student's engagement and emotional state during a learning session. Three computer vision streams process every captured frame in parallel: one tracks head pose and attention direction through facial landmarks, one reads facial expressions to detect confusion, frustration, or fatigue, and one monitors eye blink rate and gaze focus. Alongside this, a speech layer transcribes what the student says and identifies the emotional tone of their voice. These five signals are brought together by a custom fusion module that produces a single engagement score and a distress flag in real time. When the system detects sustained disengagement or distress, it generates a concise, natural-language alert for the teacher through the Gemini API — something actionable, not just a number on a screen.

The project is built entirely on pre-trained open-source models. The original contribution lies in the fusion design, the signal weighting logic, and an ablation study that compares the multimodal system against single-modal baselines to quantify how much each modality adds to detection accuracy.

---

## 1. Introduction and Motivation

There is a quiet problem in education that rarely gets discussed directly: most classroom monitoring is passive. Teachers observe students visually when they can, respond to raised hands, and catch disengagement only when it becomes impossible to ignore. By that point, the window for a timely intervention has often already passed.

This problem has grown sharper with the rise of online and hybrid learning. In a physical room, a teacher can walk around, notice slumped shoulders, hear a confused murmur. On a video call, they see thumbnail-sized faces and have almost no peripheral cues. Even in physical classrooms, a class of thirty is genuinely hard to monitor in full — human attention does not scale that way.

Researchers have been working on automated engagement detection for over a decade, and the results show clear progress. Systems built on facial action units, head pose estimation, and more recently deep learning-based emotion classifiers have demonstrated that it is possible to infer cognitive and emotional states from video alone. But single-modal approaches have a consistent weakness: they are too easy to fool, and too narrow to capture the full picture. A student who looks at the board but has zoned out will not be flagged by a head pose model. A student who is silently anxious will not necessarily show it on their face.

The motivation for this project comes from a straightforward observation: the signals that together reveal disengagement — where a person is looking, what their face is doing, how their eyes are moving, what their voice sounds like, what words they are using — are available simultaneously in any real learning session. No existing system brings all of them together, fuses them into one coherent score, and delivers it to the teacher in real time as a plain-language insight. That is the gap this project fills.

The secondary motivation is accessibility. Students with autism, ADHD, or anxiety often cannot self-report distress in the moment. A system that reads non-verbal signals and surfaces them to an educator is, in a real sense, giving those students a voice they would not otherwise have.

---

## 2. Literature Review

Research into automated student engagement detection has taken several distinct directions over the past fifteen years, each building on the previous while leaving certain problems unsolved.

Early work focused heavily on facial action units, drawing from Ekman's foundational taxonomy of human facial expressions. Systems trained on the DISFA and CK+ datasets could classify a narrow set of emotional states with reasonable accuracy, but they struggled with real classroom conditions — variable lighting, partial occlusion, and the fact that sustained learning rarely produces the extreme expressions these models were trained on. Engagement in a classroom looks subtle: a slight furrow of the brow, eyes drifting sideways, a slow blink. These are not the expressions in training datasets.

Head pose estimation became a common proxy for attention. If a student's face is oriented toward the front of the room, the assumption is that they are paying attention. MediaPipe's Face Mesh, which provides 468 facial landmarks in real time on commodity hardware, made this practical at scale. Work by Liao et al. (2021) and others demonstrated that yaw, pitch, and roll angles derived from landmarks correlate meaningfully with self-reported attention. But head pose alone has an obvious limitation: a student can face forward and be entirely absent mentally.

Eye tracking has added nuance to this picture. Blink rate is a known marker of cognitive fatigue — it rises when a person is tired and drops when they are concentrating intensely. Gaze direction is a stronger attention signal than head pose but has historically required specialised hardware. Recent work using standard webcams and landmark-based gaze estimation has narrowed this gap significantly.

Speech-based emotion recognition has developed largely separately from the computer vision stream. Wav2Vec2, pre-trained on large speech corpora and fine-tuned on datasets like RAVDESS and IEMOCAP, can classify emotional tone — frustration, calm, distress, engagement — from short audio segments with accuracy that approaches human-level on clean recordings. Whisper, OpenAI's transcription model, adds another layer: the words a student actually uses carry confusion signals that prosody alone might miss.

What the literature has not produced is a system that runs all these streams simultaneously, fuses them with explicit weighting, and generates natural-language output from the fused signal. Some multi-modal systems exist in research settings — notably work by Kaur et al. (2018) on multi-channel affect detection and Monkaresi et al. (2017) on physiological and visual fusion — but they either require non-standard hardware, operate offline, or do not include a generative output layer. The specific combination proposed here — three parallel CV streams plus speech, fused via a configurable weighted module, with LLM-generated alerts — has not been published in this configuration.

---

## 3. Problem Statement and Objectives

**Problem Statement:**
Real-time, multimodal student engagement monitoring in educational settings remains an unsolved deployment problem. Existing systems are either single-modal and therefore brittle, or multi-modal but impractical — requiring specialised hardware, offline processing, or producing outputs that educators cannot act on directly. There is no lightweight, deployable system that fuses vision, gaze, and speech into one explainable engagement signal and translates it into actionable, natural-language guidance for a teacher, in the moment the lesson is happening.

**Objectives:**

1. To build and integrate three parallel computer vision streams — head pose estimation, facial emotion recognition, and eye blink/gaze tracking — running concurrently on a standard webcam feed without requiring a GPU.

2. To build an asynchronous speech processing layer that combines real-time transcription (Whisper) with speech emotion classification (Wav2Vec2) running in parallel to the vision streams.

3. To design and implement a custom fusion module that combines all five signal outputs into a single engagement score (0–100) and a boolean distress flag, with a configurable weighting schema and a smoothing window to reduce noise.

4. To integrate the Gemini API as a reasoning layer that translates the structured engagement vector into plain-language teacher alerts and student nudges every 30 seconds.

5. To serve the complete pipeline through a FastAPI backend with endpoints for live streaming, alert retrieval, and session summary, containerised with Docker and deployed to GCP Cloud Run.

6. To conduct an ablation study comparing (a) vision-only, (b) speech-only, and (c) full multimodal fusion configurations across simulated sessions, producing a quantitative comparison of distress detection accuracy as the primary research contribution.

---

## 4. Methodology and Planning of Work

### System Architecture

The system is structured in four layers that operate in a continuous pipeline from input to output.

**Layer 1 — Vision (three parallel streams)**
Three streams process every webcam frame simultaneously. Stream A uses MediaPipe Face Mesh to extract 468 facial landmarks and compute yaw, pitch, and roll angles, mapping them to attentive, distracted, or away states. Stream B passes the same frame to DeepFace for emotion classification across seven categories. Stream C reads eye landmark coordinates from the same MediaPipe output to compute blink rate and a gaze direction index. Each stream emits a label and confidence score approximately once per second.

**Layer 2 — Speech (asynchronous, parallel to vision)**
A background thread captures microphone audio via PyAudio in five-second rolling windows. Whisper transcribes each chunk and a keyword matcher scans for confusion markers. A second thread passes the same audio to a Wav2Vec2 model fine-tuned on RAVDESS for speech emotion classification. Both threads emit their outputs asynchronously and push results to a shared queue read by the fusion module.

**Layer 3 — Fusion (original contribution)**
The fusion module reads from all five signal queues, applies a weighted combination, and produces an engagement score and distress flag over a ten-second smoothing window. The default weights — head pose: 0.25, facial emotion: 0.30, eye state: 0.20, speech emotion: 0.15, confusion keywords: 0.10 — are configurable and form the subject of the ablation study. The module is the architectural centrepiece of the project and is documented as a standalone design decision in the thesis.

**Layer 4 — Reasoning and output**
Every 30 seconds, the fusion output is formatted as a structured prompt and sent to the Gemini API, which generates a natural-language alert. Three alert types are defined: teacher alerts (class-level), student nudges (individual), and session summary. FastAPI exposes three endpoints: /stream (live engagement data), /alert (latest Gemini message), and /report (full session PDF). The system is containerised with Docker and deployed to GCP Cloud Run.

---

### Timeline: 24 May – 31 July 2025

| Period | Focus | Deliverable |
|---|---|---|
| Week 1 (24 May – 31 May) | Environment setup. Install MediaPipe, DeepFace, PyAudio, Whisper. Implement head pose stream on webcam feed. | 1 working CV stream with attentive/distracted label. Repo and README initialised. |
| Week 2 (1 Jun – 7 Jun) | Add DeepFace emotion stream and eye blink stream on same frame. Optimise FPS — skip frames for DeepFace if below 15. | 3 parallel CV streams running on live webcam at acceptable frame rate. |
| Week 3 (8 Jun – 14 Jun) | Implement speech layer. Whisper transcription on rolling 5-second mic chunks + keyword detection. Wav2Vec2 speech emotion in second background thread. | Speech layer running async alongside vision. Tested with sample audio of confused/frustrated speech. |
| Week 4 (15 Jun – 21 Jun) | Write fusion.py module. Weighted combination of 5 signals → engagement score and distress flag. Rolling 10-second window. Unit tests on edge cases. | Fusion module complete. Design decision document written. First internal end-to-end test. |
| Week 5 (22 Jun – 28 Jun) | Gemini integration. Format engagement vector as prompt every 30 seconds. Implement 3 alert types. Iterate prompt engineering across 3–4 test sessions. | Full pipeline working end-to-end. Alert output readable and non-alarming. |
| Week 6 (29 Jun – 5 Jul) | FastAPI backend. Three endpoints. Matplotlib live engagement chart. Docker build. GCP Cloud Run deployment. | Deployed system accessible via URL. Postman-tested endpoints. |
| Week 7 (6 Jul – 15 Jul) | Ablation study. Run system under 3 configurations across 5 simulated sessions. Record and compare distress detection accuracy. | Comparison table — the core research result for Chapter 4 of the thesis. |
| Week 8 (16 Jul – 31 Jul) | Thesis writing. Five chapters: Introduction, Related Work, System Architecture, Experiments, Conclusion. Conference paper draft. | Complete 15–20 page thesis draft. Submission-ready paper for IEEE EDUCON or NCVPRIPG. |

---

## 5. Facilities Required for Proposed Work

**Hardware:**
- A laptop or desktop with a minimum of 8 GB RAM and a standard integrated webcam. A discrete GPU (NVIDIA, CUDA-enabled) is helpful for faster DeepFace inference but is not a hard requirement — the system is designed to run on CPU with frame-skipping optimisation.
- A standard microphone or headset for audio capture.
- Internet access for Gemini API calls and GCP Cloud Run deployment.

**Software:**
- Python 3.10 or higher
- MediaPipe (Google, open source) — facial landmark extraction
- DeepFace (pip installable) — emotion classification
- OpenAI Whisper (open source) — speech transcription
- Wav2Vec2 via HuggingFace Transformers — speech emotion recognition
- PyAudio — microphone capture
- FastAPI + Uvicorn — API backend
- Matplotlib — real-time engagement chart
- Docker — containerisation
- Google Cloud Platform (Cloud Run) — deployment (free tier sufficient for development)
- Gemini API — generative alert layer (API key required, free quota available)
- Git and GitHub — version control and thesis reference

**Datasets (for ablation study simulation):**
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) — freely available, used for speech emotion model reference
- FER2013 (Facial Expression Recognition dataset) — via Kaggle, used as reference for emotion model validation

**No specialised lab equipment is required.** The entire project runs on commodity hardware and open-source or free-tier cloud tools.

---

## 6. References

1. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A technique for the measurement of facial movement.* Consulting Psychologists Press.

2. Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., ... & Grundmann, M. (2019). MediaPipe: A framework for building perception pipelines. *arXiv preprint arXiv:1906.08172.*

3. Serengil, S. I., & Ozpinar, A. (2020). LightFace: A hybrid deep face recognition framework. *2020 Innovations in Intelligent Systems and Applications Conference (ASYU).* IEEE.

4. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust speech recognition via large-scale weak supervision. *International Conference on Machine Learning (ICML).*

5. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A framework for self-supervised learning of speech representations. *Advances in Neural Information Processing Systems (NeurIPS), 33.*

6. Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE, 13*(5), e0196391.

7. Liao, J., Liang, Y., & Pan, J. (2021). Deep facial spatiotemporal network for engagement prediction in online learning. *Applied Intelligence, 51*(10), 6609–6621.

8. Kaur, A., Mustafa, A., Mehta, L., & Dhall, A. (2018). Prediction and localisation of student engagement in the wild. *arXiv preprint arXiv:1804.00858.*

9. Monkaresi, H., Bosch, N., Calvo, R. A., & D'Mello, S. K. (2017). Automated detection of engagement using video-based estimation of facial expressions and heart rate. *IEEE Transactions on Affective Computing, 8*(1), 15–28.

10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

11. Team, G. (2023). Gemini: A family of highly capable multimodal models. *arXiv preprint arXiv:2312.11805.*
