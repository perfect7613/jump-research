---
title: JUMP — Visual Thought Experiments
emoji: ⚗️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
license: other
---

# Compare two small simulated worlds

Ask a concrete question about a small world and one changed rule. JUMP shows the
declarative plan for confirmation, records a prediction, then renders baseline
and counterfactual states from a restricted deterministic simulator.

The frames are deterministic simulation states, not learned-latent
reconstructions. Unsupported requests fail closed. The Space does not accept
code, URLs, files, or real-world actions. The earlier v1 numeric route remains an
explicit fallback while visual v2 is the primary experience.
