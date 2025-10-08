# Efficient Vision–Language Alignment using ALBEF and Token-Level Consistency

This repository combines **ALBEF (Align Before Fuse)** and **EPIC (Per-Image Token Consistency)** to develop an **efficient fine-grained alignment framework** for vision-language pretraining.  
Our approach introduces **E-PITC**, a sparse per-token alignment mechanism using saliency-guided pruning and multi-positive contrastive learning to achieve stronger grounding with reduced computation.

---

## Project Overview

- **Goal:** Improve alignment efficiency in ALBEF by integrating token-level consistency using EPIC implementation.  
- **Key Ideas:**
  - **E-PITC (Efficient Per-Image Token Consistency):** Fine-grained alignment with top-k token matching.
  - **Saliency-Guided Token Pruning:** Focus only on important visual and textual tokens.
  - **Multi-Positive Contrastive Learning:** Handle paraphrases and semantically similar captions.
  - **Momentum Distillation:** Stable and noise-resistant pretraining using EMA teacher.


