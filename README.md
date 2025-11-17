# Efficient Vision–Language Alignment using ALBEF and Token-Level Consistency

Improving alignment efficiency in ALBEF using **Per-Image Token Consistency (EPIC)** and saliency-guided techniques. 
This repository combines ALBEF (Align Before Fuse) and EPIC (Per-Image Token Consistency) to develop an efficient fine-grained alignment framework for vision-language pretraining.
Our approach introduces E-PITC, a sparse per-token alignment mechanism using saliency-guided pruning and multi-positive contrastive learning to achieve stronger grounding with reduced computation.

---

## Project Overview

**Goal:**  
Enhance fine-grained alignment between text tokens and image regions in ALBEF by introducing a new pretraining objective: **token-level consistency** using **EPIC (Per-Image Token Consistency)**.

**Key Idea:**  
Instead of only using cross-modal MLM (CMLM), EPIC generates linguistically plausible inconsistent tokens for salient text positions and trains the model to predict, **per token**, whether it is consistent with the image.  
- E-PITC (Efficient Per-Image Token Consistency): Fine-grained alignment with top-k token matching.
- Saliency-Guided Token Pruning: Focus only on important visual and textual tokens.
- Multi-Positive Contrastive Learning: Handle paraphrases and semantically similar captions.
- Momentum Distillation: Stable and noise-resistant pretraining using EMA teacher.

**Benefits:**  
- Provides **more supervision** beyond masked tokens.  
- Reduces **modality bias** (language alone can’t easily detect inconsistencies).  
- Improves **token-to-region alignment**.

---

## EPIC Components

### 1. Saliency-Based Masking
- Uses the **momentum teacher VLM** in ALBEF to compute attention-based saliency: **image → text**.
- Selects **top-m salient tokens** (`mask_ratio = 0.35`) for corruption.

### 2. Inconsistent Token Generation
- Mask selected tokens in text.  
- Feed masked text to **auxiliary LM (`aux_lm`)** to sample replacements.  
- Tokens where replacement ≠ original are **inconsistent tokens (T)**.  
- LM also trained with **MLM loss (`LGEN`)** on masked positions.

### 3. Image-Token Consistency (ITC)
- Add `itc_head = nn.Linear(hidden_dim, 1)` on student VLM.  
- Predict token consistency:

```python
Di_ITC = torch.sigmoid(beta * h_i_VL)
```

- Compute binary cross-entropy using labels:
0 → replaced/inconsistent token\
1 → consistent token\
Loss weight: lambda_itc = 8

- Updated Tota Loss
L_total = L_MLM + L_ITA + L_ITM + λ_itc * L_ITC + L_aux_LM  

---

## Integration into ALBEF

**Components:**

- Student VLM: Main ALBEF (text + image + fusion)

- Teacher VLM: Momentum model (EMA) → used for saliency

- Aux LM: BERT-like LM to generate replacements

**Highlights:**

- Teacher forward wrapped in torch.no_grad()

- Aux LM is trainable; separate optimizer param group

- EPIC losses added to ALBEF original losses:
total_loss = (loss_mlm + loss_ita + loss_itm + lambda_itc * loss_itc + aux_outputs.loss  # LGEN)

**This repo integrates EPIC into ALBEF with:**

- Modified model_pretrain.py

Added ITC head,
Added auxiliary LM,
Added token saliency computation

- Updated pretrain.py

Included EPIC forward pass, 
Added optimizer groups,
Integrated all losses

Efficient variants: (currently working)

Top-k pruning,
Sparse token–patch alignment

All changes preserve ALBEF’s original EMA-based momentum distillation pipeline.

---

## Implementation Steps

- Compute saliency (teacher VLM).
- Prune low-saliency tokens (efficiency enhancement)  - top-K pruning (currently working).
- Mask & generate inconsistent tokens (aux LM).
- Forward student VLM & compute ITC loss, MLM loss, ITA contrastive loss, ITM matching loss, Auxiliary LM’s MLM loss.
- Combine with original ALBEF losses.
- Update momentum teacher via EMA.

---

## Current Progress

- Modified **model_pretrain.py**: Added itc_head and aux_lm; implemented saliency extraction & EPIC forward logic.
- Updated **pretrain.py** Integrated EPIC loss into training loop; added param groups for aux_lm & itc_head.
- Sanity check (epic_test.py): Initialized ALBEF + EPIC components; forward pass working(Need more resource access).
- Integrated token-level consistency into ALBEF
- Updated training loop and optimizer

---

## Next Steps:
- Full pretraining run on dataset(need more resources).

- Ablation experiments:

ALBEF baseline

ITC with random corrupted tokens

Full EPIC with customized changes for efficiency improvement.

ALBEF + Efficient EPIC (our improvements).

---

## Datasets Used:

- Flickr30k (full)

- Reduced subset (for quick testing)

- Custom caption variations (for multi-positive contrastive learning)

- VisualGenome (for EPIC pretraining)

---

## References

- ALBEF
- EPIC
- CLIP, ViLT, METER (background literature)

