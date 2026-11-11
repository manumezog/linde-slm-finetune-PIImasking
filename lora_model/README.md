---
base_model: unsloth/gemma-2-2b-it-bnb-4bit
library_name: peft
pipeline_tag: text-generation
license: mit
language:
- es
- en
tags:
- base_model:adapter:unsloth/gemma-2-2b-it-bnb-4bit
- lora
- qlora
- sft
- transformers
- trl
- unsloth
- pii-masking
- legaltech
- ner
---

# LoRA Adapter: Gemma 2 2B — Spanish Legal PII Masking & Tokenization

This directory contains the fine-tuned **QLoRA adapter weights** for [`unsloth/gemma-2-2b-it-bnb-4bit`](https://huggingface.co/unsloth/gemma-2-2b-it-bnb-4bit), trained to redact Personally Identifiable Information (PII) from Spanish legal and tax queries and output a deterministic lookup mapping.

## Model Summary

- **Base Model:** `unsloth/gemma-2-2b-it-bnb-4bit` (Gemma 2 2B Instruct, 4-bit quantized)
- **Adapter Type:** LoRA (`PEFT 0.21.2`)
- **Trainable Parameters:** `20,766,720` / `2,635,108,608` (`0.79%`)
- **LoRA Config:** `r = 16`, `lora_alpha = 16`, `lora_dropout = 0`, `bias = "none"`
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Training Hardware:** 1x NVIDIA Tesla T4 (16GB VRAM) on Google Colab
- **Training Duration:** 200 steps (~23 epochs) in 5 minutes 21 seconds
- **Final Training Loss:** `0.0295` (down from `2.223`)

For full documentation, architecture diagrams, and the 3-way inference benchmark evaluation, see the [main repository README](../README.md).
