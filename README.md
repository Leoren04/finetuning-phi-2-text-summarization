# 🧠 Fine-Tuning Phi-2 for Text Summarization (XSum) via QLoRA

> **UAS Deep Learning — Group Task | Fine-Tuning HuggingFace Models**

---

## 📋 Overview

This repository implements **QLoRA fine-tuning of Microsoft Phi-2** (2.7B decoder-only LLM) for **abstractive text summarization** using the **XSum dataset**. The project demonstrates efficient fine-tuning of a large language model using **4-bit quantization + LoRA adapters**.

### Task Summary

| Item | Description |
|------|-------------|
| **Task** | Abstractive Text Summarization |
| **Architecture** | Decoder-only LLM (Causal LM) |
| **Model** | `microsoft/phi-2` (2.7B parameters) |
| **Dataset** | XSum (BBC Extreme Summarization) |
| **Fine-tuning** | QLoRA (4-bit + LoRA adapters) |
| **Framework** | HuggingFace Transformers + PEFT + BitsAndBytes |
| **Metrics** | ROUGE-1, ROUGE-2, ROUGE-L |

---

## 🏗️ Repository Structure

```
finetuning-phi-2-text-summarization/
├── README.md
├── requirements.txt
├── notebooks/
│   └── finetuning_phi2_xsum.ipynb        # Main notebook
└── reports/
    ├── xsum_eda.png
    └── rouge_scores.png
```

---

## 🚀 How to Run

> ⚠️ **GPU Required**: Phi-2 (2.7B) membutuhkan minimal 8GB VRAM. Gunakan Colab T4/A100.

### Google Colab (Recommended)
1. Upload notebook ke Colab
2. Set Runtime → **GPU** (T4 atau A100)
3. Run all cells

```bash
pip install -r requirements.txt
jupyter notebook notebooks/finetuning_phi2_xsum.ipynb
```

---

## 🔧 QLoRA Approach

**QLoRA = Quantized LoRA**: Fine-tuning LLM besar secara efisien

```
Phi-2 (2.7B params, 4-bit quantized)
    ↓
LoRA Adapters added to attention layers (r=16, α=32)
    ↓
Only ~1% of parameters are trainable
    ↓
Massive VRAM savings while preserving model quality
```

**Instruction Format:**
```
Summarize the following article in one sentence:

{article text}

Summary: {target summary}
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **ROUGE-1** | See notebook output |
| **ROUGE-2** | See notebook output |
| **ROUGE-L** | See notebook output |
| **Training Epochs** | 2 |
| **LoRA Rank (r)** | 16 |
| **Trainable Params** | ~1% of 2.7B |

---

## 🗂️ Dataset

- **XSum**: BBC news articles with single-sentence abstractive summaries
- **Train**: 204,045 | **Val**: 11,332 | **Test**: 11,334
- **HuggingFace**: [`EdinburghNLP/xsum`](https://huggingface.co/datasets/EdinburghNLP/xsum)

---

## 👤 Identifikasi

| Item | Info |
|------|------|
| **Nama** | Rakha Primindra Danuatmaja |
| **NIM** | 1103223001 |
| **Kelas** | TK-46-GAB (Deep Learning) |

---

## 📚 References

- Dettmers, T., et al. (2023). [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- Microsoft Research. (2023). [Phi-2: The surprising power of small language models](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)
- Narayan, S., et al. (2018). [Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization (XSum)](https://arxiv.org/abs/1808.08745)
