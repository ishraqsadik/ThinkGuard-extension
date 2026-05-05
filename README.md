# ThinkGuard — reproduction and extensions

This repository evaluates the public **ThinkGuard** checkpoint (Wen et al., 2025, ACL Findings): an 8B safety guardrail built on **LLaMA Guard 3**–style moderation, trained with critique-augmented supervision so it can emit both a **safe / unsafe** verdict and a structured rationale.

**Upstream model and paper:** [Rakancorle1/ThinkGuard](https://huggingface.co/Rakancorle1/ThinkGuard) · [ThinkGuard (arXiv)](https://arxiv.org/abs/2502.13458) · [luka-group/ThinkGuard](https://github.com/luka-group/ThinkGuard)

The repo adds **`src/tg_eval/`**, **`scripts/`**, and a **Google Colab notebook** that runs all experiments end-to-end. We do not retrain the model.

## What the three experiments are

| Block in the notebook | What it runs |
|----------------------|----------------|
| **Experiment 1** | Single-pass ThinkGuard on **BeaverTails**, **ToxicChat**, **WildGuardMix**, and **OpenAI moderation** (sample caps set in the first code cell). |
| **Experiments 2a & 2b** | **Reflective** inference (three generations per row) on WildGuardMix, then on BeaverTails. |
| **Experiment 3** | **Latency** (FP16/BF16 vs NF4) and a **matched-subset** BeaverTails eval under full precision vs 4-bit weights. |

---

## Running experiments in Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishraqsadik/ThinkGuard-repro-submission/blob/main/notebooks/ThinkGuard_Full_Experiment_Colab.ipynb)

Everything is driven by **`notebooks/ThinkGuard_Full_Experiment_Colab.ipynb`**. Run cells **top to bottom** on a **GPU** runtime. If you use the badge, **File → Save a copy in Drive** so you can edit `REPO_URL` and caps.

### 1. Create a GPU runtime

1. Open the notebook in Colab (e.g. upload it, or open from GitHub with Colab).
2. **Runtime → Change runtime type →** choose a **GPU** (e.g. **T4**, **L4**, or **A100**). CPU is not suitable for the full notebook.
3. **Runtime → Run all** is fine once secrets are set (next step).

### 2. Hugging Face token (required)

Several assets are **gated** on the Hub (notably **WildGuardMix**; BeaverTails may require access depending on your account).

1. In Hugging Face: create a **read** access token and accept any **dataset / model access agreements** for the assets you use.
2. In Colab: **Secrets** (toggled on for the notebook) → add secret name **`HF_TOKEN`** with that token.
3. The first code cell loads `userdata.get("HF_TOKEN")` into `os.environ["HF_TOKEN"]`. If the assert fails, the secret name or visibility is wrong.

Without a valid token, dataset or model downloads will fail with 401/403 errors.

---

## Citation (original ThinkGuard paper)

```bibtex
@Inproceedings{wen2025thinkguard,
  title={ThinkGuard: Deliberative Slow Thinking Leads to Cautious Guardrails},
  author={Xiaofei Wen and Wenxuan Zhou and Wenjie Jacky Mo and Muhao Chen},
  booktitle={ACL - Findings},
  year={2025}
}
```
