<div align="center">


## BOW: Training Language Models to Reason Over Plausible Next Words
<div>
Ming Shen*, Zhikun Xu*, Jacob Dineen, Xiao Ye, Ben Zhou
</div>
(*Equal Contribution)
<div>
<strong>ARC Lab, Arizona State University</strong>
</div>


[![paper](https://img.shields.io/badge/arXiv-2506.13502-red)](https://arxiv.org/abs/2506.13502)


</div>


## Highlights
### Motivation
Next-word prediction trains a model on the one continuation that happens to be observed, even though most contexts allow several plausible next words. Rewarding a rationale for that single continuation can lock a model into an early, overconfident guess. **BOW** instead trains the model to describe the whole set of plausible next words, and scores that description with a frozen scorer that never sees the original context, so every useful clue has to be written into the description.

### TL;DR
In this work, we propose **Bottlenecked Next-Word Exploration (BOW)**, an RL framework that rewards a model for writing a self-contained description of the plausible next words, by how well a frozen scorer recovers the true next word from that description alone; **BOW-Reg** adds a light breadth regularizer that discourages collapsing onto a single answer.


### Performance
A BOW variant is the strongest trained method on 17 of the 20 model–benchmark pairs in general reasoning, and BOW-Reg reaches the highest SharedRef correctness and the lowest HoWN-Simple collapse on both backbones.

**General reasoning** (SC@10, zero-shot multiple-choice QA). **Bold** marks the best result among trained methods; Vanilla is the original model.

|  | CSQA | PIQA | SQA | TQA | ARC-c | WG | BBH | MMLU | MMLU-p | GPQA |
|---|---|---|---|---|---|---|---|---|---|---|
| ***Qwen2.5-7B-Instruct*** | | | | | | | | | | |
| Vanilla | 80.51 | 87.21 | 64.21 | 64.65 | 92.06 | 64.48 | 77.76 | 75.28 | 58.53 | 31.92 |
| SLM | 75.51 | 84.55 | 62.99 | 53.98 | 88.05 | 59.43 | 54.92 | 69.53 | 48.11 | 29.24 |
| ToW | 81.16 | 88.41 | **67.69** | 57.16 | **93.00** | 66.14 | 69.21 | 69.52 | 55.16 | 32.14 |
| BR | 77.64 | 87.05 | 60.53 | 61.08 | 89.76 | 62.59 | 76.82 | 70.82 | 46.17 | 31.25 |
| IGR | 78.71 | 87.27 | 63.39 | 65.73 | 92.41 | 68.43 | **78.01** | 75.60 | 56.94 | 29.69 |
| BOW-Core | 81.49 | 87.21 | 64.83 | 63.28 | 92.75 | 64.56 | 76.78 | 75.69 | **58.22** | 32.37 |
| BOW-Reg | **81.90** | **88.68** | 63.19 | **66.83** | 92.32 | **70.96** | 77.21 | **76.51** | 57.51 | **33.26** |
| ***LLaMA3.1-8B-Instruct*** | | | | | | | | | | |
| Vanilla | 77.07 | 85.15 | 69.33 | 60.47 | 89.25 | 59.59 | 74.71 | 72.90 | 50.50 | 32.14 |
| SLM | 36.12 | 67.85 | 53.78 | 9.55 | 45.82 | 52.17 | 22.01 | 39.32 | 12.80 | 24.55 |
| ToW | 22.28 | 60.07 | 46.83 | 28.64 | 34.30 | 45.07 | 22.71 | 43.77 | 21.16 | 25.00 |
| BR | 76.66 | 83.73 | 66.46 | 53.12 | 84.47 | 57.14 | 65.89 | 66.33 | 38.56 | 22.32 |
| IGR | 77.81 | 86.62 | 67.89 | 58.63 | 90.02 | 60.06 | 73.98 | 73.27 | 49.78 | 31.03 |
| BOW-Core | **78.71** | **87.11** | **68.92** | 58.75 | **90.36** | **61.96** | 74.10 | **74.21** | **53.13** | **35.04** |
| BOW-Reg | 77.97 | 86.67 | 67.28 | **60.22** | 89.16 | 59.75 | **74.56** | 73.59 | 52.71 | 32.14 |

**Targeted ambiguity** (%): SharedRef correctness, and HoWN-Simple coverage and single-sense collapse.

|  | SharedRef Correctness ↑ | HoWN-Simple Covered ↑ | HoWN-Simple Collapse ↓ |
|---|---|---|---|
| ***Qwen2.5-7B-Instruct*** | | | |
| Vanilla | 36.12 | 48.64 | 48.15 |
| SLM | 36.12 | 47.48 | 41.10 |
| ToW | 66.52 | 48.65 | 37.68 |
| BR | 56.39 | **52.20** | 48.47 |
| IGR | 15.42 | 45.77 | 61.11 |
| BOW-Core | 50.66 | 51.19 | 44.44 |
| BOW-Reg | **78.85** | 50.84 | **31.90** |
| ***LLaMA3.1-8B-Instruct*** | | | |
| Vanilla | 14.10 | 59.20 | 28.22 |
| SLM | 28.40 | 13.20 | 85.37 |
| ToW | 15.86 | 49.75 | 36.20 |
| BR | 13.66 | 56.38 | 25.00 |
| IGR | 15.86 | 55.67 | 28.22 |
| BOW-Core | 21.15 | **65.86** | 8.59 |
| BOW-Reg | **29.07** | 63.91 | **7.36** |

Baselines: SLM and ToW (continual pretraining), BR (binary-reward RL) and IGR (information-gain RL). TQA = TruthfulQA, SQA = StrategyQA, ARC-c = ARC-Challenge, WG = WinoGrande, MMLU-p = MMLU-Pro.

For more details, please refer to our paper!

## Training
Our implementation builds on [verl](https://github.com/volcengine/verl). The training data, about 45K context–next-word pairs from the True Detective stories (Del and Fishel, 2023) filtered with GPT-4.1-mini, is in `data/bow/` (`examples/data_preprocess/bow_corpus.py` rebuilds it).

Install the dependencies, then log in to Hugging Face (LLaMA3.1 is gated) and Weights & Biases:
```bash
pip install -r requirements.txt
huggingface-cli login
wandb login
```

Start the frozen scorer, LLaMA3.1-8B-Instruct served by vLLM, on its own GPU:
```bash
CUDA_VISIBLE_DEVICES=7 vllm serve meta-llama/Llama-3.1-8B-Instruct --dtype auto --api-key bow --port 7000 --gpu-memory-utilization 0.9 --max-logprobs 15206
```
Then set `JUDGE_SERVER_IP` in `verl/utils/reward_score/BoW.py` to the scorer's host (`localhost` if it runs on the training machine), and `_temp_dir` in `verl/trainer/main_ppo.py` to a writable folder.

From the repo root, train BOW-Reg on 4 GPUs:
```bash
# Qwen2.5-7B-Instruct
bash examples/grpo_trainer/bow_run_qwen2.5-7b_gpt4.1nanofiltered_thoughtv2_llama3.1judge_T5_nokl_refJudge.sh
# LLaMA3.1-8B-Instruct
bash examples/grpo_trainer/bow_run_llama3.1-8b_gpt4.1nanofiltered_thoughtv2_llama3.1judge_T5_nokl_refJudge.sh
```
To train BOW-Core instead, append `reward_model.use_judge_reference=False`.

## Citation
If you find it useful for your research and applications, please cite our paper using this BibTeX:
```bibtex
@misc{shen2026bowtraininglanguagemodels,
      title={BOW: Training Language Models to Reason Over Plausible Next Words},
      author={Ming Shen and Zhikun Xu and Jacob Dineen and Xiao Ye and Ben Zhou},
      year={2026},
      eprint={2506.13502},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.13502},
}
```
