# ReaGEN [CVPR 2026]

**Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning**

[Paper]() | [Project Page]()

## Overview

ReaGEN is a framework for adaptive chain-of-thought (CoT) generation in multimodal reasoning. Instead of exhaustively applying all reasoning stages, ReaGEN learns to predict the optimal sequence of structured reasoning stages for each input, significantly reducing inference cost while maintaining or improving accuracy.

The key idea is to decompose multimodal reasoning into **14 modular stages** with structured JSON outputs, then train a lightweight **GEN model** to predict which stages to apply and in what order -- avoiding unnecessary computation at inference time.

### Key Features

- **Modular Reasoning Stages**: 14 composable stages (e.g., `VISUAL.OBSERVATION`, `QUANTITATIVE.REASONING`, `LOGICAL.FILTERING`) each producing validated JSON outputs
- **Adaptive Stage Selection**: A trained GEN model predicts the optimal stage sequence per input
- **Teacher-Student Framework**: Knowledge distillation from large teacher models (GPT-4o, Qwen3-VL-30B) to compact student models (Qwen3-VL-4B)
- **Tree-Based Search Space**: Explores and caches multi-stage reasoning chains for training data generation
- **Attention-Based Refinement**: Uses attention signals to guide stage selection and chain refinement

## Repository Structure

```
ReaGEN/
├── run.py                    # Main execution script
├── train_gen.py              # GEN model training
├── ablation.py               # Ablation studies
├── config.yaml               # Configuration file
├── model/
│   ├── model_loader_gen.py   # GEN model architecture
│   └── model_loader_evo.py   # Evolution model loader
├── dataset/                  # Dataset loaders (MathVision, MMMU, MMStar, etc.)
├── prompts/
│   ├── teacher.py            # Teacher model refinement
│   ├── feedback.py           # Feedback generation
│   └── stage_n/              # Stage definitions and prompt engineering
├── utils/                    # Evaluation metrics, visualization, helpers
└── VReST/                    # ACL 2025 sub-project (tree search + self-reward)
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.x or 12.x
- GPU with 24GB+ VRAM (80GB+ recommended for teacher models)

### Setup

```bash
git clone https://github.com/AISmartPerception/ReaGEN.git
cd ReaGEN
pip install torch transformers numpy scipy pillow tqdm openai python-dotenv huggingface-hub bitsandbytes
```

### Configuration

1. Update `config.yaml` with your local model and dataset paths:

```yaml
model:
  model_id_student: "Qwen/Qwen3-VL-4B-Instruct"
  local_model_dir_student: "/path/to/Qwen3-VL-4B-Instruct"
  model_id_teacher: "gpt-4o"

dataset:
  data_id: "MathLLMs/MathVision"
  local_data_dir: "/path/to/mathvision"
```

2. Create an environment file with your API keys:

```bash
echo "OPENAI_API_KEY=your_key_here" > utils/open_key.env
```

## Usage

### Data Generation (Evolution)

Generate reasoning chains using the teacher-student framework:

```bash
python run.py
```

This runs the iterative evolution process: the student model generates stage-by-stage reasoning chains, the teacher refines them, and the search space is populated with scored chains.

### Training the GEN Model

Train the lightweight GEN model to predict optimal stage sequences:

```bash
python train_gen.py
```

Training uses the evolved chain data. Key hyperparameters (configurable in `config.yaml`):
- Batch size: 64
- Epochs: 300
- Learning rate: 1e-4
- Optimizer: AdamW

### Ablation Studies

```bash
python ablation.py
```

### Supported Models

**Student Models:**
- Qwen/Qwen3-VL-4B-Instruct (default)
- Qwen/Qwen3-VL-8B-Instruct
- OpenGVLab/InternVL3_5-4B
- Qwen/Qwen2.5-VL-7B-Instruct

**Teacher Models:**
- GPT-4o (via OpenAI API)
- Qwen/Qwen3-VL-30B-A3B-Thinking
- Qwen/Qwen3-VL-30B-A3B-Instruct
- Qwen/Qwen2.5-VL-32B-Instruct

### Supported Datasets

| Dataset | Task |
|---------|------|
| [MathVision](https://huggingface.co/datasets/MathLLMs/MathVision) | Mathematical reasoning with vision |
| [MathVerse](https://huggingface.co/datasets/AI4Math/MathVerse) | Mathematical problem variants |
| [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | Massive multimodal understanding |
| [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro) | Advanced MMMU (4/10 options) |
| [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | Comprehensive multimodal benchmark |
| [V*Bench](https://huggingface.co/datasets/craigwu/vstar_bench) | Visual reasoning |
| [TextVQA](https://huggingface.co/datasets/textvqa) | Text-centric VQA |
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | Cross-lingual visual understanding |

## Reasoning Stages

ReaGEN decomposes multimodal reasoning into 14 structured stages:

| Stage | Description |
|-------|-------------|
| `TASK.INTERPRETATION` | Classify the reasoning type required |
| `VISUAL.OBSERVATION` | Extract visual elements from the image |
| `TEXTUAL.UNDERSTANDING` | Parse textual concepts in the question |
| `CONTEXTUAL.LINKING` | Link textual references to visual regions |
| `FACT.EXTRACTION` | Extract measurable facts and data |
| `VARIABLE.DEFINITION` | Define mathematical variables |
| `RELATIONAL.REASONING` | Infer relationships between entities |
| `QUANTITATIVE.REASONING` | Perform numerical computations |
| `LOGICAL.FILTERING` | Prune invalid answer options |
| `HYPOTHESIS.GENERATION` | Generate candidate answers |
| `CROSSMODAL.ALIGNMENT` | Ensure cross-modal consistency |
| `SELFCONSISTENCY.CHECK` | Verify internal consistency |
| `COMPARATIVE.EVALUATION` | Compare candidate hypotheses |
| `EXPLANATION.GENERATION` | Generate justifications |

## VReST Baseline (ACL 2025)

We include an adapted version of **VReST** (Visual Reasoning with Self-reward Tree search) as a baseline. The original VReST code has been updated to incorporate our benchmark suite for fair comparison. See [`VReST/README.md`](VReST/README.md) for usage instructions.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{reagen2026,
  title={Adaptive Generation of Structured Chains-of-Thought for Efficient Multimodal Reasoning},
  author={},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

If you use the VReST component:

```bibtex
@article{zhang2025vrest,
  title={VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism},
  author={Zhang, Congzhi and Peng, Jiawei and Wang, Zhenglin and Lai, Yilong and Sun, Haowen and Chang, Heng and Ma, Fei and Yu, Weijiang},
  journal={arXiv preprint arXiv:2506.08691},
  year={2025}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) 2026 AISmartPerception
