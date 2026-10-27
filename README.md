# TARQA

TARQA: A Unified Framework for Task-Aware Relational Question Answering. Task Spec generation, SQL-based Reasoning, and question-conditioned relational Classification/Regression. The same framework supports pretrained inference, fresh head training, source
pretraining, and per-task target adaptation.

## Environment

Python 3.10+, Linux, and CUDA are required for the recorded 4-bit LLM setup.
The recorded environment used PyTorch 2.5.1+cu121 and NVIDIA A30 GPUs.
Choose the PyTorch CUDA build for your system, then install the core dependencies:

```bash
python -m pip install -r requirements.txt
python run.py --help
```

