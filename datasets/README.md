# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: TruthfulQA (generation)

### Overview
- **Source**: HuggingFace `truthful_qa` (config: `generation`)
- **Size**: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Truthfulness evaluation / abstention behavior
- **Splits**: validation (817)
- **License**: See dataset card on HuggingFace

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("truthful_qa", "generation")
dataset.save_to_disk("datasets/truthful_qa_generation")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/truthful_qa_generation")
```

### Sample Data
See `datasets/truthful_qa_generation/samples/samples.json`.

### Notes
- Commonly used for measuring truthfulness and refusal behavior.

## Dataset 2: HaluEval (general)

### Overview
- **Source**: HuggingFace `pminervini/HaluEval` (config: `general`)
- **Size**: 4,507 examples
- **Format**: HuggingFace Dataset
- **Task**: Hallucination evaluation
- **Splits**: data (4507)
- **License**: See dataset card on HuggingFace

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("pminervini/HaluEval", "general")
dataset.save_to_disk("datasets/halu_eval_general")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/halu_eval_general")
```

### Sample Data
See `datasets/halu_eval_general/samples/samples.json`.

### Notes
- Includes hallucinated vs. non-hallucinated generations for evaluation.

## Dataset 3: SQuAD v2

### Overview
- **Source**: HuggingFace `squad_v2`
- **Size**: 130,319 train / 11,873 validation
- **Format**: HuggingFace Dataset
- **Task**: Question answering with unanswerable questions
- **Splits**: train (130319), validation (11873)
- **License**: See dataset card on HuggingFace

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("squad_v2")
dataset.save_to_disk("datasets/squad_v2")
```

### Loading the Dataset
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/squad_v2")
```

### Sample Data
See `datasets/squad_v2/samples/samples.json`.

### Notes
- Includes unanswerable questions for abstention evaluation.
