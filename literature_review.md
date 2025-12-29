# Literature Review

## Research Area Overview
This review focuses on methods and benchmarks that encourage large language models (LLMs) to abstain, express uncertainty, or self-evaluate their answers to reduce hallucinations. The literature spans calibrated self-knowledge, verbalized uncertainty, hallucination detection/verifiers, and benchmark datasets that evaluate truthfulness and hallucination rates.

## Key Papers

### Paper 1: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models
- **Authors**: Potsawee Manakul; Adian Liusie; Mark J. F. Gales
- **Year**: 2023
- **Source**: arXiv
- **Key Contribution**: Sampling-based hallucination detection without access to model logits or external KBs.
- **Methodology**: Generate multiple samples, measure consistency signals to flag likely hallucinations.
- **Datasets Used**: Evaluations on LLM generations (see paper for exact sets).
- **Results**: Demonstrates competitive hallucination detection in black-box settings.
- **Code Available**: Yes (https://github.com/potsawee/selfcheckgpt)
- **Relevance to Our Research**: Provides a verifier-style signal for abstention decisions.

### Paper 2: Language Models (Mostly) Know What They Know
- **Authors**: Saurav Kadavath et al.
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Shows that LMs can self-evaluate correctness and exhibit calibration under proper prompting.
- **Methodology**: Ask the model to answer, then estimate P(True); study calibration and scaling.
- **Datasets Used**: Multiple-choice and true/false tasks (various benchmarks).
- **Results**: Larger models show improved calibration and self-evaluation.
- **Code Available**: Not specified in paper.
- **Relevance to Our Research**: Supports training signals for abstention via self-assessed confidence.

### Paper 3: Teaching Models to Express Their Uncertainty in Words
- **Authors**: Stephanie Lin; Jacob Hilton; Owain Evans
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Trains models to verbalize calibrated uncertainty without access to logits.
- **Methodology**: Supervised learning with verbalized probabilities; evaluates calibration under distribution shift.
- **Datasets Used**: Introduces CalibratedMath suite (see paper).
- **Results**: Verbalized uncertainty is well calibrated and robust to shifts.
- **Code Available**: Not specified in paper.
- **Relevance to Our Research**: Directly enables "I don't know" outputs tied to calibrated confidence.

### Paper 4: TruthfulQA: Measuring How Models Mimic Human Falsehoods
- **Authors**: Stephanie Lin; Jacob Hilton; Owain Evans
- **Year**: 2021
- **Source**: arXiv
- **Key Contribution**: Benchmark to measure truthfulness in LLM answers.
- **Methodology**: 817 handcrafted questions; compare model vs. human answers for truthfulness.
- **Datasets Used**: TruthfulQA dataset.
- **Results**: Larger models often less truthful; best models far below human truthfulness.
- **Code Available**: Yes (https://github.com/sylinrl/TruthfulQA)
- **Relevance to Our Research**: Core benchmark for abstention and truthful response behavior.

### Paper 5: Detecting Hallucinations in Large Language Model Generation: A Token Probability Approach
- **Authors**: Ernesto Quevedo; Jorge Yero; Rachel Koerner; Pablo Rivas; Tomas Cerny
- **Year**: 2024
- **Source**: arXiv
- **Key Contribution**: Lightweight supervised classifiers using token-probability features.
- **Methodology**: Use token/vocabulary probability features from evaluator LMs for hallucination detection.
- **Datasets Used**: Hallucination datasets (see paper).
- **Results**: Promising detection with simple features.
- **Code Available**: Not specified in paper.
- **Relevance to Our Research**: Practical verifier approach for abstain-or-answer gating.

### Paper 6: Constitutional AI: Harmlessness from AI Feedback
- **Authors**: Yuntao Bai et al.
- **Year**: 2022
- **Source**: arXiv
- **Key Contribution**: Trains models to follow rules and refuse harmful queries via self-critique and AI feedback.
- **Methodology**: Supervised self-critique + RL from AI preferences guided by a constitution.
- **Datasets Used**: Model-generated preference data (see paper).
- **Results**: Improved harmlessness without human labels for harmful outputs.
- **Code Available**: Not specified in paper.
- **Relevance to Our Research**: Demonstrates constraint-following and refusal training signals.

### Paper 7: HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models
- **Authors**: Junyi Li; Xiaoxue Cheng; Wayne Xin Zhao; Jian-Yun Nie; Ji-Rong Wen
- **Year**: 2023
- **Source**: arXiv
- **Key Contribution**: Large benchmark of hallucinated vs. non-hallucinated LLM outputs.
- **Methodology**: ChatGPT-based sampling-then-filtering + human annotation.
- **Datasets Used**: HaluEval benchmark.
- **Results**: Reveals topic-dependent hallucination tendencies.
- **Code Available**: Yes (https://github.com/RUCAIBox/HaluEval)
- **Relevance to Our Research**: Provides evaluation data for abstention/verifier systems.

## Common Methodologies
- **Self-evaluation and calibration**: Used in Language Models (Mostly) Know What They Know; Teaching Models to Express Their Uncertainty in Words.
- **Verifier-based detection**: Used in SelfCheckGPT; Token Probability Approach; HaluEval evaluation.
- **Rule-guided refusal**: Used in Constitutional AI.

## Standard Baselines
- **Answer-only LLM baseline**: Evaluate without abstention (TruthfulQA, HaluEval).
- **Prompted self-evaluation**: Ask model to judge its own answer correctness.
- **Sampling-based consistency**: Multiple generations to detect hallucinations.

## Evaluation Metrics
- **Accuracy / Truthfulness rate**: Primary metric for TruthfulQA-style benchmarks.
- **Calibration (ECE/Brier)**: For uncertainty/self-evaluation approaches.
- **AUROC / F1**: For hallucination detection classifiers.
- **Abstention-aware metrics**: Coverage vs. risk tradeoff (selective prediction).

## Datasets in the Literature
- **TruthfulQA**: Used for truthfulness and refusal behavior.
- **HaluEval**: Large-scale hallucination evaluation benchmark.
- **Task-specific QA sets**: Often used to test abstention on unanswerable questions.

## Gaps and Opportunities
- **Unified abstention benchmarks**: Few datasets explicitly reward abstention vs. hallucination.
- **Generalization under distribution shift**: Limited evidence on how abstention policies transfer across domains.
- **Verifier coupling**: Need systematic study of verifier reliability vs. base model.

## Recommendations for Our Experiment
- **Recommended datasets**: TruthfulQA (truthfulness), HaluEval (hallucinations), SQuAD v2 (unanswerable QA).
- **Recommended baselines**: Answer-only LLM; self-evaluation prompting; sampling-based consistency checks.
- **Recommended metrics**: Truthfulness rate, AUROC/F1 for hallucination detection, calibration error, coverage-risk curves.
- **Methodological considerations**: Separate abstention thresholding from base model quality; evaluate calibration under domain shift.
