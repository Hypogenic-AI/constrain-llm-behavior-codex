# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project, including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 7

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| SelfCheckGPT | Manakul et al. | 2023 | papers/2303.08896_selfcheckgpt.pdf | Sampling-based hallucination detection |
| Language Models (Mostly) Know What They Know | Kadavath et al. | 2022 | papers/2207.05221_lm_know_what_they_know.pdf | Self-evaluation + calibration |
| Teaching Models to Express Their Uncertainty in Words | Lin et al. | 2022 | papers/2205.14334_uncertainty_in_words.pdf | Verbalized uncertainty |
| TruthfulQA | Lin et al. | 2021 | papers/2109.07958_truthfulqa.pdf | Truthfulness benchmark |
| Token Probability Approach | Quevedo et al. | 2024 | papers/2405.19648_token_probability_hallucination.pdf | Simple hallucination detector |
| Constitutional AI | Bai et al. | 2022 | papers/2212.08073_constitutional_ai.pdf | Rule-guided refusal training |
| HaluEval | Li et al. | 2023 | papers/2305.11747_halueval.pdf | Hallucination evaluation benchmark |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA (generation) | HuggingFace `truthful_qa` | 817 | Truthfulness | datasets/truthful_qa_generation/ | Validation-only split |
| HaluEval (general) | HuggingFace `pminervini/HaluEval` | 4507 | Hallucination detection | datasets/halu_eval_general/ | General config |
| SQuAD v2 | HuggingFace `squad_v2` | 142,192 | Unanswerable QA | datasets/squad_v2/ | Train/validation splits |

See datasets/README.md for detailed descriptions.

## Code Repositories
Total repositories cloned: 3

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| SelfCheckGPT | https://github.com/potsawee/selfcheckgpt | Hallucination detection | code/selfcheckgpt/ | Sampling-based verifier |
| TruthfulQA | https://github.com/sylinrl/TruthfulQA | Benchmark + eval | code/truthfulqa/ | Includes dataset CSV |
| HaluEval | https://github.com/RUCAIBox/HaluEval | Benchmark + eval | code/halueval/ | Generation/evaluation scripts |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Queried arXiv by title keywords (self-checking, uncertainty, truthful QA, hallucination detection).
- Used GitHub search API to identify benchmark repositories.
- Used HuggingFace dataset search for benchmark datasets.

### Selection Criteria
- Direct relevance to abstention/uncertainty/hallucination control.
- Availability of public code and data.
- Mix of foundational methods and recent benchmarks.

### Challenges Encountered
- HaluEval has multiple dataset variants; selected the `general` config from `pminervini/HaluEval`.

### Gaps and Workarounds
- Some papers do not provide official code; included related benchmark repos and dataset cards.

## Recommendations for Experiment Design

1. **Primary dataset(s)**: TruthfulQA + HaluEval + SQuAD v2 for truthfulness, hallucination detection, and abstention evaluation.
2. **Baseline methods**: Answer-only LLM; self-evaluation prompt; SelfCheckGPT-style sampling.
3. **Evaluation metrics**: Truthfulness rate, AUROC/F1 for hallucination detection, calibration error, coverage-risk curves.
4. **Code to adapt/reuse**: `code/selfcheckgpt/`, `code/truthfulqa/`, `code/halueval/`.

## Research Process Notes
- Executed experiments with `gpt-4.1` on TruthfulQA (50 samples) and HaluEval (100 samples).
- Generated results in `results/abstention_experiment/` with metrics, analysis JSON, and risk–coverage plots.
- Documented findings in `REPORT.md` and reproducibility steps in `README.md`.
