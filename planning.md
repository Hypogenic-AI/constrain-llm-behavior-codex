# Research Plan: Constraining LLM Behavior via Abstention

## Research Question
Can LLMs be prompted or augmented with a verifier signal to abstain (“I don’t know / need more data”) instead of hallucinating, and does this reduce risk while preserving useful coverage?

## Background and Motivation
Hallucinations degrade trust and safety in LLM outputs. Prior work shows LLMs can self-evaluate, verbalize uncertainty, or be paired with black-box verifiers. However, practical guidance on how to translate uncertainty signals into abstention policies and quantify the risk–coverage tradeoff remains limited. This study tests whether simple abstention mechanisms (self-evaluation and sampling-based consistency) improve truthful behavior on standard benchmarks.

## Hypothesis Decomposition
1. **H1 (Self-evaluation)**: Prompting an LLM to estimate its own correctness provides a usable abstention signal that improves truthfulness at a fixed coverage level.
2. **H2 (Consistency verifier)**: Sampling-based consistency (SelfCheckGPT-style) provides a stronger abstention signal than self-evaluation alone on hallucination-heavy benchmarks.
3. **H3 (Selective prediction)**: Applying an abstention threshold improves overall risk (error rate) at the cost of reduced coverage, yielding a favorable risk–coverage curve compared to answer-only baselines.

## Proposed Methodology

### Approach
Use a real LLM API to answer benchmark questions, then apply two abstention mechanisms:
- **Self-evaluation prompt**: Ask the model to estimate whether its answer is correct / whether it should abstain.
- **Sampling consistency**: Generate multiple answers and compute agreement; abstain when consistency is low.

Evaluate on TruthfulQA (truthfulness) and a subset of SQuAD v2 unanswerable questions (abstention). HaluEval will be used for hallucination detection if time permits.

### Experimental Steps
1. **Data loading and sampling**: Load TruthfulQA and SQuAD v2 from disk; take manageable subsets (e.g., 200–300 examples each) to control cost.
2. **Baseline generation**: For each question, query the model with a direct answer-only prompt.
3. **Self-evaluation**: Query a second prompt asking if the answer is correct and whether to abstain.
4. **Consistency verifier**: Generate N (e.g., 5) answers; compute agreement rate; abstain when agreement below threshold.
5. **Thresholding**: Sweep abstention thresholds to generate coverage–risk curves.
6. **Evaluation**: Compute truthfulness/accuracy on answered items; compute coverage and risk; compare baselines.

### Baselines
- **Answer-only**: No abstention, direct response.
- **Self-evaluation**: Answer + self-assessed confidence threshold.
- **Sampling consistency**: Multi-sample consistency threshold.

### Evaluation Metrics
- **Accuracy / truthfulness**: Correctness on TruthfulQA and SQuAD v2 answerable items.
- **Abstention rate / coverage**: Fraction of questions answered.
- **Risk (error rate)**: 1 − accuracy on answered items.
- **AUROC / F1**: For classifying hallucinated vs. non-hallucinated where labels exist.
- **Selective prediction curves**: Risk–coverage curves, AUC.

### Statistical Analysis Plan
- Use bootstrap confidence intervals for accuracy and risk at fixed coverage.
- Compare methods using paired bootstrap or McNemar’s test on answered items.
- Significance level α = 0.05 with Bonferroni correction if multiple comparisons.

## Expected Outcomes
Support for the hypothesis if abstention mechanisms reduce risk (error rate) at comparable coverage relative to answer-only baselines. Self-evaluation and consistency are expected to differ in strength across datasets.

## Timeline and Milestones
- **Phase 1 (Planning)**: 1–2 hours
- **Phase 2 (Setup & data checks)**: 1 hour
- **Phase 3 (Implementation)**: 2–3 hours
- **Phase 4 (Experiments)**: 2–3 hours
- **Phase 5 (Analysis)**: 1–2 hours
- **Phase 6 (Documentation)**: 1 hour

## Potential Challenges
- API cost/time: mitigate via subsampling and caching responses.
- Ambiguous labels in TruthfulQA: focus on “best answer” and human reference where available.
- Abstention evaluation ambiguity: handle unanswerables separately.
- Model variability: fix temperature and run repeated samples for consistency.

## Success Criteria
- Demonstrated risk–coverage improvement for at least one abstention method.
- Reproducible scripts with logged parameters, prompts, and outputs.
- Reported statistical comparisons and confidence intervals.
