# Constraining LLM Behavior via Abstention

This project evaluates whether simple abstention mechanisms (self-evaluation and sampling consistency) help an LLM avoid hallucinations by declining to answer when uncertain.

## Key Findings
- Sampling consistency reduced TruthfulQA risk from 0.18 to 0.10 but at low coverage (0.20).
- Self-evaluation produced only marginal risk reduction at near-full coverage.
- HaluEval verifier achieved ~0.80 accuracy on accepted responses with coverage 0.77–0.89.

## How to Reproduce
1. Create environment and install dependencies:
   - `uv venv`
   - `uv add datasets pandas numpy matplotlib seaborn scikit-learn scipy statsmodels openai tenacity`
2. Run experiments:
   - `.venv/bin/python src/run_experiments.py --use-judge --truthfulqa-sample 50 --halueval-sample 100`
3. Run analysis:
   - `.venv/bin/python src/analyze_results.py --results-dir results/abstention_experiment`

## File Structure
- `src/run_experiments.py`: experiment runner and API calls
- `src/analyze_results.py`: metrics aggregation and plots
- `results/abstention_experiment/`: outputs, metrics, and plots
- `REPORT.md`: full research report

See `REPORT.md` for full methodology and results.
