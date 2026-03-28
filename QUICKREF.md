# Quick Reference

This file contains essential commands and resources for quick lookup.

## Common Commands

### Setup
```bash
# One-time setup
bash setup.sh

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Development Workflow
```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
make test-cov  # With coverage report

# All checks
make all

# Clean cache
make clean
```

### Per-Week Workflow
```bash
# Navigate to week directory
cd 01_rnn_from_scratch

# Run week's tests
pytest test_*.py -v

# Open notebook
jupyter lab experiment.ipynb
```

## Project Structure at a Glance

```
nlpai/
├── 01-10_topic/      # Implementation weeks
│   ├── module.py     # Your code
│   ├── test_*.py     # Unit tests
│   ├── notes.md      # Paper notes
│   ├── experiment.ipynb
│   └── README.md     # Week goals
├── notebooks/        # Additional analyses
├── data/            # (git-ignored)
├── models/          # (git-ignored)
├── README.md        # Main roadmap
├── CONTRIBUTING.md  # Code standards
├── PROGRESS.md      # What you've done
└── requirements.txt # Dependencies
```

## Key Papers by Week

| Week | Topic | Must-Read Paper |
|------|-------|-----------------|
| 1 | RNN | Hochreiter & Schmidhuber (1997) - LSTM |
| 2 | LSTM/GRU | Cho et al. (2014) - GRU |
| 3 | Attention | Bahdanau et al. (2015) |
| 4 | Transformer | Vaswani et al. (2017) |
| 5 | GPT | Radford et al. (2018) |
| 6 | Scaling | Kaplan et al. (2020) |
| 7 | Alignment | Ouyang et al. (2022) - InstructGPT |
| 8 | Genomics | Avsec et al. (2021) - Enformer |
| 9 | Single-Cell | Lotfollahi et al. (2022) - scGPT |
| 10 | Proteins | Jumper et al. (2021) - AlphaFold2 |

## Testing Checklist Before Pushing

```bash
# Run these in order
pytest -v                    # Tests pass?
make format                  # Code formatted?
flake8 . --max-line-length=100  # No style issues?
mypy . --ignore-missing-imports # Type hints OK?
```

## Documentation Checklist for Each Week

- [ ] README.md with clear goals and equations
- [ ] unit tests with gradient checks passing
- [ ] notes.md with paper insights
- [ ] experiment.ipynb with toy data results
- [ ] All code formatted with `black`
- [ ] All tests passing

## Resources by Category

### Fundamentals
- **micrograd** — Understand backprop (100 lines)
- **Deep Learning Book** — Theory and notation

### Implementation References
- **nanoGPT** — Clean GPT training codebase
- **The Annotated Transformer** — Line-by-line walkthrough
- **HuggingFace docs** — Modern best practices

### AI4Bio Curation
- **awesome-deepbio** — Computational biology papers
- **awesome-computational-biology** — Genomics resources
- **Broad AwesomeGenomics** — Modern genomics tools

### Reproducibility
- **NeurIPS Code Submission Policy** — Community standards
- **NeurIPS "Releasing Research Code"** — Practical checklist

## Common Mistakes to Avoid

❌ Don't: Only read papers, never implement  
✅ Do: Implement + toy experiments every week

❌ Don't: Skip gradient checks  
✅ Do: Always verify backprop numerically

❌ Don't: Copy code without understanding  
✅ Do: Re-derive equations, then implement

❌ Don't: Train on full datasets first  
✅ Do: Start with toy data, verify loss curves make sense

❌ Don't: Ignore randomness/seeds  
✅ Do: Set seeds, report error bars

## Troubleshooting

### Tests failing?
```bash
pytest <file.py> -v --tb=short  # See detailed error
```

### Gradients exploding/vanishing?
- Check gradient norm after each backward pass
- Verify numerical gradient matches analytic
- May indicate normalization is needed

### Loss not decreasing?
- Start with toy dataset first
- Check learning rate (try 1e-3, 1e-4)
- Verify targets are one-hot or proper labels

### Code style issues?
```bash
black . --line-length=100     # Auto-fix
isort .                       # Fix imports
```

## When You Get Stuck

1. **Understanding a math concept?**
   - Re-read the paper's derivation
   - Work through derivatives by hand
   - Reference math books: Deep Learning, ESL, PML:AI

2. **Implementation bug?**
   - Write a simple test case first
   - Print intermediate shapes/values
   - Use gradient check to narrow down issue

3. **Hyperparameter tuning?**
   - Start with published papers' values
   - Only change one at a time
   - Plot validation curves

4. **Not sure about next steps?**
   - Check README for the week
   - Read WEEK_TEMPLATE.md for structure
   - Look at a reference implementation

## GitHub Workflow

```bash
# Create weekly branch
git checkout -b week-1-rnn

# Regular commits
git add .
git commit -m "Implement RNN forward pass"
git push origin week-1-rnn

# When ready, create PR on GitHub
# (base: main, compare: week-1-rnn)
```

---

**Keep this file handy! Reference it weekly.**

Last updated: March 2026
