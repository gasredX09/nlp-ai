# Contributing Guidelines

Thank you for contributing to this research project! To maintain high standards for reproducibility and code quality, please follow these guidelines.

## Code Standards

### Python Style & Formatting
- **Format:** Use `black` with 100-character line length
  ```bash
  black --line-length=100 .
  ```
- **Imports:** Organize with `isort`
  ```bash
  isort . --profile black
  ```
- **Linting:** Check with `flake8`
  ```bash
  flake8 . --max-line-length=100
  ```
- **Type hints:** Add where helpful, validate with `mypy`
  ```bash
  mypy . --ignore-missing-imports
  ```

### Code Comments & Documentation
- Explain *why*, not just *what*
- For mathematical operations, include the equation in comments (LaTeX-style is fine)
- Include tensor shape annotations in docstrings
- Example:
  ```python
  def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
      """
      Forward pass of RNN cell.
      
      Args:
          x: Input tensor (B, D) - batch size × input dimension
          h: Hidden state tensor (B, H) - batch size × hidden dimension
      
      Returns:
          h_new: Updated hidden state (B, H)
      
      Equations:
          h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)
      """
  ```

## Testing Standards

### Unit Tests
- Write tests alongside implementations
- File naming: `test_<module>.py`
- Test coverage minimum: **80%** for core modules
- Run locally before pushing:
  ```bash
  pytest -v --cov=. --cov-report=term-missing
  ```

### Gradient Checks
- For any custom backprop or loss, include numerical gradient verification
- Compare analytic gradients to finite differences
- Example test pattern:
  ```python
  def test_gradient_check():
      """Verify backprop correctness via finite differences."""
      # Implementation with numerical gradient computation
  ```

### Reproducibility
- Set random seeds in tests
- Use fixed toy datasets (small, included in repo)
- Document all hyperparameters used

## Weekly Submission Template

Each week's directory should include:

```
XX_topic/
├── README.md              # Goals, checkpoints, key concepts
├── module.py              # Core implementation(s)
├── test_module.py         # Unit tests + gradient checks
├── notes.md               # Derivations, insights, paper notes
├── experiment.ipynb       # Toy dataset training & visualization
└── results.txt            # Training curves, metrics, observations
```

### README for Each Week
Must contain:
- [ ] **Learning goals** (bulleted)
- [ ] **Key equations** (with proper notation)
- [ ] **Expected outputs** (loss curves, test results)
- [ ] **How to run** (clear command-line instructions)
- [ ] **Paper references** (with links if available)

### Notes.md
- 1–2 pages of structured notes
- Include diagrams/ASCII art where helpful
- Answer: What are the key insights? Where do gradients come from? Why does this matter?

### experiment.ipynb
- Load a toy dataset
- Train the model
- Plot training curves, attention patterns, or learned representations
- Comment on what you observe

## Research Standards (NeurIPS-Aligned)

Follow the [NeurIPS Code Submission Policy](https://nips.cc/public/guides/CodeSubmissionPolicy):

- ✅ All code must run without pre-training downloads (or document what's needed)
- ✅ Provide clear instructions to reproduce results
- ✅ Report error bars / multiple runs where applicable
- ✅ Disclose limitations and failure cases
- ✅ Use fixed random seeds and document environment
- ✅ Pin all dependencies with versions in `requirements.txt`

## Git Workflow

1. **Branch naming:** `week-N-topic` (e.g., `week-1-rnn`, `week-8-genomics`)
2. **Commit messages:** Clear, descriptive
   - ✅ `Implement BPTT and gradient clipping for RNN`
   - ✅ `Add gradient check test for linear layer`
   - ❌ `changed stuff`, `fix`, `update`
3. **Pull requests:** Include a summary of what was implemented and any key observations
4. **Before pushing:** Run the full test suite
   ```bash
   pytest -v
   black --check .
   flake8 .
   ```

## Documentation & Writing

### Technical Summaries
- Assume reader has ML fundamentals but may not know this specific topic
- Lead with intuition, follow with math
- Always explain notation
- Example structure:
  ```
  **Why LSTMs?** (Problem: vanishing gradients)
  **Intuition:** Gates allow gradients to flow unchanged
  **Formal:** h_t = ... (equations)
  **Gradient flow:** dh_t/dW = ... (trace backward pass)
  **When it matters:** Long sequences (T > 50)
  ```

### Blog Posts / Papers
- Include a "What's the hypothesis?" section
- Show baseline results
- Honestly discuss what didn't work
- Suggest next steps

## Environment & Reproducibility

### Local Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Document Your Environment
If you install additional packages, update `requirements.txt`:
```bash
pip freeze > requirements.txt
```

### Reproducible Runs
Always include random seed:
```python
import torch
import numpy as np

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## AI4Bio Specialization Notes

When working on AI4Bio weeks (8–10):
- Include biological baselines (PCA for single-cell, benchmarked CNNs for genomics)
- Document train/val/test splits carefully (avoid data leakage across loci/cells)
- Compare to published results with clear attribution
- If using public datasets, cite the data source and pre-print first

## Questions?

Refer to:
- Week README for specific goals
- [NeurIPS Code Policy](https://nips.cc/public/guides/CodeSubmissionPolicy)
- Example repos: [nanoGPT](https://github.com/karpathy/nanoGPT), [micrograd](https://github.com/karpathy/micrograd)

---

**Happy coding & researching!**
