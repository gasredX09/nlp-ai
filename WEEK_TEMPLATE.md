# Week Template

Use this as a template when starting a new week. Copy and adapt the structure below.

## Directory Structure
```
XX_topic/
├── README.md              # Goals, checkpoints, equations
├── module.py              # Core implementation
├── test_module.py         # Unit tests with gradient checks
├── notes.md               # Paper notes and derivations
├── experiment.ipynb       # Toy dataset training
└── results.txt            # Metrics and observations
```

## README.md Template

```markdown
# Week X: [Topic]

## Learning Goals
- [ ] Implement [core component]
- [ ] Verify with gradient checks
- [ ] Train on toy dataset
- [ ] Achieve [target metric]
- [ ] Document insights

## Key Concepts

### Problem Statement
Why does this component matter? What problem does it solve?

### Core Equations
- Equation 1: $h_t = ...$
- Equation 2: $\nabla h = ...$

### Key References
- [Paper 1](link) - What it teaches
- [Paper 2](link) - What it teaches

## Shape Reference
| Tensor | Shape | Notes |
|--------|-------|-------|
| input  | (B, T, D) | Batch × Time × Dimension |
| output | (B, T, O) | Batch × Time × Output |

## Implementation Checklist
- [ ] Forward pass complete and tested
- [ ] Backward pass with gradient checks
- [ ] Toy experiment runs
- [ ] Training curve is reasonable
- [ ] Code documented and formatted

## How to Run
```bash
# Run tests
python test_module.py -v

# Run experiment
jupyter lab experiment.ipynb
```

## Expected Results
- Test output: All checks pass
- Training curve: Loss decreases smoothly
- Gradient: Numerical vs analytic within 1e-5

## Observations
(To be filled after implementation)

---
**Status:** In Progress  
**Completed:** [date]
```

## notes.md Template

```markdown
# Week X Notes: [Topic]

## Paper Summary: [Key Paper]

### Main Idea
One sentence statement of the core contribution.

### Why It Matters
The problem it solves and impact on the field.

### Key Equations
Rewrite and explain key equations from the paper.

### Gradient Flow
How do gradients propagate? Why does this matter?

### Strengths & Limitations
What works well? What doesn't?

### My Implementation Insights
How does the theory translate to code?

---

## Key Insights

1. **Insight**: Explanation
2. **Insight**: Explanation

## Questions for Next Week
- What question arises from this work?
- What's the next logical step?

---
**Reading Time**: X hours  
**Implementation Time**: Y hours
```

## test_module.py Template

```python
"""
Unit tests and gradient checks for [module].

Tests verify:
1. Output shapes are correct
2. Gradients are computed correctly (numerical vs analytic)
3. Edge cases are handled
"""

import torch
import pytest
from module import YourClass


class TestYourClass:
    """Test suite for YourClass."""
    
    def test_forward_shape(self):
        """Verify output shapes."""
        batch_size, seq_len, dim = 2, 10, 8
        model = YourClass(dim)
        x = torch.randn(batch_size, seq_len, dim)
        out = model(x)
        assert out.shape == torch.Size([batch_size, seq_len, dim])
    
    def test_gradient_check(self):
        """Verify backprop correctness via finite differences."""
        def numerical_gradient(f, x, eps=1e-5):
            grad = torch.zeros_like(x)
            for i in range(x.numel()):
                x_plus = x.clone()
                x_plus.view(-1)[i] += eps
                x_minus = x.clone()
                x_minus.view(-1)[i] -= eps
                
                grad.view(-1)[i] = (f(x_plus) - f(x_minus)) / (2 * eps)
            return grad
        
        # Your gradient check here
        pass
    
    def test_edge_case(self):
        """Test boundary conditions."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## experiment.ipynb Template

Structure your notebook with these cells:

1. **Setup** — Imports and configuration
2. **Load Data** — Toy dataset
3. **Instantiate Model** — Create instance
4. **Training Loop** — Train for N epochs
5. **Evaluation** — Compute metrics
6. **Visualization** — Plot training curves, attention, etc.
7. **Observations** — What did you learn?

---

**Use this template to maintain consistency and quality across all weeks.**
