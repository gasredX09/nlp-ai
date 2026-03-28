# LLMs and AI4Bio Research: From-Scratch Implementation Roadmap

A daily implementation journey to master sequence models, build domain expertise in computational biology, and develop a research-grade portfolio.

## Vision

This repository documents a systematic path to:
1. **Mechanistic Understanding** — Understand and re-derive why components exist (gates in LSTMs, attention scaling, normalization) and debug training by reasoning about gradients.
2. **Implementation Competence** — Build foundational architectures end-to-end: forward pass, backward pass (autograd), optimizers, training loops, evaluation.
3. **AI4Bio Leadership** — Connect biological questions to modeling choices, evaluate foundation models on real datasets, and identify frontier research gaps.

## Project Structure

```
nlpai/
├── 01_rnn_from_scratch/          # Week 1: RNN fundamentals and BPTT
│   ├── rnn_cell.py              # Core RNN cell implementation
│   ├── test_rnn_cell.py         # Unit tests and gradient checks
│   ├── notes.md                 # Learning notes and derivations
│   └── README.md                # Weekly goals and checkpoints
│
├── 02_lstm_gru/                 # Week 2: Gating mechanisms
├── 03_attention_mechanism/      # Week 3: Scaled dot-product attention
├── 04_transformer/              # Week 4: Transformer from scratch
├── 05_gpt_training/             # Week 5: GPT-style language model
├── 06_llm_scaling/              # Week 6: Scaling laws and optimization
├── 07_alignment_finetuning/     # Week 7: RLHF, DPO, LoRA
├── 08_ai4bio_genomics/          # Week 8: Genomic sequence modeling
├── 09_single_cell_modeling/     # Week 9: Single-cell foundation models
├── 10_protein_molecules/        # Week 10: Protein and molecular modeling
│
├── notebooks/                   # Paper-to-code notebooks
├── data/                        # Datasets (excluded from git)
├── models/                      # Trained weights (excluded from git)
├── results/                     # Experimental results
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
└── README.md                    # This file
```

## Weekly Cadence

Each week follows a **two-track loop**:

**Track A: Build & Verify (4–5 days)**
- Implement one architectural component
- Write unit tests with gradient checks
- Train on a toy dataset and verify basic behavior
- **Artifact:** Passing test suite + training curve

**Track B: Read & Summarize (2–3 sessions)**
- Read 1–2 papers on topic
- Produce a 1-page technical note with diagrams and math
- Implement one minimal reproduction experiment
- **Artifact:** Paper notes + experimental notebook

### Weekly Deliverables

Each week's directory should contain:
- Source code with inline comments explaining mechanics
- `test_*.py` with comprehensive unit tests and gradient checks
- `notes.md` with derivations, diagrams, and key insights
- `README.md` with concrete goals and evaluation metrics
- `experiment.ipynb` with toy dataset evaluation

## Week 1: RNN from Scratch

**Goals:**
- [ ] Implement RNN cell forward and backward pass (BPTT)
- [ ] Understand vanishing/exploding gradients
- [ ] Build character-level language model baseline
- [ ] Gradient check: verify backprop correctness
- [ ] Train on tiny Shakespeare dataset, check loss curves

**Key References:**
- Karpathy, A. (2015). "The Unreasonable Effectiveness of Recurrent Neural Networks."
- Bengio et al. (1994). "Learning long-term dependencies with gradient descent is difficult."

**Must-Understand:**
- Backprop through time (BPTT) unrolling
- Why gradients vanish over long sequences
- How teacher forcing works in training

## Key References by Topic

### Foundational (Autograd & Optimization)
- **micrograd** (Karpathy) — Tiny reverse-mode autograd (~100 lines)
- **Kaplan et al.** — "Scaling Laws for Neural Language Models"

### Sequence Models
- **LSTM paper** (Hochreiter & Schmidhuber, 1997)
- **Seq2Seq** (Sutskever et al., 2014)
- **Attention Is All You Need** (Vaswani et al., 2017)
- **nanoGPT** (Karpathy) — Compact GPT training codebase

### Post-Training & Alignment
- **InstructGPT** (Ouyang et al., 2022) — RLHF foundations
- **DPO** (Rafailov et al., 2023) — Preference optimization without RL loops
- **LoRA** (Hu et al., 2021) — Parameter-efficient finetuning
- **QLoRA** (Dettmers et al., 2023) — Memory-efficient finetuning

### AI4Bio Specialization

**Genomics & Regulatory Modeling:**
- DeepSEA — Regulatory variant effects from sequence
- Enformer — Long-range sequence modeling for expression
- Borzoi — RNA-seq coverage prediction at scale
- DNABERT — DNA-language models
- Nucleotide Transformer — Large-scale foundation models

**Single-Cell & Multi-Omics:**
- scVI — Deep generative model for transcriptomics
- totalVI — Joint RNA + protein (CITE-seq) modeling
- scGPT — Single-cell foundation model (billions of cells)
- MultiVI — Multimodal integration with missing modalities

**Proteins & Molecules:**
- AlphaFold2 — Structure prediction
- AlphaFold3 — Diffusion-based complex modeling
- ProteinMPNN — Sequence design from structure
- ESMFold — Structure from sequence
- DiffDock — Diffusion-based docking

## Practice Stack

- **Deep-ML** — ML coding challenges (2–3 problems/week)
- **Kaggle** — End-to-end modeling competitions
- **CS336** (Stanford) — Language model components and scaling
- **Awesome-LLM**, **awesome-deepbio** — Curated paper lists
- **Hugging Face Course** — Practical transformer and NLP workflows

## Repository Standards

All code follows **research-grade reproducibility**:
- ✅ Comprehensive unit tests with gradient checks
- ✅ Training curves and ablations documented
- ✅ Clear README for each module with reproducible runs
- ✅ Requirements pinned, environment reproducible
- ✅ Results reported with error bars and baselines
- ✅ Code formatted (Black, isort), type-hinted where helpful

**Checklist:** Follow [NeurIPS Code Submission Policy](https://nips.cc/public/guides/CodeSubmissionPolicy) and "Releasing Research Code" best practices.

## Setup Instructions

```bash
# Clone and navigate
git clone <repo>
cd nlpai

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests for a week
pytest 01_rnn_from_scratch/ -v

# Train a model
python 01_rnn_from_scratch/train.py --epochs 50 --lr 0.001
```

## Portfolio & Communication

**Three portfolio pillars:**

1. **Reproduction-Grade Implementations**
   - Pick 2–3 load-bearing papers (e.g., Transformer, Enformer, scGPT)
   - Reproduce with careful evaluation and ablations
   - Clean repo, clear README, passing tests

2. **Technical Writing**
   - 1–2 short research posts answering: hypothesis → baseline → failures → next steps
   - Emphasize reproducibility and honest limitations

3. **Signature AI4Bio Project**
   - Foundation-model evaluation benchmark
   - Compare embeddings across tasks
   - Publish results, document where models fail

## Weekly Progress Tracking

1. **Day 1–2:** Read paper(s), sketch implementation plan
2. **Day 3–4:** Core implementation + unit tests
3. **Day 5:** Toy experiment + training curve
4. **Day 6:** Refine, document, write technical note
5. **Day 7:** Reflect, plan next week, polish README

## Resources

- **Papers with Code** → Hugging Face Papers (trending)
- **arXiv Recommender** — Personalized paper feeds
- **Awesome-LLM** — Curated LLM papers and repos
- **Awesome-DeepBio** — Computational biology resource hub
- **NeurIPS Code Policy** — Reproducibility checklist

## Getting Started

Start with **01_rnn_from_scratch**:
```bash
cd 01_rnn_from_scratch
python test_rnn_cell.py  # Run gradient checks
python train.py --epochs 50  # Train on toy data
```

Check the week's README for concrete goals and expected outcomes.

---

**Last Updated:** March 2026  
**Status:** In Progress (Week 1 — RNN Fundamentals)
