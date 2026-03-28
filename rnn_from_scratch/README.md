# Day 1: RNN From Scratch

This folder contains a clean, from-scratch implementation of a vanilla RNN in PyTorch.

## Learning goals

- Implement the vanilla RNN equations directly.
- Track tensor shapes at each step.
- Understand sequence processing with shared weights over time.

## Files

- `rnn_cell.py`: `VanillaRNNCell` and sequence-level `VanillaRNN`.
- `test_rnn_cell.py`: Dummy-data tests that verify output shapes.
- `notes.md`: Conceptual answers for Day 1.

## Core equations

- `h_t = tanh(x_t W_xh + h_{t-1} W_hh + b_h)`
- `y_t = h_t W_hy + b_y`

## Shape cheat sheet

- Input `x`: `(B, T, D)`
- Single step `x_t`: `(B, D)`
- Hidden state `h_t`: `(B, H)`
- `W_xh`: `(D, H)`
- `W_hh`: `(H, H)`
- `b_h`: `(H,)`
- `W_hy`: `(H, O)`
- Output `y_t`: `(B, O)`

## Run

```bash
python test_rnn_cell.py
```

Expected: printed tensor shapes and `All shape checks passed.`
