# Day 1 Notes: Vanilla RNN Concepts

## a) Why are RNN weights shared across time?

RNNs apply the same transition rule at every time step so the model can process sequences of different lengths with a fixed number of parameters. Sharing weights also encodes the idea that the same type of temporal update should happen at each position in the sequence.

## b) What is the hidden state?

The hidden state is a learned summary of past information up to time step `t`. It acts like memory: `h_t` combines the new input `x_t` with previous memory `h_{t-1}`.

## c) Why do gradients vanish/explode in RNNs?

Backpropagation through time multiplies many Jacobians across long sequences. Repeated multiplication by values smaller than 1 causes vanishing gradients; repeated multiplication by values larger than 1 causes exploding gradients.

## d) Why is RNN better than feedforward for sequences?

A feedforward network treats each input independently unless we manually engineer context windows. An RNN naturally carries context through the hidden state, making it sequence-aware by design.

## e) Why did transformers largely replace RNNs in LLMs?

RNNs are sequential in time and hard to parallelize over tokens, and they struggle with very long-range dependencies. Transformers use attention, which captures long-range interactions more directly and allows much better parallel training efficiency.
