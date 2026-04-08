# Can a Neural Network Learn to Predict a PRNG?

No -- and that's the point. This project trains a **differentiable logic gate network** to predict the next output of a 16-bit [PCG (Permuted Congruential Generator)](https://www.pcg-random.org/paper.html) from a sliding window of previous outputs. The network uses **learned soft AND/OR/XOR gates** instead of traditional activations -- a structural match for attacking a bit-manipulation algorithm.

## Skills & Frameworks

**Core**: Python, PyTorch, NumPy | **Architecture**: Differentiable logic gate networks, sequential fold design, bipolar/binary bit encoding | **Infrastructure**: Docker, AWS Batch, S3 | **Concepts**: PRNG cryptanalysis, information-theoretic bounds, BCEWithLogitsLoss, softmax gate selection, StepLR scheduling

## Summary

- **Generator**: 16-bit PCG-XSH-RR, 32-bit state, full-period sequence (65,536 outputs)
- **Architecture**: Alternating `Linear(64 -> 256)` and `DiffLogicLayer(256 -> 64)` blocks, 3 layers deep, ~52.7K parameters
- **Core idea**: Each `DiffLogicLayer` learns per-bit soft selections over `{AND, OR, XOR}` via softmax, then folds 4 input timesteps sequentially to preserve temporal ordering
- **Training**: BCEWithLogitsLoss, Adam (lr = 1e-4), StepLR decay (gamma = 0.5 every 10 epochs), 50 epochs
- **Infrastructure**: Dockerized for AWS Batch with S3-backed dataset and result storage

## How to Use

```bash
python data.py        # Generate 65,536-row bipolar bit vector CSV
python network.py     # Train DiffLogicNet, logs to training_log.txt
```

**Docker** (AWS Batch):
```bash
docker build -t pcrng-approx .
docker run -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... pcrng-approx
```

The entrypoint pulls `data.csv` from S3, trains, and uploads results keyed by Batch job ID.

## Architecture

```
Input: (batch, 64)    -- 4 timesteps x 16 bits, flattened
  |
  Linear(64, 256)
  |
  DiffLogicLayer(256 -> 64)   -- learns soft {AND, OR, XOR} per bit
  |
  Linear(64, 256)
  |
  DiffLogicLayer(256 -> 64)
  |
  Linear(64, 256)
  |
  DiffLogicLayer(256 -> 64)
  |
  Linear(64, 16)
  |
Output: (batch, 16)   -- predicted next bit vector
```

**Sequential fold** -- the key design decision. Instead of processing all 4 input vectors in parallel, `DiffLogicLayer` chains them:

$$h_1 = \text{gate}_0(v_0, v_1), \quad h_2 = \text{gate}_1(h_1, v_2), \quad \text{out} = \text{gate}_2(h_2, v_3)$$

Each gate step is a soft mixture of AND, OR, XOR weighted by a learned softmax distribution per bit. This preserves temporal dependency -- $v_3$ sees context from all prior timesteps through $h_2$, mirroring PCG's sequential state transition.

## Results

<p align="center">
  <img src="graphs/train_test_loss.png" width="330"/>
  <img src="graphs/bit_accuracy.png" width="330"/>
</p>

<p align="center">
  <img src="graphs/decimal_mae.png" width="330"/>
  <img src="graphs/generalization_gap.png" width="330"/>
</p>

| Metric | Value | Baseline |
|---|---|---|
| Peak bit accuracy | 62.4% (epoch 3) | 50% |
| Best decimal MAE | ~19,785 (epoch 4) | ~21,845 |
| Final decimal MAE | ~21,900 | ~21,845 |
| Generalization gap | +0.024 by epoch 50 | 0 |

## Analysis

- **Beats random on bits, not on numbers.** 62% bit accuracy comes from exploiting the **marginal bit bias** in PCG-XSH-RR's lower positions (LSB mean of -0.47 vs 0.00) -- not sequential dependency. The network learned the per-bit base rate, not the next state.

- **Generalization gap confirms memorization.** Train loss drops while test loss flatlines and diverges -- the model memorizes training sequences without learning the state transition. This is exactly what a well-designed PRNG should produce.

- **Architecture is sound but information-starved.** A 4-timestep window gives 64 bits of output, but reconstructing a 32-bit hidden state requires inverting a multiply-and-add through the XSH-RR output permutation -- which is specifically designed to destroy that information.

## Key Design Decisions

1. **SIREN to logic gates**: Started with sinusoidal activations (periodic activations for modular arithmetic). Training was unstable with loss surfaces full of local minima. Switched to differentiable logic gates after [Petersen et al. (2022)](https://arxiv.org/abs/2210.08277) -- structural match for a bitwise algorithm.

2. **Independent to sequential fold**: First `DiffLogicLayer` version processed all 4 inputs independently, discarding temporal order. Restructuring as a sequential fold improved convergence speed but didn't raise the accuracy ceiling.

3. **Bipolar encoding**: Generated data as $\{-1, 1\}$ vectors (for potential Hopfield-style models), remapped to $\{0, 1\}$ for logic gates. In hindsight, unnecessary indirection.

## Future Directions

- **Larger context windows** -- information-theoretic lower bound suggests >= 2 full consecutive outputs to reconstruct state
- **Direct state recovery** -- reframe as state inversion rather than next-output prediction
- **Validate on weaker PRNGs first** -- LCG or xorshift32 to isolate architecture limitations from PRNG strength
- **Cross-bit attention** -- current gate layer treats bit positions independently; attention could learn the XSH-RR rotation step
