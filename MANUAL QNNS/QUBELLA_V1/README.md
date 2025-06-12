# QUBELLA V1

## Problem Statement
Learning a quantum rotation transformation using Quantum Neural Networks (QNNs). Specifically, training a QNN to perform a π/2 rotation on the X-axis for any given initial qubit state.

## QNN Design Framework

### 1. Data Collection
- **Dataset size**: 1 (toy problem)
- **Task**: Learn to apply RY(π/2) rotation to arbitrary initial states
- **Target**: Transform initial state |ψ⟩ → RY(π/2)|ψ⟩

### 2. Quantum State Embedding
- **Input encoding**: RY(θ_input) gate
- **Initial state**: |ψ_initial⟩ = RY(θ_input)|0⟩
- **Random input parameter**: θ_input ∈ [0, 2π)

### 3. Ansatz Architecture
- **Circuit depth**: 1 layer
- **Parameterized gate**: RY(θ_weight)
- **Total circuit**: |ψ_output⟩ = RY(θ_weight) RY(θ_input)|0⟩
- **Trainable parameters**: 1 (θ_weight)

### 4. Loss Functions

#### SamplerQNN (Probability-based)
Bhattacharyya coefficient measuring probability distribution similarity:

$$L_{Sampler} = 1 - BC^2$$

where:
$$BC = \sum_i \sqrt{p_i^{target} \cdot p_i^{output}}$$

#### EstimatorQNN (Expectation-based)
Squared difference between Z expectation values:

$$L_{Estimator} = (\langle Z \rangle_{output} - \langle Z \rangle_{target})^2$$

where:
$$\langle Z \rangle = \langle \psi | Z | \psi \rangle$$

### 5. Optimizer
- **Algorithm**: ADAM
- **Learning rate**: 0.01
- **Max iterations**: 1000
- **Hyperparameters**: β₁=0.9, β₂=0.99, ε=1e-8

## Results
- **Target weight**: π/2 ≈ 1.5708 radians
- **SamplerQNN accuracy**: >99.99% fidelity
- **EstimatorQNN accuracy**: >99.99% fidelity
- **Optimization time**: ~4-22 seconds (hardware dependent)

## Technical Implementation
- **Framework**: Qiskit, Qiskit Machine Learning
- **Primitives**: StatevectorEstimator, StatevectorSampler
- **Observable**: Pauli-Z operator
- **Gradient method**: Automatic differentiation 