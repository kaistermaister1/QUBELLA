#!/usr/bin/env python
# coding: utf-8

# Goal: Teach a QNN to rotate a qubit by pi/2 on the x-axis
# Training set size: 1
# Use both EstimatorQNN and SamplerQNN
# Have 1 qubit, 1 gate, and 1 observable Z
# Loss function: arccos(dot product of the output and the target)

from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import RYGate
from qiskit_algorithms.optimizers import ADAM
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

# algorithm_globals.random_seed = 13 # Set the random seed

# Create a quantum circuit with 1 qubit and 1 gate parameter + observable
qc=QuantumCircuit(1)
params = [Parameter("input1"),Parameter("weight1")]
qc.ry(params[0],0) # Rotate 0 qubit to set initial state
qc.ry(params[1],0) # Rotation weight
qc.draw("mpl", style="clifford")
observable = SparsePauliOp.from_list([("Z", 1)])

# Define EstimatorQNN and SamplerQNN
estimator = Estimator() # Choose the simulation estimator (outputs expectation value)
estimator_qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=[params[0]],
    weight_params=[params[1]],
    estimator=estimator,
)
sampler = Sampler() # Choose the simulation sampler (outputs probability distribution)
sampler_qnn = SamplerQNN(
    circuit=qc, 
    input_params=[params[0]],
    weight_params=[params[1]], 
    sampler=sampler
)

# Set random values for input and weight parameter
# Input parameter won't change, but weight will be optimized
inputweight = algorithm_globals.random.random(1)
weight = algorithm_globals.random.random(1)
initial_weight_value = weight.copy()  # Save the initial weight before optimization
print(f"Initial Weight: {weight}")

# Define our loss functions
def target_amps(inputweight):
    initial_state = Statevector.from_instruction(RYGate(inputweight[0]))
    target_state = initial_state.evolve(RYGate(np.pi/2)).data # Extract amplitudes
    return target_state
def Sloss(sampler_qnn_forward, inputweight):
    target_probs = np.abs(target_amps(inputweight))**2
    output_probs = np.array(sampler_qnn_forward).reshape(-1) # Reshape from 2D to 1D array
    BC = (np.sum(np.sqrt(target_probs * output_probs)))**2 # Bhattacharyya coefficient
    return 1 - BC, -BC*np.sqrt(target_probs / output_probs) # Classical fidelity and partial derivative
def Eloss(estimator_qnn_forward, inputweight):
    Z = Operator.from_label("Z")  # 2×2 Pauli-Z matrix
    target_state = target_amps(inputweight)
    target_expectation  = np.vdot(target_state, Z @ target_state).real
    output_expectation = estimator_qnn_forward[0][0]
    return (output_expectation - target_expectation)**2, 2*(output_expectation - target_expectation) # Expectation loss and partial derivative

# Compute loss and gradient
def loss_and_gradient(
    weights: np.ndarray,
    inputs: np.ndarray,
    estimator_qnn=None,
    sampler_qnn=None
) -> tuple[float, np.ndarray]:
    if estimator_qnn is not None:
        y = estimator_qnn.forward(inputs, weights)
        loss, dL_dE = Eloss(y, inputs)
        _, dE_dθ = estimator_qnn.backward(inputs, weights)
        # dE_dθ might have shape (1,1,1) → squeeze to scalar
        dE_dθ = np.squeeze(dE_dθ)
        grad = dL_dE * dE_dθ
    elif sampler_qnn is not None:
        y = sampler_qnn.forward(inputs, weights)
        loss, dL_dE = Sloss(y, inputs)
        _, dE_dθ = sampler_qnn.backward(inputs, weights)
        grad = dL_dE @ np.squeeze(dE_dθ)
    else:
        raise ValueError("Provide either estimator_qnn or sampler_qnn")

    # Always return a 1-D array
    return float(loss), np.atleast_1d(grad)

# Instantiate ADAM optimizer
optimizer = ADAM(
    maxiter=1000,    # total number of Adam steps
    lr=0.01,        # learning rate
    beta_1=0.9,
    beta_2=0.99,
    eps=1e-8
)

# Build callables for ADAM
def make_adam_functions(qnn, is_estimator=True):
    def loss_fn(w):
        loss, _ = loss_and_gradient(
            w, inputweight,
            estimator_qnn=qnn if is_estimator else None,
            sampler_qnn=qnn if not is_estimator else None,
        )
        return loss
    def grad_fn(w):
        _, grad = loss_and_gradient(
            w, inputweight,
            estimator_qnn=qnn if is_estimator else None,
            sampler_qnn=qnn if not is_estimator else None,
        )
        return grad
    return loss_fn, grad_fn

# Show initial state before optimization
estimator_qnn_forward = estimator_qnn.forward(inputweight, weight)
sampler_qnn_forward = sampler_qnn.forward(inputweight, weight)
print("\n" + "="*50)
print("INITIAL STATE BEFORE OPTIMIZATION")
print("="*50)
initial_vector = Statevector.from_instruction(RYGate(inputweight[0]))
print(f"Initial vector (after input parameter): {initial_vector.data}")
print(f"Initial RY gate rotation value (weight): {weight[0]:.6f}")
print(f"Initial EstimatorQNN forward pass result: {estimator_qnn_forward}")
print(f"Initial SamplerQNN forward pass result: {sampler_qnn_forward}")

# Run the optimization
print("\nRunning ADAM optimization...")
s_loss_fn, s_grad_fn = make_adam_functions(sampler_qnn, is_estimator=False)
e_loss_fn, e_grad_fn = make_adam_functions(estimator_qnn, is_estimator=True)
initial_weights = np.array([weight], dtype=float)

# Time the SamplerQNN optimization
start_time_s = time.time()
s_result = optimizer.minimize(fun=s_loss_fn, x0=initial_weights, jac=s_grad_fn)
end_time_s = time.time()
s_optimization_time = end_time_s - start_time_s

# Time the EstimatorQNN optimization
start_time_e = time.time()
e_result = optimizer.minimize(fun=e_loss_fn, x0=initial_weights, jac=e_grad_fn)
end_time_e = time.time()
e_optimization_time = end_time_e - start_time_e

new_s_weights = s_result.x
new_e_weights = e_result.x

# Show results after optimization
print("\n" + "="*50)
print("RESULTS AFTER OPTIMIZATION")
print("="*50)
print(f"SamplerQNN optimized weight: {new_s_weights.item():.6f}")
print(f"SamplerQNN optimization time: {s_optimization_time:.2f} seconds")
s_final_forward = sampler_qnn.forward(inputweight, new_s_weights)
print(f"SamplerQNN final forward pass result: {s_final_forward}")

print(f"\nEstimatorQNN optimized weight: {new_e_weights.item():.6f}")
print(f"EstimatorQNN optimization time: {e_optimization_time:.2f} seconds")
e_final_forward = estimator_qnn.forward(inputweight, new_e_weights)
print(f"EstimatorQNN final forward pass result: {e_final_forward}")

## -------------- STATISTICS -------------- ##

# Visualization of quantum states
print("\n" + "="*50)
print("QUANTUM STATE VISUALIZATION")
print("="*50)

# Calculate final states after optimization
s_final_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(new_s_weights.item()))
e_final_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(new_e_weights.item()))
target_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(np.pi/2))

# Create visualization with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('QUBELLA V1', fontsize=16, fontweight='bold')

# Plot 1: Quantum State Vectors (Simplified and Zoomed)
vectors = [initial_vector.data, target_vector.data, s_final_vector.data, e_final_vector.data]
labels = ['Initial', 'Target', 'SamplerQNN', 'EstimatorQNN']
colors = ['blue', 'green', 'red', 'orange']
line_styles = ['-', '-', '--', ':']
line_widths = [1.5, 2.5, 1.5, 1.5] # Thin lines, Target slightly thicker

for i, (vec, color, style, width) in enumerate(zip(vectors, colors, line_styles, line_widths)):
    ax1.plot([0, vec[0].real], [0, vec[1].real], color=color, linewidth=width, linestyle=style, alpha=0.9)
    ax1.plot(vec[0].real, vec[1].real, 'o', color=color, markersize=4)

# Draw unit circle
theta = np.linspace(0, 2*np.pi, 100)
ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=1)

# Create a zoom inset to magnify the target area
axins = zoomed_inset_axes(ax1, zoom=8, loc='center left') # zoom = 8x

# Plot the target and result vectors in the inset
for i, (vec, color, style, width) in enumerate(zip(vectors[1:], colors[1:], line_styles[1:], line_widths[1:])):
    axins.plot([0, vec[0].real], [0, vec[1].real], color=color, linewidth=width*1.5, linestyle=style)
    axins.plot(vec[0].real, vec[1].real, 'o', color=color, markersize=5)

# Set the limits of the zoomed area
target_x, target_y = target_vector.data[0].real, target_vector.data[1].real
zoom_range_x = 0.05
zoom_range_y = 0.05
axins.set_xlim(target_x - zoom_range_x, target_x + zoom_range_x)
axins.set_ylim(target_y - zoom_range_y, target_y + zoom_range_y)

# Hide ticks and labels on the inset for cleanliness
axins.set_xticks([])
axins.set_yticks([])
axins.grid(True, linestyle='--', alpha=0.4)
axins.set_facecolor('whitesmoke')

# Draw the box and connectors showing the zoom area
mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")

ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_xlabel('Amplitude of |0⟩', fontsize=12)
ax1.set_ylabel('Amplitude of |1⟩', fontsize=12)
ax1.set_title('Quantum State Vectors', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Create a clean legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=color, lw=width, linestyle=style, label=label, marker='o')
                  for color, width, style, label in zip(colors, line_widths, line_styles, labels)]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Plot 2: Rotation weights comparison
states = ['Initial', 'Target', 'SamplerQNN', 'EstimatorQNN']
weights = [initial_weight_value.item(), np.pi/2, new_s_weights.item(), new_e_weights.item()]

bars = ax2.bar(range(len(states)), weights, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Quantum States', fontsize=12)
ax2.set_ylabel('RY Gate Weight (radians)', fontsize=12)
ax2.set_title('Learned Rotation Weights', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(states)))
ax2.set_xticklabels(states)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=np.pi/2, color='green', linestyle='--', alpha=0.7, label=f'Target (π/2 = {np.pi/2:.4f})')

# Add value labels on bars
for bar, weight in zip(bars, weights):
    height = bar.get_height()
    ax2.annotate(f'{weight:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

ax2.legend(fontsize=10, loc='upper right')

# Plot 3: Expectation values comparison (for EstimatorQNN analysis)
# Calculate Z expectation values for all states
Z = Operator.from_label("Z")
initial_exp = np.vdot(initial_vector.data, Z @ initial_vector.data).real
target_exp = np.vdot(target_vector.data, Z @ target_vector.data).real
s_final_exp = np.vdot(s_final_vector.data, Z @ s_final_vector.data).real
e_final_exp = np.vdot(e_final_vector.data, Z @ e_final_vector.data).real

exp_values = [initial_exp, target_exp, s_final_exp, e_final_exp]
bars3 = ax3.bar(range(len(states)), exp_values, color=colors, alpha=0.7, edgecolor='black')

ax3.set_xlabel('Quantum States', fontsize=12)
ax3.set_ylabel('⟨Z⟩ Expectation Value', fontsize=12)
ax3.set_title('Z Expectation Values', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(states)))
ax3.set_xticklabels(states)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=target_exp, color='green', linestyle='--', alpha=0.7, label=f'Target = {target_exp:.4f}')

# Add value labels on bars
for bar, exp_val in zip(bars3, exp_values):
    height = bar.get_height()
    ax3.annotate(f'{exp_val:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

ax3.legend(fontsize=10, loc='upper right')

# Adjust subplot parameters to prevent overlap and make room for text
plt.tight_layout()
fig.subplots_adjust(top=0.88, bottom=0.25, left=0.05, right=0.98, wspace=0.3)

# Create the simplified text for the info box (optimization times only)
info_text = (
    f"ADAM Optimization Times:\n"
    f"  SamplerQNN:   {s_optimization_time:.2f} s\n"
    f"  EstimatorQNN: {e_optimization_time:.2f} s"
)

# Add the info box to the bottom left corner
fig.text(0.02, 0.02, info_text, ha='left', va='bottom', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

plt.show()

## -------------- COST LANDSCAPE VISUALIZATION -------------- ##
print("\n" + "="*50)
print("COST LANDSCAPE VISUALIZATION")
print("="*50)

# Generate a range of weight values to plot the cost landscape
weight_range = np.linspace(-2*np.pi, 2*np.pi, 200)

# Calculate the loss for each weight for both SamplerQNN and EstimatorQNN
s_losses = [s_loss_fn(np.array([w])) for w in weight_range]
e_losses = [e_loss_fn(np.array([w])) for w in weight_range]

# Create a new figure with two 2D subplots
fig_cost, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig_cost.suptitle('QNN Cost Landscapes', fontsize=16, fontweight='bold')

# Mark the important weight values
print(f"Using saved initial weight: {initial_weight_value}")
initial_w = np.array(initial_weight_value).item()
target_w = np.pi/2
s_optimized_w = new_s_weights.item()
e_optimized_w = new_e_weights.item()

# Calculate the corresponding loss values (ensuring they sit on the curves)
s_initial_loss = s_loss_fn(np.array([initial_w]))
s_target_loss = s_loss_fn(np.array([target_w]))
s_optimized_loss = s_loss_fn(np.array([s_optimized_w]))

e_initial_loss = e_loss_fn(np.array([initial_w]))
e_target_loss = e_loss_fn(np.array([target_w]))
e_optimized_loss = e_loss_fn(np.array([e_optimized_w]))

# Plot 1: SamplerQNN Loss Landscape
ax1.plot(weight_range, s_losses, 'r-', linewidth=2, label='SamplerQNN Loss')
ax1.scatter(initial_w, s_initial_loss, c='blue', marker='o', s=100, label='Initial Weight', zorder=5)
ax1.scatter(target_w, s_target_loss, c='green', marker='*', s=150, label='Target Weight (π/2)', zorder=5)
ax1.scatter(s_optimized_w, s_optimized_loss, c='red', marker='X', s=100, label='Optimized Weight', zorder=5)

ax1.set_xlabel('Weight Parameter (radians)', fontsize=12)
ax1.set_ylabel('Loss (Cost)', fontsize=12)
ax1.set_title('SamplerQNN Loss Landscape', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='upper right')

# Add vertical lines for important points
ax1.axvline(x=initial_w, color='blue', linestyle='--', alpha=0.5)
ax1.axvline(x=target_w, color='green', linestyle='--', alpha=0.5)
ax1.axvline(x=s_optimized_w, color='red', linestyle='--', alpha=0.5)

# Plot 2: EstimatorQNN Loss Landscape
ax2.plot(weight_range, e_losses, 'orange', linewidth=2, label='EstimatorQNN Loss')
ax2.scatter(initial_w, e_initial_loss, c='blue', marker='o', s=100, label='Initial Weight', zorder=5)
ax2.scatter(target_w, e_target_loss, c='green', marker='*', s=150, label='Target Weight (π/2)', zorder=5)
ax2.scatter(e_optimized_w, e_optimized_loss, c='orange', marker='X', s=100, label='Optimized Weight', zorder=5)

ax2.set_xlabel('Weight Parameter (radians)', fontsize=12)
ax2.set_ylabel('Loss (Cost)', fontsize=12)
ax2.set_title('EstimatorQNN Loss Landscape', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, loc='upper right')

# Add vertical lines for important points
ax2.axvline(x=initial_w, color='blue', linestyle='--', alpha=0.5)
ax2.axvline(x=target_w, color='green', linestyle='--', alpha=0.5)
ax2.axvline(x=e_optimized_w, color='orange', linestyle='--', alpha=0.5)

# Add text annotations with values
ax1.text(0.02, 0.98, f'Initial: {initial_w:.3f}\nTarget: {target_w:.3f}\nOptimized: {s_optimized_w:.3f}', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add training time and iterations to EstimatorQNN plot
estimator_info = f'Initial: {initial_w:.3f}\nTarget: {target_w:.3f}\nOptimized: {e_optimized_w:.3f}\nTime: {e_optimization_time:.2f}s\nIterations: {e_result.nfev}'
ax2.text(0.02, 0.98, estimator_info, 
         transform=ax2.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()