# Goal: Teach a QNN to rotate a qubit by pi/2 on the x-axis
# Training set size: 1
# Use both EstimatorQNN and SamplerQNN
# Have 1 qubit, 2 gates, and 1 observable Z

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
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir = os.path.join(script_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# =============================================================================
# SIMULATOR SETUP - Choose ONE of the options below by commenting/uncommenting
# =============================================================================

# --- OPTION 1: Automatic GPU Detection ---
# This will try to use qiskit-aer-gpu if available, otherwise it will
# fall back to the default CPU-based simulators.
try:
    # qiskit-aer-gpu is only officially distributed for Linux.
    # On Windows/macOS, this will fail unless built from source.
    from qiskit_aer.primitives import Estimator as AerEstimator
    from qiskit_aer.primitives import Sampler as AerSampler
    from functools import partial
    # Check if a GPU device is available in the installed qiskit-aer
    import qiskit_aer
    if 'GPU' in qiskit_aer.AerSimulator().available_devices():
        print("GPU detected. Using Aer-GPU simulators.")
        Estimator = partial(AerEstimator, backend_options={"device": "GPU", "method": "statevector"})
        Sampler = partial(AerSampler, backend_options={"device": "GPU", "method": "statevector"})
    else:
        raise RuntimeError("GPU device not found in qiskit-aer.")
except (ImportError, RuntimeError) as e:
    print(f"GPU simulators not available ({e}). Falling back to default CPU simulators.")
    print("Note: Pre-compiled qiskit-aer-gpu is only available on Linux.")
    from qiskit.primitives import StatevectorEstimator as Estimator
    from qiskit.primitives import StatevectorSampler as Sampler


# --- OPTION 2: (CPU Only) ---
# from qiskit.primitives import StatevectorEstimator as Estimator
# from qiskit.primitives import StatevectorSampler as Sampler

# =============================================================================

# Choose primitives
estimator = Estimator()
sampler = Sampler()

algorithm_globals.random_seed = 13342892 # Set the random seed

# Create a quantum circuit with 1 qubit and 2 gate parameters + observable
qc=QuantumCircuit(1)
params = [Parameter("input1"),Parameter("weight1"),Parameter("weight2")]
qc.ry(params[0],0) # Rotate 0 qubit to set initial state
qc.ry(params[1],0) # Rotation weight 1
qc.ry(params[2],0) # Rotation weight 2
circuit_plot = qc.draw("mpl", style="clifford")
circuit_plot.savefig(os.path.join(plots_dir, "v1.1circuit.png"))
observable = SparsePauliOp.from_list([("Z", 1)])

# Define EstimatorQNN and SamplerQNN
estimator_qnn = EstimatorQNN(
    circuit=qc,
    observables=observable,
    input_params=[params[0]],
    weight_params=[params[1],params[2]],
    estimator=estimator,
)
sampler_qnn = SamplerQNN(
    circuit=qc, 
    input_params=[params[0]],
    weight_params=[params[1],params[2]], 
    sampler=sampler
)

# Set random values for input and weight parameter
# Input parameter won't change, but weight will be optimized
inputweight = algorithm_globals.random.random(1)
weights = algorithm_globals.random.random(2)
initial_weight_values = weights.copy()  # Save the initial weight before optimization
print(f"Initial Weight: {weights}")

# Define our loss functions
def target_amps(inputweight):
    initial_state = Statevector.from_instruction(RYGate(inputweight[0]))
    target_state = initial_state.evolve(RYGate(np.pi/2)).data # Extract amplitudes
    return target_state
def Sloss(sampler_qnn_forward, inputweight):
    target_probs = np.abs(target_amps(inputweight))**2
    output_probs = np.array(sampler_qnn_forward).reshape(-1) # Reshape from 2D to 1D array
    
    # Add small epsilon to prevent division by zero
    eps = 1e-12
    output_probs_safe = np.maximum(output_probs, eps)
    
    BC = (np.sum(np.sqrt(target_probs * output_probs_safe)))**2 # Bhattacharyya coefficient
    gradient = -BC * np.sqrt(target_probs / output_probs_safe) # Classical fidelity partial derivative
    
    # Handle potential NaN/inf values in gradient
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    
    return 1 - BC, gradient
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

# Instantiate ADAM optimizer with more conservative settings for quantum optimization
optimizer = ADAM(
    maxiter=1000,     # Reduce iterations since we're taking smaller, more stable steps
    lr=0.01,        # Much smaller learning rate for quantum optimization (was 0.01)
    beta_1=0.9,
    beta_2=0.99,    # Slightly higher beta_2 for more momentum averaging
    eps=1e-8
)

# Build callables for ADAM with loss tracking
def adam_functions(qnn, is_estimator=True):
    loss_history = []
    iteration_count = [0]  # Use list to allow modification in nested function
    
    def loss_fn(w):
        loss, _ = loss_and_gradient(
            w, inputweight,
            estimator_qnn=qnn if is_estimator else None,
            sampler_qnn=qnn if not is_estimator else None,
        )
        loss_history.append(loss)
        iteration_count[0] += 1
        return loss
    
    def grad_fn(w):
        _, grad = loss_and_gradient(
            w, inputweight,
            estimator_qnn=qnn if is_estimator else None,
            sampler_qnn=qnn if not is_estimator else None,
        )
        # Clip gradients to prevent extreme values that cause oscillations
        grad = np.clip(grad, -1.0, 1.0)
        return grad
    
    return loss_fn, grad_fn, loss_history

# Show initial state before optimization
estimator_qnn_forward = estimator_qnn.forward(inputweight, weights)
sampler_qnn_forward = sampler_qnn.forward(inputweight, weights)
print("\n" + "="*50)
print("INITIAL STATE BEFORE OPTIMIZATION")
print("="*50)
initial_vector = Statevector.from_instruction(RYGate(inputweight[0]))
print(f"Initial vector (after input parameter): {initial_vector.data}")
print(f"Initial RY gates rotation values (weights): {weights[0]:.6f}, {weights[1]:.6f}")
print(f"Initial EstimatorQNN forward pass result: {estimator_qnn_forward}")
print(f"Initial SamplerQNN forward pass result: {sampler_qnn_forward}")

# Run the optimization with loss tracking
print("\nRunning ADAM optimization...")
s_loss_fn, s_grad_fn, s_loss_history = adam_functions(sampler_qnn, is_estimator=False)
e_loss_fn, e_grad_fn, e_loss_history = adam_functions(estimator_qnn, is_estimator=True)
initial_weights = weights.copy()  # Use 1D array directly

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
print(f"SamplerQNN optimized weights: {new_s_weights[0]:.6f}, {new_s_weights[1]:.6f}")
print(f"SamplerQNN optimization time: {s_optimization_time:.2f} seconds")
s_final_forward = sampler_qnn.forward(inputweight, new_s_weights)
print(f"SamplerQNN final forward pass result: {s_final_forward}")

print(f"\nEstimatorQNN optimized weights: {new_e_weights[0]:.6f}, {new_e_weights[1]:.6f}")
print(f"EstimatorQNN optimization time: {e_optimization_time:.2f} seconds")
e_final_forward = estimator_qnn.forward(inputweight, new_e_weights)
print(f"EstimatorQNN final forward pass result: {e_final_forward}")

## -------------- STATISTICS -------------- ##

print("\n" + "="*50)
print("QUANTUM STATE VISUALIZATION")
print("="*50)

# Calculate final states after optimization
s_final_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(new_s_weights[0])).evolve(RYGate(new_s_weights[1]))
e_final_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(new_e_weights[0])).evolve(RYGate(new_e_weights[1]))
target_vector = Statevector.from_instruction(RYGate(inputweight[0])).evolve(RYGate(np.pi/2))

# Create visualization with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('QUBELLA V1.1 - 2 Weight Parameters', fontsize=16, fontweight='bold')

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

# Plot 2: Combined rotation weights comparison
states = ['Initial W1', 'Initial W2', 'SamplerQNN W1', 'SamplerQNN W2', 'EstimatorQNN W1', 'EstimatorQNN W2']
weight_values = [initial_weight_values[0], initial_weight_values[1], new_s_weights[0], new_s_weights[1], new_e_weights[0], new_e_weights[1]]
bar_colors = ['lightblue', 'lightblue', 'red', 'red', 'orange', 'orange']

bars = ax2.bar(range(len(states)), weight_values, color=bar_colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Weight Parameters', fontsize=12)
ax2.set_ylabel('RY Gate Weight (radians)', fontsize=12)
ax2.set_title('Learned Rotation Weights (2 Parameters)', fontsize=14, fontweight='bold')
ax2.set_xticks(range(len(states)))
ax2.set_xticklabels(states, rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# Add value labels on bars
for bar, weight in zip(bars, weight_values):
    height = bar.get_height()
    ax2.annotate(f'{weight:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

# Plot 3: Expectation values comparison (for EstimatorQNN analysis)
# Calculate Z expectation values for all states
Z = Operator.from_label("Z")
initial_exp = np.vdot(initial_vector.data, Z @ initial_vector.data).real
target_exp = np.vdot(target_vector.data, Z @ target_vector.data).real
s_final_exp = np.vdot(s_final_vector.data, Z @ s_final_vector.data).real
e_final_exp = np.vdot(e_final_vector.data, Z @ e_final_vector.data).real

exp_values = [initial_exp, target_exp, s_final_exp, e_final_exp]
exp_states = ['Initial', 'Target', 'SamplerQNN', 'EstimatorQNN']
bars3 = ax3.bar(range(len(exp_states)), exp_values, color=colors, alpha=0.7, edgecolor='black')

ax3.set_xlabel('Quantum States', fontsize=12)
ax3.set_ylabel('⟨Z⟩ Expectation Value', fontsize=12)
ax3.set_title('Z Expectation Values', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(exp_states)))
ax3.set_xticklabels(exp_states)
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
    f"  EstimatorQNN: {e_optimization_time:.2f} s\n"
    f"Initial Weights: [{initial_weight_values[0]:.3f}, {initial_weight_values[1]:.3f}]"
)

# Add the info box to the bottom left corner
fig.text(0.02, 0.02, info_text, ha='left', va='bottom', fontsize=10,
         bbox=dict(boxstyle="round,pad=0.5", facecolor='whitesmoke', alpha=0.9, edgecolor='gray'))

plt.savefig(os.path.join(plots_dir, "v1.1stats.png"))
plt.show()

## -------------- 3D COST LANDSCAPE VISUALIZATION -------------- ##
print("\n" + "="*50)
print("3D COST LANDSCAPE VISUALIZATION")
print("="*50)

# Generate a grid of weight values for 3D plotting (coarser grid for performance)
w1_range = np.linspace(-2*np.pi, 2*np.pi, 50)
w2_range = np.linspace(-2*np.pi, 2*np.pi, 50)
W1, W2 = np.meshgrid(w1_range, w2_range)

# Calculate losses for 3D surface (this might take a moment)
print("Computing 3D loss surfaces... (this may take a moment)")
s_losses_3d = np.zeros_like(W1)
e_losses_3d = np.zeros_like(W1)

for i in range(len(w1_range)):
    for j in range(len(w2_range)):
        weights_ij = np.array([W1[i,j], W2[i,j]])
        s_losses_3d[i,j] = s_loss_fn(weights_ij)
        e_losses_3d[i,j] = e_loss_fn(weights_ij)

# Create 3D plots
fig_3d = plt.figure(figsize=(16, 8))
fig_3d.suptitle('3D Cost Landscapes - 2 Weight Parameters', fontsize=16, fontweight='bold')

# SamplerQNN 3D surface
ax1_3d = fig_3d.add_subplot(121, projection='3d')
surf1 = ax1_3d.plot_surface(W1, W2, s_losses_3d, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

# Mark important points
ax1_3d.scatter(initial_weight_values[0], initial_weight_values[1], s_loss_fn(initial_weight_values), 
               color='blue', s=100, label='Initial')
ax1_3d.scatter(new_s_weights[0], new_s_weights[1], s_loss_fn(new_s_weights), 
               color='red', s=100, label='Optimized')

ax1_3d.set_xlabel('Weight 1 (radians)')
ax1_3d.set_ylabel('Weight 2 (radians)')
ax1_3d.set_zlabel('Loss')
ax1_3d.set_title('SamplerQNN Loss Landscape')
ax1_3d.legend()

# EstimatorQNN 3D surface
ax2_3d = fig_3d.add_subplot(122, projection='3d')
surf2 = ax2_3d.plot_surface(W1, W2, e_losses_3d, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)

# Mark important points
ax2_3d.scatter(initial_weight_values[0], initial_weight_values[1], e_loss_fn(initial_weight_values), 
               color='blue', s=100, label='Initial')
ax2_3d.scatter(new_e_weights[0], new_e_weights[1], e_loss_fn(new_e_weights), 
               color='orange', s=100, label='Optimized')

ax2_3d.set_xlabel('Weight 1 (radians)')
ax2_3d.set_ylabel('Weight 2 (radians)')
ax2_3d.set_zlabel('Loss')
ax2_3d.set_title('EstimatorQNN Loss Landscape')
ax2_3d.legend()

# Add colorbars
fig_3d.colorbar(surf1, ax=ax1_3d, shrink=0.5, aspect=20)
fig_3d.colorbar(surf2, ax=ax2_3d, shrink=0.5, aspect=20)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "v1.1landscape3d.png"))
plt.show()

## -------------- LOSS CONVERGENCE PLOTS -------------- ##
print("\n" + "="*50)
print("LOSS CONVERGENCE VISUALIZATION")
print("="*50)

# Create loss convergence plots
fig_loss, (ax1_loss, ax2_loss) = plt.subplots(1, 2, figsize=(16, 6))
fig_loss.suptitle('Training Loss Convergence - 2 Weight Parameters', fontsize=16, fontweight='bold')

# Plot SamplerQNN loss convergence
iterations_s = range(1, len(s_loss_history) + 1)
ax1_loss.plot(iterations_s, s_loss_history, 'r-', linewidth=2, label='SamplerQNN Loss', alpha=0.8)
ax1_loss.set_xlabel('Training Iteration', fontsize=12)
ax1_loss.set_ylabel('Loss Value', fontsize=12)
ax1_loss.set_title('SamplerQNN Loss Convergence', fontsize=14, fontweight='bold')
ax1_loss.grid(True, alpha=0.3)
ax1_loss.set_yscale('log')  # Use log scale for better visualization
ax1_loss.legend()

# Add final loss annotation
final_s_loss = s_loss_history[-1]
ax1_loss.annotate(f'Final Loss: {final_s_loss:.6f}', 
                 xy=(len(s_loss_history), final_s_loss), 
                 xytext=(len(s_loss_history)*0.7, final_s_loss*2),
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                 fontsize=10, fontweight='bold', color='red')

# Plot EstimatorQNN loss convergence
iterations_e = range(1, len(e_loss_history) + 1)
ax2_loss.plot(iterations_e, e_loss_history, 'orange', linewidth=2, label='EstimatorQNN Loss', alpha=0.8)
ax2_loss.set_xlabel('Training Iteration', fontsize=12)
ax2_loss.set_ylabel('Loss Value', fontsize=12)
ax2_loss.set_title('EstimatorQNN Loss Convergence', fontsize=14, fontweight='bold')
ax2_loss.grid(True, alpha=0.3)
ax2_loss.set_yscale('log')  # Use log scale for better visualization
ax2_loss.legend()

# Add final loss annotation
final_e_loss = e_loss_history[-1]
ax2_loss.annotate(f'Final Loss: {final_e_loss:.6f}', 
                 xy=(len(e_loss_history), final_e_loss), 
                 xytext=(len(e_loss_history)*0.7, final_e_loss*2),
                 arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                 fontsize=10, fontweight='bold', color='orange')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "v1.1convergence.png"))
plt.show()

# Print convergence statistics
print(f"SamplerQNN Convergence Statistics:")
print(f"  Initial Loss: {s_loss_history[0]:.8f}")
print(f"  Final Loss: {s_loss_history[-1]:.8f}")
print(f"  Loss Reduction: {(s_loss_history[0] - s_loss_history[-1])/s_loss_history[0]*100:.2f}%")
print(f"  Total Iterations: {len(s_loss_history)}")

print(f"\nEstimatorQNN Convergence Statistics:")
print(f"  Initial Loss: {e_loss_history[0]:.8f}")
print(f"  Final Loss: {e_loss_history[-1]:.8f}")
print(f"  Loss Reduction: {(e_loss_history[0] - e_loss_history[-1])/e_loss_history[0]*100:.2f}%")
print(f"  Total Iterations: {len(e_loss_history)}")

## -------------- OPTIMIZATION SUMMARY -------------- ##
print("\n" + "="*50)
print("OPTIMIZATION SUMMARY")
print("="*50)

# Display optimization summary
print(f"Initial weights: [{initial_weight_values[0]:.6f}, {initial_weight_values[1]:.6f}]")
print(f"Target: Initial state rotated by π/2 = {np.pi/2:.6f} radians")
print(f"")
print(f"SamplerQNN Results:")
print(f"  Optimized weights: [{new_s_weights[0]:.6f}, {new_s_weights[1]:.6f}]")
print(f"  Total rotation: {new_s_weights[0] + new_s_weights[1]:.6f} radians")
print(f"  Difference from π/2: {abs((new_s_weights[0] + new_s_weights[1]) - np.pi/2):.6f} radians")
print(f"  Final loss: {s_loss_fn(new_s_weights):.8f}")
print(f"  Optimization time: {s_optimization_time:.2f} seconds")
print(f"")
print(f"EstimatorQNN Results:")
print(f"  Optimized weights: [{new_e_weights[0]:.6f}, {new_e_weights[1]:.6f}]")
print(f"  Total rotation: {new_e_weights[0] + new_e_weights[1]:.6f} radians")
print(f"  Difference from π/2: {abs((new_e_weights[0] + new_e_weights[1]) - np.pi/2):.6f} radians")
print(f"  Final loss: {e_loss_fn(new_e_weights):.8f}")
print(f"  Optimization time: {e_optimization_time:.2f} seconds")