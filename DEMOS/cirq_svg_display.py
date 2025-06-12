import cirq
from IPython.display import SVG, display

# Define two qubits
q0 = cirq.LineQubit(0)
q1 = cirq.LineQubit(1)

# Create a circuit
circuit = cirq.Circuit(
    cirq.H(q0),        # Hadamard gate on qubit 0
    cirq.CNOT(q0, q1)  # CNOT gate with q0 as control and q1 as target
)

# Method 1: Using cirq's built-in SVG display
print("Method 1: Using cirq's built-in SVG display")
display(circuit)

# Method 2: Using IPython's SVG display
print("\nMethod 2: Using IPython's SVG display")
svg = circuit.to_svg()
display(SVG(svg))

# Method 3: Using cirq's HTML display
print("\nMethod 3: Using cirq's HTML display")
html = circuit.to_html()
display(html) 