# Outlook into the Future: Quantum Machine Learning
# Coding example 
# In this example, we generate a toy dataset with two features and binary labels. We then define a quantum feature map using the ZZFeatureMap from Qiskit's circuit library. We create a quantum circuit with two qubits and append the feature map to it. We measure all the qubits in the circuit.
# Next, we execute the quantum circuit on a simulator backend to obtain the measurement outcomes. These outcomes represent the quantum states' probabilities after the measurement.
# To make predictions for each data point, we initialize the circuit's input state with the data point and execute the circuit again. We extract the output statevector and calculate the probabilities of the positive class (1) and the negative class (-1). The prediction is made by comparing these probabilities.
# Finally, we evaluate the accuracy of the predictions and plot the histogram of the measurement outcomes.
# Here is a coding example for a simple quantum machine learning task using the Qiskit library in Python. This example demonstrates how to use a quantum circuit for a binary classification problem:
	import numpy as np
	from qiskit import Aer, execute, QuantumCircuit
	from qiskit.circuit.library import ZZFeatureMap
	from qiskit.visualization import plot_histogram
	
	# Generate a toy dataset
	np.random.seed(0)
	num_samples = 100
	features = np.random.rand(num_samples, 2)
	labels = np.array([1 if x[0] > x[1] else -1 for x in features])
	
	# Quantum feature map
	feature_map = ZZFeatureMap(2)
	
	# Quantum circuit
	qc = QuantumCircuit(2)
	qc.append(feature_map, [0, 1])
	qc.measure_all()
	
	# Run the quantum circuit
	backend = Aer.get_backend('qasm_simulator')
	job = execute(qc, backend, shots=1000)
	result = job.result()
	counts = result.get_counts(qc)
	
	# Predict the class labels
	predictions = np.zeros(num_samples)
	for i, (x, _) in enumerate(zip(features, labels)):
	    input_state = np.concatenate((x, np.zeros(2)))
	    qc.initialize(input_state, qc.qubits)
	    job = execute(qc, backend, shots=1000)
	    result = job.result()
	    output_state = result.get_statevector(qc)
	    prob_1 = np.abs(output_state[0]) ** 2
	    prob_minus_1 = np.abs(output_state[2]) ** 2
	    predictions[i] = 1 if prob_1 > prob_minus_1 else -1
	
	# Evaluate accuracy
	accuracy = np.sum(predictions == labels) / num_samples
	print("Accuracy:", accuracy)
	
	# Plot the histogram of measurement outcomes
	plot_histogram(counts)

# Exercise and solutions
# Exercise: Implementing a Simple Quantum Circuit for Quantum Machine Learning
# Objective: Understand the foundational elements of quantum machine learning by creating a basic quantum circuit using Qiskit, a popular framework for quantum computing.
# Task: Implement a quantum circuit that prepares a superposition of states and then measures the probabilities of each state. This will serve as an introduction to quantum systems' behaviour, which will be foundational in more complex quantum machine learning tasks.
# Steps:
# 1.	Install Qiskit:
	pip install qiskit
# 2.	Create a quantum circuit:
	import qiskit
	from qiskit import Aer, QuantumCircuit
	from qiskit.visualization import plot_histogram
	
	# Define a quantum circuit with 1 qubit and 1 classical bit
	qc = QuantumCircuit(1, 1)
	
	# Apply a Hadamard gate to the qubit to create a superposition of states
	qc.h(0)
	
	# Measure the qubit and store the result in the classical bit
	qc.measure(0, 0)
	
	# Visualize the circuit
	qc.draw('mpl')

# 3.	Simulate the quantum circuit:
	# Use Qiskit's Aer backend to simulate the circuit
	simulator = Aer.get_backend('qasm_simulator')
    result = qiskit.execute(qc, simulator, shots=1000).result()
	
	# Get measurement results
	counts = result.get_counts(qc)
	
    # Plot the results
	plot_histogram(counts)
 
# Expected Outcome: You should see a histogram with approximately equal probabilities (close to 50%) for both |0⟩ and |1⟩ states, showcasing the quantum superposition principle.
# Discussion: This simple quantum circuit demonstrates how quantum systems can exist in a superposition of states. When measured, they collapse to one of the basis states. This behaviour forms the foundation of quantum machine learning algorithms, which exploit superposition, entanglement, and quantum interference for computational advantages.
# Further exploration:
# •	Investigate more advanced quantum gates and their effects on qubits.
# •	Explore Qiskit's library for quantum machine learning to delve deeper into Quantum Neural Networks or Quantum SVM implementations.

