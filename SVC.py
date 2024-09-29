from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Get iris dataset

iris = load_iris()

# Split to data and label
X = iris.data
y = iris.target

#Split to training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Classic SVC
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Here you can test other kernels for example 'rbf', 'sigmoid', 'poly'
# But for Iris linear is the simple and enough precise
svc_model = SVC(kernel='linear')
# Training model
svc_model.fit(X_train, y_train)

y_pred = svc_model.predict(X_test)
print(f"Classical SVC Accuracy: {accuracy_score(y_test, y_pred)}")

# Quantum SVC

# Less effective but you can try because it's to complicated
#######################################################################################################
# feature_map = PauliFeatureMap(feature_dimension=4, reps=2, paulis=['Z', 'ZZ', 'X', 'XX', 'Y', 'YY'])
#######################################################################################################

feature_map = ZZFeatureMap(feature_dimension=4, reps=2, entanglement="linear")

sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)

quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

qsvc_model = QSVC(quantum_kernel=quantum_kernel)
# Training quantum model
qsvc_model.fit(X_train, y_train)

# Evaluate the quantum model
y_pred_qsvc = qsvc_model.predict(X_test)

print(f"Quantum SVC Accuracy: {accuracy_score(y_test, y_pred_qsvc)}")

