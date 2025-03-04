# Comparative Analysis of Classical and Quantum Support Vector Machines on the Iris Dataset

This project, developed during the HackYeah IBM Challenge, investigates the performance differences between classical and quantum Support Vector Machines (SVMs) using the Iris dataset. The classical SVM is implemented with scikit-learn, while the quantum SVM utilizes Qiskit's machine learning module.

## Project Overview

The primary objective is to compare the accuracy and feasibility of classical and quantum SVMs in classifying the Iris dataset, which consists of three iris species: Setosa, Versicolour, and Virginica. Each species is represented by 50 samples with four features: sepal length, sepal width, petal length, and petal width.

## Methodology

1. **Data Preparation**:
   - Load the Iris dataset using scikit-learn's `load_iris` function.
   - Split the dataset into training and testing sets with an 80-20 ratio.
   - Standardize the features using `StandardScaler` to ensure zero mean and unit variance.

2. **Classical SVM Implementation**:
   - Utilize scikit-learn's `SVC` with a linear kernel.
   - Train the model on the standardized training data.
   - Predict the labels for the test data and calculate the accuracy.

3. **Quantum SVM Implementation**:
   - Employ Qiskit's `ZZFeatureMap` for feature mapping with 4 features and 2 repetitions.
   - Use `ComputeUncompute` fidelity with a `Sampler` primitive to compute the quantum kernel.
   - Integrate the quantum kernel into Qiskit's `QSVC` classifier.
   - Train the quantum SVM on the training data, predict test labels, and compute the accuracy.

## Results

The classical SVM with a linear kernel achieved an accuracy of 1.0 (100%), while the quantum SVM achieved an accuracy of 0.9667 (96.67%) on the test dataset. This indicates that the classical SVM performed slightly better than the quantum SVM for this specific dataset.

## Conclusion

The higher accuracy of the classical SVM suggests that, for the Iris dataset, a linear kernel is sufficient to capture the underlying patterns. The quantum SVM, while slightly less accurate, demonstrates the potential of quantum machine learning models. Future work could involve experimenting with different quantum feature maps or applying these models to more complex datasets where quantum advantages might be more pronounced.

## References

- [Training a Quantum Model on a Real Dataset](https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html)
- [Plot different SVM classifiers in the iris dataset](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html)

## Acknowledgments

Special thanks to the HackYeah IBM Challenge organizers and the Qiskit community for their support and resources. 
