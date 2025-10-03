# Task 7: Support Vector Machine (SVM) Classification

This project is a submission for Task 7 of the Elevate Labs AI/ML Internship. The objective is to implement Support Vector Machines for both linear and non-linear classification, perform hyperparameter tuning, and visualize the results.

## Project Objective

The goal is to build and evaluate SVM classifiers on the Wisconsin Breast Cancer dataset. The project explores the difference between linear and RBF kernels, uses `GridSearchCV` for systematic hyperparameter tuning, and leverages **Principal Component Analysis (PCA)** to visualize the decision boundary in 2D.

## Dataset

The project uses the **Wisconsin Breast Cancer dataset**, which is conveniently available in Scikit-learn.
-   **Features**: 30 numeric, predictive attributes computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
-   **Target**: Diagnosis (Malignant = 0, Benign = 1).

## Methodology

1.  **Data Loading & Preprocessing**: The dataset is loaded, split into training (80%) and testing (20%) sets, and then scaled using `StandardScaler`, which is crucial for SVM's performance.
2.  **Model Benchmarking**: Three different SVM models are trained and evaluated:
    * **Model 1**: A basic SVM with a `linear` kernel.
    * **Model 2**: A basic SVM with a non-linear `rbf` kernel using default parameters.
    * **Model 3**: An optimized SVM where `GridSearchCV` is used to find the best hyperparameters (`C` and `gamma`) for the `rbf` kernel through 5-fold cross-validation.
3.  **Dimensionality Reduction & Visualization**: To visualize the decision boundary of our best model, PCA is applied to the 30-dimensional feature space to reduce it to 2 principal components. The final, tuned SVM is then re-trained on this 2D data to plot its classification boundary.

## Results & Analysis

### Model Performance Comparison

| Model | Kernel | Hyperparameters | Test Accuracy |
| :--- | :---: | :--- | :---: |
| Model 1 | `linear` | C=1.0 (default) | 95.61% |
| Model 2 | `rbf` | C=1.0, gamma='scale' (defaults) | 98.25% |
| **Model 3 (Tuned)** | **`rbf`** | **C=10, gamma=0.01 (via GridSearchCV)** | **99.12%** |

The results clearly show that the non-linear RBF kernel outperforms the linear kernel for this dataset. Furthermore, systematic hyperparameter tuning with `GridSearchCV` provided an additional performance boost, resulting in the highest accuracy of **99.12%**.

### Decision Boundary Visualization

The decision boundary of the best-tuned model, visualized on the two principal components of the data, is shown below. It effectively separates the 'Malignant' and 'Benign' classes.

![SVM Decision Boundary on PCA Data](visualizations/svm_decision_boundary_on_pca-reduced_data.png)

## Execution Instructions

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/YOUR_USERNAME/elevate-labs-task-07-svm-classification.git
    cd elevate-labs-task-07-svm-classification
    ```
2.  **Create a virtual environment and install dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
4.  Open `notebooks/svm_classification_analysis.ipynb` and run all cells.

---

## Interview Questions & Answers

#### 1. What is a support vector?
Support vectors are the data points from the training set that are closest to the decision boundary (or hyperplane). They are the most critical points because they are the only ones that influence the position and orientation of the hyperplane.

#### 2. What does the C parameter do?
The `C` parameter is the regularization parameter in SVM. It controls the trade-off between achieving a low training error and a low testing error.
* **Low `C`**: Creates a wider margin, allowing for more misclassifications (high bias, low variance). The model is more tolerant of errors.
* **High `C`**: Creates a narrower margin, trying to classify every training example correctly (low bias, high variance). This can lead to overfitting.

#### 3. What are kernels in SVM?
Kernels are functions that take low-dimensional input data and transform it into a higher-dimensional space. This "kernel trick" allows SVM to find a linear decision boundary in the higher-dimensional space, which corresponds to a complex, non-linear boundary in the original space.

#### 4. What is the difference between linear and RBF kernel?
* **Linear Kernel**: Creates a linear decision boundary (a hyperplane). It's computationally faster and works well when the data is already linearly separable.
* **RBF (Radial Basis Function) Kernel**: Creates a complex, non-linear boundary. It's more flexible and can handle intricate relationships in the data. It's controlled by the `gamma` hyperparameter, which defines how much influence a single training example has.

#### 5. What are the advantages of SVM?
* Effective in high-dimensional spaces.
* Memory efficient because it only uses a subset of training points (the support vectors).
* Versatile, as different Kernel functions can be specified for the decision function.

#### 6. Can SVMs be used for regression?
Yes. The technique is called Support Vector Regression (SVR). Instead of finding a hyperplane that maximizes the margin between classes, SVR finds a hyperplane that fits as many data points as possible within a certain margin (called the epsilon-tube).

#### 7. What happens when data is not linearly separable?
When data is not linearly separable, SVM uses two main techniques:
1.  **Soft Margin**: The `C` parameter allows some points to be misclassified or be inside the margin to find a boundary that generalizes better.
2.  **Kernel Trick**: A kernel function (like RBF) is used to project the data into a higher dimension where it becomes linearly separable.

#### 8. How is overfitting handled in SVM?
Overfitting is primarily handled by tuning the hyperparameters:
* **`C` parameter**: A smaller `C` value encourages a wider margin, making the model less sensitive to individual data points and reducing the risk of overfitting.
* **`gamma` parameter (for RBF kernel)**: A smaller `gamma` value creates a smoother, less complex decision boundary, which also helps prevent overfitting. Using cross-validation helps find the right balance for these parameters.
