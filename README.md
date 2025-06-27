# Dimensionality Reduction: PCA with Logistic Regression vs SVM
This project compares the classification performance of **Logistic Regression** and **Support Vector Machine (SVM)** after applying **Principal Component Analysis (PCA)** on the Iris dataset. The goal is to explore how PCA affects model performance, separability, and confidence in predictions.

---

## Objective

- Apply **PCA** for dimensionality reduction.
- Train and compare **Logistic Regression** and **SVM** classifiers post-PCA.
- Visualize explained variance and 2D data projections.
- Plot decision boundaries in PCA space.
- Evaluate prediction metrics and model confidence.

---

## Dataset Details

- **Dataset**: Iris (from `sklearn.datasets`)
- **Features**: 4 (Sepal/Petal length & width)
- **Classes**: Setosa (0), Versicolor (1), Virginica (2)
- **Samples**: 150

---

## Steps & Methodology

### 1. Load & Explore Data
- Loaded via `load_iris()` from scikit-learn
- Converted to Pandas DataFrame

### 2. Apply PCA (2 Components)
- Standardized features using `StandardScaler`
- Applied PCA (`n_components=2`)
- Visualized 2D projection of classes

### 3. Train-Test Split
- 80/20 train-test split on PCA-transformed features

### 4. Model Training
- Trained both:
  - `LogisticRegression()`
  - `SVC(probability=True)`
- Models trained on PCA-transformed data

### 5. Decision Boundary Visualization
- Used meshgrid and `contourf` to show class separation
- Plotted for both models:
  - **Logistic Regression**
  - **SVM**

### 6. Model Evaluation
- Evaluated using:
  - Accuracy
  - Precision, Recall, F1-score (via `classification_report`)
- Logistic & SVM models both achieved **90% accuracy**

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 90%     |
| SVM                | 90%     |

#### Logistic Regression Report:
- High precision & recall for class 0 (Setosa)
- Slightly lower recall for class 1

#### SVM Report:
- Same performance as Logistic Regression in this case

### 7. Prediction Confidence Visualization
- Extracted predicted probabilities from both models
- Created **KDE plots** to visualize:
  - Confidence in predictions
  - Distribution of correct vs incorrect predictions

### 8. Realistic Sample Prediction
- Queried a new sample: `[5.1, 3.5, 1.4, 0.2]`
- Transformed through PCA pipeline
- Generated probability predictions from both models:
```text
Logistic Regression: [0.9807, 0.0192, 0.0000]
SVM:                 [0.9716, 0.0155, 0.0128]
```
---
## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/PCA_Logistic_vs_SVM

# Open Jupyter Notebook
cd PCA_Logistic_vs_SVM
jupyter notebook
