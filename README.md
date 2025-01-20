# Prediction-of-down-syndrome-in-mice

# Prediction of Down Syndrome in Mice

## Overview
This project involves predicting Down syndrome in mice using a dataset containing biological markers. It employs various machine learning techniques such as Random Forest and Support Vector Machines (SVM). The data is preprocessed using advanced techniques like KNN imputation and feature selection to ensure high model accuracy. This README provides an overview of the steps involved, the methodology, and the results obtained.

---

## Dataset
The dataset contains biological markers extracted from mice. Key attributes include:
- A range of features representing biological markers.
- The target column indicating the presence or absence of Down syndrome.

---

## Steps

### 1. Data Preprocessing
- **Loading Data**: The dataset is read using `pandas`.
- **Handling Missing Values**: Columns with over 100 missing values are removed, and remaining missing values are imputed using KNN imputer.
- **Variance Inflation Factor (VIF)**: Features with high multicollinearity were identified and removed to improve model performance.

### 2. Feature Selection
- **Recursive Feature Elimination with Cross-Validation (RFECV)**: Used to select the most significant features for the models.

### 3. Data Splitting
- The data was shuffled and split into training-validation and test sets:
  - 90% for training-validation.
  - 10% for testing.
- Further train-validation split (80:20) was performed.

### 4. Model Training and Evaluation
#### Random Forest Classifier
- Hyperparameters:
  - `n_estimators`: [100, 300, 500, 700]
  - `max_depth`: [1, 3, 6, 8]
- The best model achieved maximum accuracy on the validation set.

#### Support Vector Machine (SVM)
- Preprocessed data using `StandardScaler`.
- Parameters:
  - `C`: [0.1, 1, 5]
  - `gamma`: [0.1, 1, 10]
- The best hyperparameters were used to predict on the test set.

### 5. Final Model
After feature selection, models were retrained, and the performance was evaluated on the test set for both Random Forest and SVM.

---

## Results
- **Random Forest Classifier**:
  - Best accuracy: ~85%
  - Final test accuracy: ~82%

- **SVM**:
  - Best accuracy: ~87%
  - Final test accuracy: ~85%

---

## Libraries Used
- **Core Libraries**: `numpy`, `pandas`
- **Visualization**: `seaborn`, `matplotlib`
- **Machine Learning**:
  - `sklearn.ensemble.RandomForestClassifier`
  - `sklearn.svm.SVC`
  - `sklearn.feature_selection.RFECV`
  - `sklearn.preprocessing.StandardScaler`
  - `sklearn.metrics`
  - `statsmodels` for VIF

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python down_syndrome_prediction.py
   ```
4. View results and accuracy metrics in the console.

---

## Future Enhancements
- Experimenting with advanced feature selection techniques.
- Adding more machine learning models for comparison.
- Hyperparameter optimization using GridSearchCV or RandomizedSearchCV.

---

## License
This project is licensed under the MIT License.

---

