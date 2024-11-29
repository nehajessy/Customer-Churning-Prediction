# Customer Churn Prediction

This project demonstrates the process of predicting customer churn using various machine learning algorithms. The dataset used is **Churn Modeling.csv**, which contains information about customers, including demographics, account details, and whether they exited (churned).

## Project Overview

The main objective of this project is to predict whether a customer will exit the bank (`Exited = 1`) or not (`Exited = 0`). This involves:
- Data understanding and preprocessing.
- Data visualization for exploratory analysis.
- Training and evaluating multiple machine learning models.
- Hyperparameter tuning to improve model performance.

---

## Project Structure

- **Dataset**: 
  - The dataset `Churn Modeling.csv` contains 10,000 rows and 14 columns.
  - Target column: `Exited`.

- **Code**:
  - Data loading and preprocessing.
  - Exploratory Data Analysis (EDA).
  - Model training with various algorithms.
  - Hyperparameter tuning for the Random Forest Classifier.

---

## Steps Involved

### 1. **Data Understanding and Preprocessing**
- Inspected the dataset for missing values, data types, and unique values.
- Encoded categorical columns (`Geography` and `Gender`) using `LabelEncoder`.
- Dropped irrelevant columns such as `RowNumber`, `CustomerId`, and `Surname`.
- Split the dataset into features (`X`) and target (`Y`).
- Divided the data into training and testing sets (80:20 split).

### 2. **Data Visualization**
- Used Seaborn and Matplotlib for visualizing relationships between various features.
- Created bar plots and distribution plots for better insights into the dataset.

### 3. **Model Training and Evaluation**
- Implemented the following machine learning models:
  1. **Logistic Regression**
  2. **Naive Bayes**
  3. **Decision Tree Classifier**
  4. **Random Forest Classifier**
  5. **K-Nearest Neighbors (KNN)**
  6. **Support Vector Machine (SVM)**

- Metrics Used:
  - Accuracy
  - ROC-AUC Score
  - Confusion Matrix
  - Recall
  - Cohen’s Kappa Score

### 4. **Hyperparameter Tuning**
- Improved the Random Forest Classifier by adjusting parameters such as:
  - `n_estimators`: Number of trees in the forest.
  - `min_samples_split`: Minimum number of samples required to split an internal node.
  - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
  - `max_features`: Number of features considered for best split.
  - `max_samples`: Maximum number of samples for building trees.

---

## Requirements

The following libraries are required to run the code:

```bash
pandas
numpy
seaborn
matplotlib
scikit-learn
```

To install these dependencies, run:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/nehajessy/Customer-Churning-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd customer-churn-prediction
   ```
3. Place the `Churn Modeling.csv` file in the project directory.
4. Run the Python script:
   ```bash
   python customer_churning_prediction.py
   ```

---

## Results

Each model's performance is evaluated on the following metrics:
- **Accuracy**: Measures how often the model is correct.
- **ROC-AUC Score**: Measures the ability of the model to distinguish between classes.
- **Confusion Matrix**: Highlights the true positives, true negatives, false positives, and false negatives.
- **Cohen’s Kappa Score**: Evaluates the agreement between predictions and actual labels.

---

