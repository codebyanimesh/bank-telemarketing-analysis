# 🏦 Bank Telemarketing: Comparing Classifiers 

🔗 **[Click here to view the full Jupyter Notebook Analysis](notebooks/bank_telemarketing_analysis.ipynb)**

## 📖 Overview & Business Understanding
This repository contains a Jupyter Notebook analyzing a dataset of 17 direct marketing campaigns (phone calls) conducted by a Portuguese banking institution. 

**Business Objective:** The primary goal is to compare the performance of four machine learning classifiers—K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM)—to predict whether a client will subscribe to a bank term deposit. Accurately predicting this conversion likelihood allows the bank to optimize resource allocation, reduce telemarketing fatigue, and increase overall campaign ROI.

## 🧹 Data Preparation & Cleaning
* The dataset consists of 41,188 instances and 21 columns. 
* The `duration` column was dropped entirely prior to modeling to prevent data leakage, as call duration is not known before a call is made.
* Categorical variables were one-hot encoded, and continuous variables were scaled using `StandardScaler` to ensure distance-based algorithms (like KNN and SVM) functioned correctly.

## 📊 Summary of Findings & Actionable Insights

### 1. Model Comparisons & Metric Rationale
* **Baseline Challenge:** The dataset is heavily imbalanced, with roughly 88.7% of clients refusing the subscription. A "dummy" model predicting "no" for everyone achieves 88.7% accuracy. Therefore, **Accuracy is a misleading metric**.
* **Evaluation Metric:** We shifted our primary evaluation metric to **Recall**, as the business priority is to capture as many true "Yes" customers (actual subscribers) as possible without missing them.
* **Algorithm Performance:** SVM and Logistic Regression performed similarly in terms of accuracy (~90%), but SVM required significantly more computational time (over 2.5 minutes to train compared to Logistic Regression's < 1 second). Default Decision Trees severely overfit the training data.
* **Tuning Success:** By utilizing `GridSearchCV` to tune a Decision Tree model with `class_weight='balanced'`, we successfully optimized for Recall, capturing 62% of actual subscribers (compared to much lower baseline recall rates). 

### 2. Inferential Statistics & Key Drivers (Coefficients)
Extracting the coefficients from the Logistic Regression model and feature importances from the tuned Decision Tree revealed that macroeconomic indicators and timing are the strongest predictors of conversion:
* **Positive Drivers (Increases Likelihood to Subscribe):** High Consumer Price Index (`cons.price.idx`) and the Euribor 3-month rate (`euribor3m`). 
* **Negative Drivers (Decreases Likelihood to Subscribe):** A higher negative employment variation rate (`emp.var.rate`) and being contacted via telephone (`contact_telephone`) rather than cellular. 

### 3. Next Steps and Recommendations
1. **Target Timing:** Re-allocate telemarketing resources away from historically poor-performing months (like May) and concentrate efforts during higher-yield months like March, September, or October.
2. **Economic Agility:** Tie campaign aggressiveness to macroeconomic indicators like the Euribor rate and Consumer Price Index. As these fluctuate, marketing messaging and target demographics should pivot accordingly.
3. **Deploy the Tuned Model:** Implement the tuned Decision Tree as a preliminary lead-scoring filter. Directing telemarketing agents primarily to clients flagged as "Yes" by the model will drastically reduce call volume and overhead while capturing the majority of willing subscribers.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Logistic Regression, KNN, Decision Trees, SVM, GridSearchCV)
* **Data Visualization:** `matplotlib`, `seaborn`

## 📂 Repository Structure
```text
├── data/
│   └── bank-additional-full.csv   # Raw dataset from UCI Machine Learning Repository
├── notebooks/
│   └── bank_telemarketing_analysis.ipynb           # Main Jupyter Notebook containing the analysis and modeling
└── README.md                      # Project documentation