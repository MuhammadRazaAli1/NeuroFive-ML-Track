## Handling Imbalanced Data: Customer Churn

### The Problem
Real-world business datasets are rarely balanced. In the Telco Churn dataset, the majority of customers stay (~73%), while a minority churn (~27%). Training a model directly on this data makes it biased towards predicting the majority class, resulting in poor identification of actual churners (Low Recall).

### The Solution (SMOTE)
1. **Baseline Model:** First, I trained a standard Logistic Regression model. It achieved good overall accuracy but struggled to accurately identify the churning customers.
2. **Data Balancing:** I applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively to the training data. This artificially balanced the dataset by generating synthetic examples of the minority class without causing data leakage.
3. **Results:** After retraining the model on the SMOTE-balanced data, the **Recall for Class 1 (Churners) improved significantly**. This means the business can now successfully identify and target far more at-risk customers before they leave, proving that optimizing for Recall/F1 is far more valuable than optimizing for raw accuracy.