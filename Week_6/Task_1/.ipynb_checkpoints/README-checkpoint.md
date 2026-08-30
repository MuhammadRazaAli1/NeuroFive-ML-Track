# 🚗 AutoPrice AI: Intelligent Vehicle Valuation Engine

## 📌 Problem Statement
Pricing a used vehicle accurately is a challenge due to market volatility, depreciation factors, and subjective condition assessments. This project aims to remove the guesswork by building an end-to-end machine learning regression system that predicts fair market value based on technical specifications and historical usage.

## 🛠️ Approach & Workflow
1. **Data Cleaning:** Processed a comprehensive vehicle dataset, strategically dropping high-cardinality text columns (like `description` and `title`) to optimize the baseline model.
2. **Feature Engineering:** Calculated `car_age` from the manufacturing year to provide a more intuitive linear feature for depreciation.
3. **ML Pipeline:** Built a robust Scikit-Learn `Pipeline` utilizing `StandardScaler` for continuous variables (odometer, engine_cc, hp) and `OneHotEncoder` for categorical features (make, fuel, body_type).
4. **Modeling:** Trained and evaluated **Random Forest** and **XGBoost** regressors, selecting the highest-performing model based on R² and RMSE scores.
5. **Deployment:** Saved the entire pipeline using `joblib` and deployed an interactive, real-time web application using **Streamlit**.

## 🚀 Live Application
Test the valuation engine here: **[Insert Your Streamlit App Link]**

## 💻 How to Run Locally
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `streamlit run app.py`