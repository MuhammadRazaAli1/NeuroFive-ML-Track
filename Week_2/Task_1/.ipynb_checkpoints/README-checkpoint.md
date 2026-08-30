# Titanic Survival Prediction - Classification Model

## Approach
For this task, I built a baseline machine learning classification model to predict passenger survival on the Titanic. 
1. **Data Preprocessing:** Cleaned missing values (Age, Embarked) and dropped irrelevant columns (Name, Ticket, Cabin, PassengerId).
2. **Encoding:** Converted categorical variables (`Sex`, `Embarked`) into numerical format using `pd.get_dummies()`, applying `drop_first=True` to prevent multicollinearity.
3. **Modeling:** Split the dataset into 80% training and 20% testing data. I then trained a `LogisticRegression` model, setting `max_iter=1000` to ensure proper convergence without warnings.
4. **Evaluation:** Evaluated the model using standard accuracy metrics and plotted a confusion matrix using Seaborn for better interpretability.

## Final Results
- **Model Used:** Logistic Regression
- **Accuracy:** ~80% (varies slightly based on exact split)

The confusion matrix indicates that the model is generally reliable but provides a clear baseline that could be improved in the future using more complex algorithms like Random Forests or by engineering new features (e.g., family size).