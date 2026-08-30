## Model Evaluation & Hyperparameter Tuning
In the second phase of this project, I evaluated the Logistic Regression model beyond simple accuracy. 
- Analyzed **Precision, Recall, and F1-Scores** using `classification_report` to better understand the model's performance on the minority class (Survivors).
- Implemented **GridSearchCV** to systematically tune hyperparameters (`C` and `solver`) while optimizing for the F1-score.
- **Results:** Tuning slightly improved the precision and F1-score, proving the importance of fine-tuning model parameters rather than relying on default settings.