## Ensemble Models vs Baseline

In this phase, I upgraded the predictive modeling approach by transitioning from a single linear model (Logistic Regression) to powerful tree-based ensemble methods (Random Forest and XGBoost). 

### Performance Comparison Table

| Model | Metric Evaluated | Score (Accuracy / F1) |
| :--- | :--- | :--- |
| **Logistic Regression (Baseline)** | Accuracy & F1-Score | ~0.80 / 0.76 |
| **Random Forest Classifier** | Accuracy & F1-Score | ~0.82 / 0.77 |
| **XGBoost Classifier** | Accuracy & F1-Score | **~0.83 / 0.78** |

*Note: Exact scores may fluctuate slightly based on data splits, but Ensemble models consistently outperformed the baseline.*

**Key Takeaway:** XGBoost proved to be the most robust model by sequentially learning from the errors of previous trees, whereas Random Forest also showed a strong improvement over the baseline through parallel tree averaging (bagging).