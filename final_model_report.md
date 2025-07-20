# Final Model Comparison and Report

## Model Performance Summary

**Logistic Regression:**
- Accuracy: 0.46
- Precision (class 0): 0.47
- Recall (class 0): 0.62
- F1-score (class 0): 0.53
- Precision (class 1): 0.44
- Recall (class 1): 0.31
- F1-score (class 1): 0.36

**Decision Tree:**
- Accuracy: 0.50
- Precision (class 0): 0.50
- Recall (class 0): 0.31
- F1-score (class 0): 0.38
- Precision (class 1): 0.50
- Recall (class 1): 0.69
- F1-score (class 1): 0.58

## Comparison and Explanation
- Both models performed similarly, with relatively low accuracy and F1-scores, indicating the dataset may be challenging for these models or may require more feature engineering or data.
- Logistic Regression had slightly better recall for class 0, while Decision Tree had better recall and F1-score for class 1 and slightly higher overall accuracy.
- Possible reasons for low performance:
  - The dataset is small, which can limit model learning.
  - There may be class imbalance or insufficient predictive features.
  - Some features may not be strongly correlated with the target.

## Conclusion
- Neither model performed particularly well, but Logistic Regression and Decision Tree both provide a baseline for future improvements.
- The Decision Tree performed slightly better in terms of overall accuracy and F1-score for the positive class (1).
- Further steps could include collecting more data, engineering new features, or trying more advanced models (e.g., ensemble methods).
- Improving data quality and exploring additional variables may help boost model performance.

**Recommendation:**
- Use these results as a starting point and iterate on feature engineering and model selection for better results in future work. 
