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

**Decision Tree (shallow tuned):**
- Accuracy: 0.62
- Precision (class 0): 0.53
- Recall (class 0): 0.71
- F1-score (class 0): 0.61
- Precision (class 1): 0.73
- Recall (class 1): 0.55
- F1-score (class 1): 0.63

## Comparison and Explanation
- The shallow tuned Decision Tree outperformed Logistic Regression on test accuracy (0.62 vs 0.46) and balanced performance across classes.
- The Decision Tree benefits from capturing non-linear relationships and simple thresholds that align with lending decision rules (e.g., splits on `CreditScore`, `LoanAmount`, or `IncomePerLoan`).
- Logistic Regression underperforms likely due to linear decision boundaries and small dataset size.
- Caveats:
  - The dataset is small; results are sensitive to the specific train/test split.
  - Performance may vary across splits; cross-validated averages are advisable for robustness.

## Conclusion
- The Decision Tree (tuned, shallow depth) is the better model on this dataset, achieving ~0.62 accuracy with reasonable precision/recall trade-offs for both classes.
- Further steps could include collecting more data, engineering features (e.g., binning credit score, log-transforming skewed incomes), or trying ensembles (Random Forest/Gradient Boosting) if permitted by the assignment.

**Recommendation:**
- Report both models but emphasize the tuned Decision Tree as the preferred choice for this dataset due to higher accuracy and interpretable splits. 