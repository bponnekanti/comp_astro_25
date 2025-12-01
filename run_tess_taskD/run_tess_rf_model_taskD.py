#%%
import sys
sys.path.append('..')

import numpy as np
from tess_rf_model import TESSRandomForest
from sklearn.metrics import precision_score, confusion_matrix


# Define the four combinations of n_estimators and max_depth
hyperparams_list = [
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 5},
    {'n_estimators': 200, 'max_depth': 10},
]

# Initialize the class
rf_pipeline = TESSRandomForest(csv_path='../tess_data.csv', samples_per_class=350)

report_lines = []

for i, params in enumerate(hyperparams_list, 1):
    print(f"\nRunning model {i}: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    y_pred, proba, threshold, _ = rf_pipeline.run(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
    )
    
    # Compute confusion matrix and precision
    cm = confusion_matrix(rf_pipeline.y_test, y_pred)
    precision = precision_score(rf_pipeline.y_test, y_pred)
    
    report_lines.append(f"Model {i} - n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")
    report_lines.append(f"Confusion Matrix:\n{cm}")
    report_lines.append(f"Precision: {precision:.4f}\n")
    
# Save report
with open('report_random_forest_assignment2_taskD.txt', 'w') as f:
    f.write("\n".join(report_lines))

print("\nTask D completed! Report saved as 'report_random_forest_assignment2_taskD.txt'.")
# %%
