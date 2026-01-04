import pandas as pd

# Mapping numeric predictions to labels
label_map = {
    0: 'Low',
    1: 'Medium',
    2: 'High'
}

# Convert predictions
y_pred_labels = pd.Series(y_pred.flatten()).map(label_map) # Flatten y_pred

# Create prediction dataframe
pred_df = pd.DataFrame({
    "ID": x_test.index.values,   # Keep original IDs
    "Target": y_pred_labels.values
})

# Save to CSV
pred_df.to_csv('submission.csv', index=False)

pred_df.head()
