from sklearn.model_selection import train_test_split
x = df_clean_fe.drop('Target', axis=1)
y = df_clean_fe['Target']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

!pip install catboost
from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=500,           # number of boosting iterations
    learning_rate=0.1,        # step size shrinkage
    depth=6,                  # max depth of trees
    loss_function='MultiClass',  # multi-class classification
    random_seed=42,
    verbose=100               # prints progress
)

# Train the model
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")
