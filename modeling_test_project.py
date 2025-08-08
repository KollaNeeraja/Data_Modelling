#  Imports
import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

#  Load Dataset
df = pd.read_csv("N:/Projects/Raj_Project/ModelingTestProject/Data Scientist_Model Test Data.csv")

#  Feature Selection (Top 15 from Random Forest)
X = df.drop(columns='y')
y = df['y']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = feature_importances.sort_values(ascending=False).head(15).index
X_top = X[top_features]

#  Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_top, y, test_size=0.2, random_state=42)

#  Define Base and Meta Models
base_models = [
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)),
    ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))
]
meta_model = LassoCV(alphas=[0.01, 0.1, 1.0], cv=5, max_iter=10000, n_jobs=-1)

#  Build Pipeline
stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=True
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', stacked_model)
])

#  Train Pipeline
pipeline.fit(X_train, y_train)

#  Evaluate Model
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
cv_r2 = cross_val_score(pipeline, X_top, y, cv=5, scoring='r2').mean()

print(f"Test R² Score: {r2:.4f}")
print(f"Cross-validated R²: {cv_r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")

#  SHAP for Interpretability (Optional for Dev/Debugging — not visual)
final_gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
final_gb_model.fit(X_top, y)

explainer = shap.Explainer(final_gb_model)
shap_values = explainer(X_top)

# Example of accessing SHAP values (can be logged or saved if needed)
shap_array = shap_values.values  # or shap_values[:, i].values for specific feature i

