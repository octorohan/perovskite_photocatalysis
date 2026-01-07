import joblib

pipe_liq = joblib.load("models/rf_liq_pipeline.joblib")

# Get expected input feature names BEFORE encoding
feature_cols = pipe_liq.feature_names_in_

joblib.dump(feature_cols, "models/feature_schema.joblib")

print("✅ Feature schema saved:", len(feature_cols), "columns")
