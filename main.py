import pandas as pd
import numpy as np
import io
import uvicorn
import warnings

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

warnings.filterwarnings('ignore')

# Neural Network Builder
def create_nn_model(input_shape, output_units=1, activation='sigmoid', loss='binary_crossentropy'):
    """Creates the NN model."""
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dense(output_units, activation=activation)
    ])
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# Initialize FastAPI app
app = FastAPI()
app_storage = {} 

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Upload Endpoint
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Reads the CSV file, tries different encodings, and stores it."""
    try:
        contents = await file.read()
        for encoding in ['utf-8', 'iso-8859-1', 'windows-1256']:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                app_storage['dataframe'] = df
                return {"columns": list(df.columns)}
            except UnicodeDecodeError:
                continue
        raise HTTPException(status_code=400, detail="Failed to decode file. Check encoding (e.g., utf-8, iso-8859-1).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {e}")

# 2. Training Endpoint
@app.post("/train")
async def train_models(target_column: str = Form(...)):
    """Handles all preprocessing, validation, and model training."""
    if 'dataframe' not in app_storage:
        raise HTTPException(status_code=400, detail="CSV file not found. Please upload first.")

    df = app_storage['dataframe'].copy()

    # --- Target Column Validation ---
    if target_column not in df.columns:
        raise HTTPException(status_code=400, detail="Target column not found.")

    # 1. Clean target column
    df.dropna(subset=[target_column], inplace=True)
    num_unique_targets = df[target_column].nunique()
    if num_unique_targets < 2:
        raise HTTPException(status_code=400, detail="Target must have at least 2 unique values for classification.")
    
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[target_column])
    X = df.drop(target_column, axis=1)

    app_storage['target_le'] = target_le
    app_storage['target_column'] = target_column

    # Start Preprocessing Rules
    
    # 1. Drop high-cardinality/ID columns
    cols_to_drop = []
    total_rows = len(X)
    for col in X.columns:
        unique_count = X[col].nunique()
        if unique_count == total_rows:
            cols_to_drop.append(col)
        elif X[col].dtype == 'object' and (unique_count / total_rows) > 0.25:
            cols_to_drop.append(col)
    X = X.drop(columns=cols_to_drop)

    # 2. Handle Missing Values
    missing_info = X.isnull().sum()
    cols_with_missing = missing_info[missing_info > 0].index
    cols_to_drop_due_to_missing_cat = []
    rows_to_drop_due_to_missing = pd.Index([])
    
    for col in cols_with_missing:
        if col not in X.columns: continue
        missing_ratio = X[col].isnull().sum() / total_rows
        
        if 0 < missing_ratio < 0.02: # Rule 1: Drop rows with < 2% missing
            rows_to_drop_due_to_missing = rows_to_drop_due_to_missing.union(X[X[col].isnull()].index)
        elif X[col].dtype == 'object': # Rule 2: Drop categorical cols with missing
            cols_to_drop_due_to_missing_cat.append(col)

    X = X.drop(index=rows_to_drop_due_to_missing)
    y = y[X.index] # Sync y with X
    X = X.drop(columns=cols_to_drop_due_to_missing_cat)
    
    cols_to_drop.extend(cols_to_drop_due_to_missing_cat)
    app_storage['dropped_columns'] = list(set(cols_to_drop))

    # 3. Identify final feature types
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    app_storage['original_numerical_cols'] = numerical_cols
    app_storage['original_categorical_cols'] = categorical_cols
    app_storage['original_feature_order'] = list(X.columns)
    
    # 4. Split data (train/test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 5. Impute remaining missing values
    # Use 'mean' for numeric, 'most_frequent' for categorical
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    X_train_imputed = X_train.copy()
    X_test_imputed = X_test.copy()
    
    if numerical_cols:
        X_train_imputed[numerical_cols] = numeric_imputer.fit_transform(X_train[numerical_cols])
        X_test_imputed[numerical_cols] = numeric_imputer.transform(X_test[numerical_cols])
    if categorical_cols:
        X_train_imputed[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
        X_test_imputed[categorical_cols] = categorical_imputer.transform(X_test[categorical_cols])

    app_storage['numeric_imputer'] = numeric_imputer if numerical_cols else None
    app_storage['categorical_imputer'] = categorical_imputer if categorical_cols else None

    # 6. Remove outliers
    outliers_removed_count = 0
    if numerical_cols:
        scaler_for_outliers = StandardScaler()
        X_train_scaled_temp = scaler_for_outliers.fit_transform(X_train_imputed[numerical_cols])
        
        outlier_mask = (np.abs(X_train_scaled_temp) > 3).any(axis=1)
        
        # Convert numpy.int64 to standard int for JSON serialization
        outliers_removed_count = int(outlier_mask.sum()) 
        
        X_train_clean = X_train_imputed[~outlier_mask]
        y_train_clean = y_train[~outlier_mask]
    else:
        X_train_clean = X_train_imputed
        y_train_clean = y_train

    app_storage['outliers_removed'] = outliers_removed_count

    # 7. Create final preprocessor pipeline (Scaling + Encoding)
    final_numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    final_categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', final_numeric_transformer, numerical_cols),
            ('cat', final_categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    # 8. Fit and transform data
    preprocessor_fitted = preprocessor.fit(X_train_clean)
    X_train_processed = preprocessor_fitted.transform(X_train_clean)
    X_test_processed = preprocessor_fitted.transform(X_test_imputed) 
    
    n_features_processed = X_train_processed.shape[1] 
    app_storage['preprocessor'] = preprocessor_fitted

    # 9. Set dynamic model parameters
    class_balance = (y_train_clean == 0).sum() / (y_train_clean == 1).sum() if (y_train_clean == 1).sum() > 0 else 1
    
    # Configure NN for binary or multiclass
    nn_output_units = 1
    nn_activation = 'sigmoid'
    nn_loss = 'binary_crossentropy'
    y_train_nn = y_train_clean
    y_test_nn = y_test
    
    if num_unique_targets > 2:
        nn_output_units = num_unique_targets
        nn_activation = 'softmax'
        nn_loss = 'categorical_crossentropy'
        y_train_nn = to_categorical(y_train_clean, num_classes=num_unique_targets)
        y_test_nn = to_categorical(y_test, num_classes=num_unique_targets)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=max(100, min(300, len(X_train_clean) // 100)), class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(scale_pos_weight=class_balance, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss'),
        "Neural Network (MLP)": create_nn_model(n_features_processed, nn_output_units, nn_activation, nn_loss)
    }
    
    accuracies = {}
    trained_models_storage = {}

    # 10. Train models
    for name, model in models.items():
        if name == "Neural Network (MLP)":
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
            model.fit(
                X_train_processed, y_train_nn,
                epochs=100,
                batch_size=32,
                validation_data=(X_test_processed, y_test_nn),
                callbacks=[early_stop],
                verbose=0
            )
            y_pred_prob = model.predict(X_test_processed)
            if nn_output_units == 1:
                y_pred = (y_pred_prob > 0.5).astype("int32")
            else:
                y_pred = np.argmax(y_pred_prob, axis=1)
            trained_models_storage[name] = model
        else:
            model.fit(X_train_processed, y_train_clean)
            y_pred = model.predict(X_test_processed)
            trained_models_storage[name] = model

        acc = accuracy_score(y_test, y_pred) 
        accuracies[name] = f"{acc * 100:.2f}%"
        
    app_storage['models'] = trained_models_storage

    # 11. Prepare feature info for prediction form
    prediction_features = {}
    for col in numerical_cols:
        prediction_features[col] = {'type': 'number', 'values': None}
    for col in categorical_cols:
        unique_vals = [str(val) for val in X[col].unique() if pd.notna(val)]
        prediction_features[col] = {'type': 'dropdown', 'values': unique_vals}

    # Return results
    return {
        "accuracies": accuracies,
        "prediction_features": prediction_features,
        "dropped_columns": app_storage.get('dropped_columns', []),
        "outliers_removed": outliers_removed_count
    }

# 3. Prediction Endpoint
@app.post("/predict")
async def predict_row(request: Request):
    """Predicts the outcome for a single row of data."""
    try:
        data = await request.json()
        raw_input_data = data.get('features')

        # Load saved objects
        numeric_imputer = app_storage.get('numeric_imputer')
        categorical_imputer = app_storage.get('categorical_imputer')
        preprocessor = app_storage.get('preprocessor')
        models = app_storage.get('models')
        target_le = app_storage.get('target_le')
        
        numerical_cols = app_storage.get('original_numerical_cols', [])
        categorical_cols = app_storage.get('original_categorical_cols', [])
        original_cols = app_storage.get('original_feature_order', [])

        df_row = pd.DataFrame([raw_input_data])
        
        if original_cols:
            df_row = df_row[original_cols] # Ensure column order

        # 1. Impute input data
        if numeric_imputer and numerical_cols:
            # Ensure numeric types before imputation
            for col in numerical_cols:
                df_row[col] = pd.to_numeric(df_row[col], errors='coerce')
            df_row[numerical_cols] = numeric_imputer.transform(df_row[numerical_cols])
        if categorical_imputer and categorical_cols:
            df_row[categorical_cols] = categorical_imputer.transform(df_row[categorical_cols])

        # 2. Preprocess input (Scaling + Encoding)
        X_processed_row = preprocessor.transform(df_row)
        
        # 3. Make predictions
        predictions = {}
        for name, model in models.items():
            if name == "Neural Network (MLP)":
                pred_prob = model.predict(X_processed_row)
                if pred_prob.shape[1] == 1: # Binary
                    pred_numeric = (pred_prob > 0.5).astype("int32")
                else: # Multiclass
                    pred_numeric = np.argmax(pred_prob, axis=1)
            else:
                pred_numeric = model.predict(X_processed_row)
            
            # Convert numeric prediction back to original label
            pred_label = target_le.inverse_transform(pred_numeric.flatten())
            predictions[name] = str(pred_label[0])

        return {"predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e} | Input: {raw_input_data}")

# 4. Mount static frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# 5. Run server
if __name__ == "__main__":
    print("--- Required libraries: fastapi, uvicorn, scikit-learn, pandas, python-multipart, xgboost, tensorflow ---")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)