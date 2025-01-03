import os
import math
import logging
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from app.api import deps
from app.models import child as child_model
from app.models.stunting import ModelMetrics
from app.schemas import stunting as stunting_schema

# Constants
MODEL_DIR = "stunting-models"
MODEL_PATH = os.path.join(MODEL_DIR, "stunting_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100
K_FOLD = 2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

def load_and_preprocess_images(children):
    x_front, x_back, x_left, x_right, y = [], [], [], [], []
    for child in children:
        if validate_images(child):
            x_front.append(preprocess_image(os.path.join("static", "child_images", child.image_front_name)))
            x_back.append(preprocess_image(os.path.join("static", "child_images", child.image_back_name)))
            x_left.append(preprocess_image(os.path.join("static", "child_images", child.image_left_name)))
            x_right.append(preprocess_image(os.path.join("static", "child_images", child.image_right_name)))
            y.append([child.height, child.weight])
    return np.vstack(x_front), np.vstack(x_back), np.vstack(x_left), np.vstack(x_right), np.array(y)

def create_and_compile_model():
    input_front = Input(shape=(*IMAGE_SIZE, 3))
    input_back = Input(shape=(*IMAGE_SIZE, 3))
    input_left = Input(shape=(*IMAGE_SIZE, 3))
    input_right = Input(shape=(*IMAGE_SIZE, 3))

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    def process_input(input_layer):
        x = base_model(input_layer)
        return GlobalAveragePooling2D()(x)

    x = Concatenate()([process_input(input) for input in [input_front, input_back, input_left, input_right]])
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='linear')(x)

    model = Model(inputs=[input_front, input_back, input_left, input_right], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
    return model

def train_model_with_data(model, x_train, y_train, x_val, y_val):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )
    return model

def calculate_metrics(y_true, y_pred):
    mae_height = round(mean_absolute_error(y_true[:, 0], y_pred[:, 0]), 2)
    mae_weight = round(mean_absolute_error(y_true[:, 1], y_pred[:, 1]), 2)
    rmse_height = round(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])), 2)
    rmse_weight = round(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])), 2)
    return mae_height, mae_weight, rmse_height, rmse_weight

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def validate_images(child):
    required_images = ['image_front_name', 'image_back_name', 'image_left_name', 'image_right_name']
    for img in required_images:
        if not getattr(child, img):
            return False
        image_path = os.path.join("static", "child_images", getattr(child, img))
        if not os.path.exists(image_path):
            return False
    return True

def train_model(db: Session):
    children = db.query(child_model.Child).all()
    x_front, x_back, x_left, x_right, y = load_and_preprocess_images(children)

    if len(y) == 0:
        raise HTTPException(status_code=400, detail="No valid data found for training")

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    # Split each input array separately
    x_train_front, x_val_front, \
    x_train_back, x_val_back, \
    x_train_left, x_val_left, \
    x_train_right, x_val_right, \
    y_train, y_val = train_test_split(
        x_front, x_back, x_left, x_right, y_scaled,
        test_size=0.2, random_state=42
    )

    model = create_and_compile_model()
    model = train_model_with_data(
        model,
        [x_train_front, x_train_back, x_train_left, x_train_right],
        y_train,
        [x_val_front, x_val_back, x_val_left, x_val_right],
        y_val
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return True

@router.post("/train-model", response_model=stunting_schema.TrainModelResponse)
def train_stunting_model(db: Session = Depends(deps.get_db)):
    try:
        success = train_model(db)
        return {
            "value": success,
            "message": "Model trained successfully and saved" if success else "Model training failed"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check-model", response_model=stunting_schema.CheckModelResponse)
def check_model_exists():
    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)
    if model_exists and scaler_exists:
        model_modified = os.path.getmtime(MODEL_PATH)
        scaler_modified = os.path.getmtime(SCALER_PATH)
        return {
            "model_exists": True,
            "model_path": MODEL_PATH,
            "model_modified": datetime.fromtimestamp(model_modified),
            "scaler_path": SCALER_PATH,
            "scaler_modified": datetime.fromtimestamp(scaler_modified)
        }
    else:
        return {"model_exists": False}

@router.get("/evaluate-model", response_model=stunting_schema.EvaluationResponse)
def evaluate_stunting_model(db: Session = Depends(deps.get_db)):
    try:
        metrics = evaluate_model(db)
        # Save metrics to the database
        db_metrics = ModelMetrics(**metrics)
        db.add(db_metrics)
        db.commit()
        db.refresh(db_metrics)

        return {
            "value": True,
            "message": "Model evaluation completed successfully and metrics saved to database",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def evaluate_model(db: Session):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    children = db.query(child_model.Child).all()
    x_front, x_back, x_left, x_right, y_true = load_and_preprocess_images(children)

    if len(y_true) == 0:
        raise HTTPException(status_code=400, detail="No valid data found for evaluation")

    # Load the model and scaler
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Make predictions
    y_pred_scaled = model.predict([x_front, x_back, x_left, x_right])
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics
    mae_height, mae_weight, rmse_height, rmse_weight = calculate_metrics(y_true, y_pred)

    logging.info(f"Model evaluation completed. MAE Height: {mae_height:.2f}, MAE Weight: {mae_weight:.2f}, RMSE Height: {rmse_height:.2f}, RMSE Weight: {rmse_weight:.2f}")

    return {
        "mae_height": mae_height,
        "mae_weight": mae_weight,
        "rmse_height": rmse_height,
        "rmse_weight": rmse_weight,
        "created_at": datetime.now()
    }

@router.get("/get-metrics", response_model=stunting_schema.ModelMetrics)
def get_metrics(db: Session = Depends(deps.get_db)):
    metrics = db.query(ModelMetrics).order_by(ModelMetrics.created_at.desc()).first()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found")
    return {
        "mae_height": metrics.mae_height,
        "mae_weight": metrics.mae_weight,
        "rmse_height": metrics.rmse_height,
        "rmse_weight": metrics.rmse_weight,
        "created_at": metrics.created_at
    }

@router.post("/predict", response_model=stunting_schema.PredictionResponse)
def predict_height_weight(child_id: int, db: Session = Depends(deps.get_db)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    if not validate_images(child):
        raise HTTPException(status_code=400, detail="Child is missing one or more required images")

    x_front, x_back, x_left, x_right, _ = load_and_preprocess_images([child])

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    scaled_prediction = model.predict([x_front, x_back, x_left, x_right])
    prediction = scaler.inverse_transform(scaled_prediction)

    predicted_height, predicted_weight = prediction[0]

    logging.info(f"Prediction made for child ID {child_id}: Height: {predicted_height:.2f}, Weight: {predicted_weight:.2f}")

    return {
        "child_id": child.id,
        "actual_height": child.height,
        "actual_weight": child.weight,
        "predicted_height": float(predicted_height),
        "predicted_weight": float(predicted_weight)
    }

@router.post("/cross-validate", response_model=stunting_schema.CrossValidationResponse)
def cross_validate_model(db: Session = Depends(deps.get_db)):
    children = db.query(child_model.Child).all()
    x_front, x_back, x_left, x_right, y = load_and_preprocess_images(children)

    if len(y) == 0:
        raise HTTPException(status_code=400, detail="No valid data found for cross-validation")

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    mae_height_scores, mae_weight_scores, rmse_height_scores, rmse_weight_scores = [], [], [], []

    for fold, (train_index, val_index) in enumerate(kf.split(x_front), 1):
        logging.info(f"Cross-validating fold {fold}/{K_FOLD}")

        x_train = [x[train_index] for x in [x_front, x_back, x_left, x_right]]
        x_val = [x[val_index] for x in [x_front, x_back, x_left, x_right]]
        y_train, y_val = y_scaled[train_index], y_scaled[val_index]

        model = create_and_compile_model()
        model = train_model_with_data(model, x_train, y_train, x_val, y_val)

        y_pred_scaled = model.predict(x_val)
        y_pred = scaler.inverse_transform(y_pred_scaled)
        y_true = scaler.inverse_transform(y_val)

        mae_height, mae_weight, rmse_height, rmse_weight = calculate_metrics(y_true, y_pred)

        mae_height_scores.append(mae_height)
        mae_weight_scores.append(mae_weight)
        rmse_height_scores.append(rmse_height)
        rmse_weight_scores.append(rmse_weight)

        logging.info(f"Fold {fold} - MAE Height: {mae_height:.2f}, MAE Weight: {mae_weight:.2f}, RMSE Height: {rmse_height:.2f}, RMSE Weight: {rmse_weight:.2f}")

    mean_mae_height, mean_mae_weight = np.mean(mae_height_scores), np.mean(mae_weight_scores)
    mean_rmse_height, mean_rmse_weight = np.mean(rmse_height_scores), np.mean(rmse_weight_scores)

    logging.info(f"Mean MAE Height: {mean_mae_height:.2f}, Mean MAE Weight: {mean_mae_weight:.2f}")
    logging.info(f"Mean RMSE Height: {mean_rmse_height:.2f}, Mean RMSE Weight: {mean_rmse_weight:.2f}")

    return {
        "k_fold": K_FOLD,
        "mae_height_scores": mae_height_scores,
        "mae_weight_scores": mae_weight_scores,
        "rmse_height_scores": rmse_height_scores,
        "rmse_weight_scores": rmse_weight_scores,
        "mean_mae_height": mean_mae_height,
        "mean_mae_weight": mean_mae_weight,
        "mean_rmse_height": mean_rmse_height,
        "mean_rmse_weight": mean_rmse_weight
    }

