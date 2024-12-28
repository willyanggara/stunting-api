import tensorflow as tf
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api import deps
from app.models import child as child_model
from app.schemas import child as child_schema
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
import joblib
import math
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

MODEL_DIR = "stunting-models"
MODEL_PATH = os.path.join(MODEL_DIR, "stunting_model.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


def create_model():
    # Create four input layers for each image
    input_front = Input(shape=(224, 224, 3))
    input_back = Input(shape=(224, 224, 3))
    input_left = Input(shape=(224, 224, 3))
    input_right = Input(shape=(224, 224, 3))

    # Base model (MobileNetV2) for feature extraction
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Unfreeze the last few layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Process each input
    x_front = base_model(input_front)
    x_back = base_model(input_back)
    x_left = base_model(input_left)
    x_right = base_model(input_right)

    # Global average pooling
    x_front = GlobalAveragePooling2D()(x_front)
    x_back = GlobalAveragePooling2D()(x_back)
    x_left = GlobalAveragePooling2D()(x_left)
    x_right = GlobalAveragePooling2D()(x_right)

    # Concatenate features from all images
    x = Concatenate()([x_front, x_back, x_left, x_right])

    # Dense layers with dropout
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(2, activation='linear')(x)  # 2 outputs: height and weight

    # Create the model
    model = Model(inputs=[input_front, input_back, input_left, input_right], outputs=output)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')

    return model

def load_or_create_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        return create_model()


model = load_or_create_model()


def preprocess_image(image_path):
    img = load_img(image_path)
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

    X_front, X_back, X_left, X_right = [], [], [], []
    y = []

    for child in children:
        if validate_images(child):
            X_front.append(preprocess_image(os.path.join("static", "child_images", child.image_front_name)))
            X_back.append(preprocess_image(os.path.join("static", "child_images", child.image_back_name)))
            X_left.append(preprocess_image(os.path.join("static", "child_images", child.image_left_name)))
            X_right.append(preprocess_image(os.path.join("static", "child_images", child.image_right_name)))
            y.append([child.height, child.weight])

    if not X_front:
        raise HTTPException(status_code=400, detail="No valid data found for training")

    X_front = np.vstack(X_front)
    X_back = np.vstack(X_back)
    X_left = np.vstack(X_left)
    X_right = np.vstack(X_right)
    y = np.array(y)

    # Normalize the target values
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    # Split data
    X_train_front, X_test_front, X_train_back, X_test_back, X_train_left, X_test_left, X_train_right, X_test_right, y_train, y_test = train_test_split(
        X_front, X_back, X_left, X_right, y_scaled, test_size=0.2, random_state=42
    )

    logging.info(f"Starting model training with {len(X_front)} samples")

    # Callbacks
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(
        [X_train_front, X_train_back, X_train_left, X_train_right],
        y_train,
        epochs=100,
        batch_size=16,
        validation_data=([X_test_front, X_test_back, X_test_left, X_test_right], y_test),
        callbacks=[reduce_lr, early_stopping],
        verbose=1
    )

    logging.info("Model training completed")

    # Save the trained model and scaler
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)


@router.post("/train-model")
def train_stunting_model(db: Session = Depends(deps.get_db)):
    train_model(db)
    return {"message": "Model trained successfully and saved"}


@router.get("/check-model")
def check_model_exists():
    model_exists = os.path.exists(MODEL_PATH)
    scaler_exists = os.path.exists(SCALER_PATH)
    if model_exists and scaler_exists:
        model_modified = os.path.getmtime(MODEL_PATH)
        scaler_modified = os.path.getmtime(SCALER_PATH)
        return {
            "model_exists": True,
            "model_path": MODEL_PATH,
            "model_modified": datetime.fromtimestamp(model_modified).isoformat(),
            "scaler_path": SCALER_PATH,
            "scaler_modified": datetime.fromtimestamp(scaler_modified).isoformat()
        }
    else:
        return {"model_exists": False}


@router.post("/predict")
def predict_height_weight(child_id: int, db: Session = Depends(deps.get_db)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    child = db.query(child_model.Child).filter(child_model.Child.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    if not validate_images(child):
        raise HTTPException(status_code=400, detail="Child is missing one or more required images")

    # Preprocess the images
    img_front = preprocess_image(os.path.join("static", "child_images", child.image_front_name))
    img_back = preprocess_image(os.path.join("static", "child_images", child.image_back_name))
    img_left = preprocess_image(os.path.join("static", "child_images", child.image_left_name))
    img_right = preprocess_image(os.path.join("static", "child_images", child.image_right_name))

    # Make prediction
    scaled_prediction = model.predict([img_front, img_back, img_left, img_right])

    # Load the scaler and inverse transform the prediction
    scaler = joblib.load(SCALER_PATH)
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

@router.get("/evaluate-model")
def evaluate_stunting_model(db: Session = Depends(deps.get_db)):
    try:
        metrics = evaluate_model(db)
        return {
            "message": "Model evaluation completed successfully",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def evaluate_model(db: Session):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    children = db.query(child_model.Child).all()

    X_front, X_back, X_left, X_right = [], [], [], []
    y_true = []

    for child in children:
        if validate_images(child):
            X_front.append(preprocess_image(os.path.join("static", "child_images", child.image_front_name)))
            X_back.append(preprocess_image(os.path.join("static", "child_images", child.image_back_name)))
            X_left.append(preprocess_image(os.path.join("static", "child_images", child.image_left_name)))
            X_right.append(preprocess_image(os.path.join("static", "child_images", child.image_right_name)))
            y_true.append([child.height, child.weight])

    if not X_front:
        raise HTTPException(status_code=400, detail="No valid data found for evaluation")

    X_front = np.vstack(X_front)
    X_back = np.vstack(X_back)
    X_left = np.vstack(X_left)
    X_right = np.vstack(X_right)
    y_true = np.array(y_true)

    # Load the model and scaler
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Make predictions
    y_pred_scaled = model.predict([X_front, X_back, X_left, X_right])
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics
    mae_height = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_weight = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    rmse_height = math.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    rmse_weight = math.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))

    logging.info(f"Model evaluation completed. MAE Height: {mae_height:.2f}, MAE Weight: {mae_weight:.2f}, RMSE Height: {rmse_height:.2f}, RMSE Weight: {rmse_weight:.2f}")

    return {
        "mae_height": mae_height,
        "mae_weight": mae_weight,
        "rmse_height": rmse_height,
        "rmse_weight": rmse_weight
    }

