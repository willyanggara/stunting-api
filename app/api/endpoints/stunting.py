import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from keras.api.applications.mobilenet_v2 import MobileNetV2
from keras.api.applications.mobilenet_v2 import preprocess_input
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.api.layers import Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout
from keras.api.models import Model, load_model
from keras.api.optimizers import Adam
from keras.api.preprocessing.image import load_img, img_to_array

from app.api import deps
from app.models import child as child_model
from app.models.stunting import ModelMetrics
from app.schemas import stunting as stunting_schema
from app.utils.prediction import predict_child_condition

# Constants
MODEL_DIR = "stunting-models"
MODEL_PATH = os.path.join(MODEL_DIR, "stunting_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
IMAGE_SIZE = (224, 224)  # reduce pixel image
BATCH_SIZE = 16
EPOCHS = 100
K_FOLD = 2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()


async def load_and_preprocess_images(children):
    x_front, x_back, x_left, x_right, y = [], [], [], [], []
    for child in children:
        if await validate_images(child):
            x_front.append(await preprocess_image(os.path.join("static", "child_images", child.image_front_name)))
            x_back.append(await preprocess_image(os.path.join("static", "child_images", child.image_back_name)))
            x_left.append(await preprocess_image(os.path.join("static", "child_images", child.image_left_name)))
            x_right.append(await preprocess_image(os.path.join("static", "child_images", child.image_right_name)))
            y.append([child.height, child.weight])
    return np.vstack(x_front), np.vstack(x_back), np.vstack(x_left), np.vstack(x_right), np.array(y)


async def create_and_compile_model():
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


async def train_model_with_data(model, x_train, y_train, x_val, y_val):
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


async def calculate_metrics(y_true, y_pred):
    mae_height = round(mean_absolute_error(y_true[:, 0], y_pred[:, 0]), 2)
    mae_weight = round(mean_absolute_error(y_true[:, 1], y_pred[:, 1]), 2)
    rmse_height = round(np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])), 2)
    rmse_weight = round(np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])), 2)
    return mae_height, mae_weight, rmse_height, rmse_weight


async def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


async def validate_images(child):
    required_images = ['image_front_name', 'image_back_name', 'image_left_name', 'image_right_name']
    for img in required_images:
        if not getattr(child, img):
            return False
        image_path = os.path.join("static", "child_images", getattr(child, img))
        if not os.path.exists(image_path):
            return False
    return True


async def train_model(db: AsyncSession):
    children = await db.execute(select(child_model.Child))
    children = children.scalars().all()
    x_front, x_back, x_left, x_right, y = await load_and_preprocess_images(children)

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

    model = await create_and_compile_model()
    model = await train_model_with_data(
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


@router.post("/train-model", response_model=stunting_schema.ModelResponse)
async def train_stunting_model(db: AsyncSession = Depends(deps.get_db)):
    try:
        success = await train_model(db)
        model_path, model_modified, scaler_path, scaler_modified = await check_model_scaler_existence(MODEL_PATH,
                                                                                                      SCALER_PATH)
        return {
            "message": "Model trained successfully and saved" if success else "Model training failed",
            "model_exists": model_path != "",
            "model_path": model_path,
            "model_modified": model_modified,
            "scaler_path": scaler_path,
            "scaler_modified": scaler_modified
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/check-model", response_model=stunting_schema.ModelResponse)
async def check_model_exists():
    model_path, model_modified, scaler_path, scaler_modified = await check_model_scaler_existence(MODEL_PATH, SCALER_PATH)
    return {
        "message": "Model Exist" if model_path != "" else "Model Not found",
        "model_exists": model_path != "",
        "model_path": model_path,
        "model_modified": model_modified,
        "scaler_path": scaler_path,
        "scaler_modified": scaler_modified
    }


async def check_model_scaler_existence(model_path: str, scaler_path: str):
    model_exists = await asyncio.to_thread(os.path.exists, model_path)
    scaler_exists = await asyncio.to_thread(os.path.exists, scaler_path)

    if model_exists and scaler_exists:
        model_modified = datetime.fromtimestamp(await asyncio.to_thread(os.path.getmtime, model_path)).strftime(
            "%d %B %Y, %H:%M")
        scaler_modified = datetime.fromtimestamp(await asyncio.to_thread(os.path.getmtime, scaler_path)).strftime(
            "%d %B %Y, %H:%M")
        return model_path, model_modified, scaler_path, scaler_modified

    return "", "", "", ""


@router.get("/evaluate-model", response_model=stunting_schema.ModelMetrics)
async def evaluate_stunting_model(db: AsyncSession = Depends(deps.get_db)):
    try:
        metrics = await evaluate_model(db)
        # Save metrics to the database
        db_metrics = ModelMetrics(**metrics)
        db.add(db_metrics)
        await db.commit()
        await db.refresh(db_metrics)

        metrics["created_at"] = metrics["created_at"].strftime("%d %B %Y, %H:%M")

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def evaluate_model(db: AsyncSession) -> dict[str, datetime | Any]:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    children = await db.execute(select(child_model.Child))
    children = children.scalars().all()
    x_front, x_back, x_left, x_right, y_true = await load_and_preprocess_images(children)

    if len(y_true) == 0:
        raise HTTPException(status_code=400, detail="No valid data found for evaluation")

    # Load the model and scaler
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Make predictions
    y_pred_scaled = model.predict([x_front, x_back, x_left, x_right])
    y_pred = scaler.inverse_transform(y_pred_scaled)

    # Calculate metrics
    mae_height, mae_weight, rmse_height, rmse_weight = await calculate_metrics(y_true, y_pred)

    logging.info(
        f"Model evaluation completed. MAE Height: {mae_height:.2f}, MAE Weight: {mae_weight:.2f}, RMSE Height: {rmse_height:.2f}, RMSE Weight: {rmse_weight:.2f}")

    return {
        "mae_height": mae_height,
        "mae_weight": mae_weight,
        "rmse_height": rmse_height,
        "rmse_weight": rmse_weight,
        "created_at": datetime.now()
    }


@router.get("/get-metrics", response_model=stunting_schema.ModelMetrics)
async def get_metrics(db: AsyncSession = Depends(deps.get_db)):
    metrics = await db.execute(select(ModelMetrics).order_by(ModelMetrics.created_at.desc()))
    metrics = metrics.scalars().first()
    if not metrics:
        raise HTTPException(status_code=404, detail="No metrics found")

    formatted_created_metric = metrics.created_at.strftime("%d %B %Y, %H:%M")
    return {
        "mae_height": metrics.mae_height,
        "mae_weight": metrics.mae_weight,
        "rmse_height": metrics.rmse_height,
        "rmse_weight": metrics.rmse_weight,
        "created_at": formatted_created_metric
    }


@router.post("/predict", response_model=stunting_schema.PredictionResponse)
async def predict_height_weight(child_id: int, db: AsyncSession = Depends(deps.get_db)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    child = await db.execute(select(child_model.Child).filter(child_model.Child.id == child_id))
    child = child.scalars().first()
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    prediction = await predict_child(child, model, scaler, db)
    if prediction is None:
        raise HTTPException(status_code=400, detail="Child is missing one or more required images")

    await db.commit()
    return prediction


@router.post("/predict-all", response_model=stunting_schema.SystemPerformanceResponse)
async def predict_all(db: AsyncSession = Depends(deps.get_db)):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise HTTPException(status_code=400, detail="Model or scaler not found. Please train the model first.")

    children = await db.execute(select(child_model.Child))
    children = children.scalars().all()
    if not children:
        raise HTTPException(status_code=404, detail="No children found in the database")

    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    predictions = []
    y_true = []
    y_pred = []
    for child in children:
        prediction = await predict_child(child, model, scaler, db)
        if prediction:
            predictions.append(prediction)
        if child.is_stunting is not None and prediction["predicted_stunting"] is not None:
            y_true.append(child.is_stunting)
            y_pred.append(prediction["predicted_stunting"])

    await db.commit()

    return await calculate_performance(y_true, y_pred)


async def predict_child(child: child_model.Child, model, scaler, db: AsyncSession) -> dict:
    if not await validate_images(child):
        raise HTTPException(status_code=500, detail=f"child with name {child.name} image not complete")

    x_front, x_back, x_left, x_right, _ = await load_and_preprocess_images([child])
    scaled_prediction = model.predict([x_front, x_back, x_left, x_right])
    prediction = scaler.inverse_transform(scaled_prediction)
    predicted_height, predicted_weight = prediction[0]

    # Add prediction logic
    child_condition = await predict_child_condition(db, child)
    predict_is_stunting = child_condition["is_stunting"]
    predict_is_wasting = child_condition["is_wasting"]
    predict_is_overweight = child_condition["is_overweight"]

    # Update the child record with predictions
    child.predict_height = round(float(predicted_height), 2)
    child.predict_weight = round(float(predicted_weight), 2)
    child.predict_stunting = predict_is_stunting
    child.predict_wasting = predict_is_wasting
    child.predict_overweight = predict_is_overweight

    return {
        "child_id": child.id,
        "actual_height": child.height,
        "actual_weight": child.weight,
        "actual_stunting": child.is_stunting,
        "predicted_height": child.predict_height,
        "predicted_weight": child.predict_weight,
        "predicted_stunting": child.predict_stunting,
        "predicted_wasting": child.predict_wasting,
        "predicted_overweight": child.predict_overweight,
    }


@router.get("/system-performance", response_model=stunting_schema.SystemPerformanceResponse)
async def calculate_system_performance(db: AsyncSession = Depends(deps.get_db)):
    children = await db.execute(select(child_model.Child))

    children = children.scalars().all()

    if not children:
        raise HTTPException(status_code=404, detail="No children found in the database")

    y_true = []
    y_pred = []

    for child in children:
        if child.is_stunting is not None and child.predict_stunting is not None:
            y_true.append(child.is_stunting)
            y_pred.append(child.predict_stunting)

    if not y_true or not y_pred:
        raise HTTPException(status_code=400, detail="No valid predictions found")

    return await calculate_performance(y_true, y_pred)


async def calculate_performance_metrics(y_true, y_pred):
    precision = round(await asyncio.to_thread(precision_score, y_true, y_pred), 2)
    recall = round(await asyncio.to_thread(recall_score, y_true, y_pred), 2)
    f1 = round(await asyncio.to_thread(f1_score, y_true, y_pred), 2)
    return precision, recall, f1


async def calculate_performance_confusion_matrix(y_true, y_pred):
    true_positives = false_positives = true_negatives = false_negatives = 0

    for t, p in zip(y_true, y_pred):
        if t and p:
            true_positives += 1
        elif not t and p:
            false_positives += 1
        elif not t and not p:
            true_negatives += 1
        elif t and not p:
            false_negatives += 1

    return true_positives, false_positives, true_negatives, false_negatives


async def calculate_performance(y_true, y_pred) -> stunting_schema.SystemPerformanceResponse:
    precision, recall, f1 = await calculate_performance_metrics(y_true, y_pred)
    true_positives, false_positives, true_negatives, false_negatives = await calculate_performance_confusion_matrix(
        y_true, y_pred)

    data = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "total_samples": len(y_true)
    }

    return stunting_schema.SystemPerformanceResponse(**data)


@router.post("/cross-validate", response_model=stunting_schema.CrossValidationResponse)
async def cross_validate_model(db: AsyncSession = Depends(deps.get_db)):
    # Fetching children asynchronously
    result = await db.execute(select(child_model.Child))
    children = result.scalars().all()

    x_front, x_back, x_left, x_right, y = load_and_preprocess_images(children)

    if len(y) == 0:
        raise HTTPException(status_code=400, detail="No valid data found for cross-validation")

    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y)

    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    mae_height_scores, mae_weight_scores, rmse_height_scores, rmse_weight_scores = [], [], [], []

    async def process_fold(fold, train_index, val_index):
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

        logging.info(
            f"Fold {fold} - MAE Height: {mae_height:.2f}, MAE Weight: {mae_weight:.2f}, RMSE Height: {rmse_height:.2f}, RMSE Weight: {rmse_weight:.2f}"
        )

    tasks = []
    for fold, (train_index, val_index) in enumerate(kf.split(x_front), 1):
        tasks.append(process_fold(fold, train_index, val_index))

    # Running the fold evaluations concurrently
    await asyncio.gather(*tasks)

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
