"""
Web UI on http://127.0.0.1:5000
Home: project title, AICW, partners, team, guide
Analyze: independent disease and quality prediction using uploaded or camera images
"""
import io
import os

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import pandas as pd

from advice import build_advice
from ap_locations import get_all_districts
from market_data import get_market_data
from model_utils import (
    CSV_DISEASE_LABELS,
    CSV_QUALITY_LABELS,
    get_or_load_all_models,
    lookup_dataset_labels,
    predict_disease,
    predict_quality,
    verify_cauliflower_image,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="CauliVision")
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

models = get_or_load_all_models()


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "home.html",
        {
            "page": "home",
            "disease_labels": CSV_DISEASE_LABELS,
            "quality_labels": CSV_QUALITY_LABELS,
        },
    )


@app.get("/analyze", response_class=HTMLResponse)
def analyze(request: Request):
    return templates.TemplateResponse(
        request,
        "analyze.html",
        {
            "page": "analyze",
            "disease_labels": CSV_DISEASE_LABELS,
            "quality_labels": CSV_QUALITY_LABELS,
        },
    )


@app.get("/market", response_class=HTMLResponse)
def market_prices(request: Request):
    districts = get_all_districts()
    default_district = districts[0] if districts else "Annamayya"
    return templates.TemplateResponse(
        request,
        "market.html",
        {
            "page": "market",
            "districts": districts,
            "default_district": default_district,
        },
    )


@app.get("/api/market-data")
def market_data(district: str | None = None):
    return get_market_data(district)


@app.get("/api/predict-market-price")
def predict_market_price(
    district: str,
    mandi_name: str,
    quantity_kg: int = Query(gt=0, le=100_000),
):
    if models.get("price_model") is None:
        raise HTTPException(status_code=503, detail="Market price prediction model is unavailable.")

    market_response = get_market_data(district)
    selected_market = next(
        (
            market
            for market in market_response["markets"]
            if market["mandi_name"].casefold() == mandi_name.casefold()
        ),
        None,
    )
    if selected_market is None:
        raise HTTPException(status_code=404, detail="Market was not found for this district.")

    features = pd.DataFrame(
        [[
            quantity_kg,
            float(selected_market["modal_price"]),
            float(selected_market["min_price"]),
            float(selected_market["max_price"]),
        ]],
        columns=[
            "quantity_sold_kg",
            "mandi_modal_rs_per_kg",
            "mandi_min_rs_per_kg",
            "mandi_max_rs_per_kg",
        ],
    )
    predicted_price = float(models["price_model"].predict(features)[0])
    return {
        "ok": True,
        "source": "project_price_model",
        "input_source": market_response["source"],
        "district": selected_market["district"],
        "mandi_name": selected_market["mandi_name"],
        "quantity_kg": quantity_kg,
        "predicted_price_rs_per_kg": round(predicted_price, 2),
        "estimated_total_rs": round(predicted_price * quantity_kg, 2),
        "model_dataset": "04_cauliflower_price_dataset.csv",
        "warning": (
            "Estimate from the project price dataset and model; not a live quote "
            "or guaranteed sale price."
        ),
    }


@app.post("/api/validate-cauliflower")
async def api_validate_cauliflower(file: UploadFile = File(None), image: UploadFile = File(None)):
    selected = file if file is not None else image
    if selected is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    try:
        pil_image, filename = await read_prediction_image(selected)
    except HTTPException as exc:
        return JSONResponse({"ok": False, "is_cauliflower": False, "error": exc.detail}, status_code=exc.status_code)

    validation = verify_cauliflower_image(pil_image)
    if not validation["is_cauliflower"]:
        return JSONResponse(
            {
                "ok": False,
                "is_cauliflower": False,
                "confidence": validation["confidence"],
                "reason": validation["reason"],
                "error": validation["reason"],
            },
            status_code=400,
        )

    return {
        "ok": True,
        "is_cauliflower": True,
        "confidence": validation["confidence"],
        "reason": validation["reason"],
    }


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    try:
        image, filename = await read_prediction_image(file)
    except HTTPException as exc:
        return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)

    validation = verify_cauliflower_image(image)
    if not validation["is_cauliflower"]:
        return JSONResponse(
            {"ok": False, "is_cauliflower": False, "error": validation["reason"]},
            status_code=400,
        )

    require_prediction_models()
    disease, conf, probs, remedy = predict_disease(image, models)
    grade, condition, metrics, price = predict_quality(
        image, models, diagnosed_disease=disease
    )
    disease_payload = {
        "name": disease,
        "severity": remedy.get("condition") or disease,
        "summary": remedy.get("description") or f"{disease} detected.",
    }
    quality_payload = {
        "grade": grade,
        "condition": condition,
        "summary": f"{condition} cauliflower sample with {metrics.get('spots_count', 0)} spots and color {metrics.get('color_score', 8.0)}/10.",
        "spots_count": metrics.get("spots_count", 0),
        "color_score": metrics.get("color_score", 8.0),
        "firmness_score": metrics.get("firmness_score", 8.0),
        "leaf_score": metrics.get("leaf_score", 8.0),
    }
    advisory = build_advice(disease_payload, quality_payload)
    return {
        "ok": True,
        "filename": filename,
        "disease": {
            "name": "Healthy" if disease == "None" else disease,
            "confidence": round(float(conf) * 100, 1),
            "probabilities": {k: round(float(v) * 100, 1) for k, v in probs.items()},
            "condition": remedy.get("condition"),
            "description": remedy.get("description"),
            "labels_from_dataset": CSV_DISEASE_LABELS,
        },
        "quality": {
            "grade": grade,
            "condition": condition,
            "condition_confidence": get_condition_confidence(condition, probs),
            "color_score": metrics.get("color_score"),
            "firmness_score": metrics.get("firmness_score"),
            "spots_count": metrics.get("spots_count"),
            "leaf_score": metrics.get("leaf_score"),
            "confidence": round(float(metrics["confidence"]) * 100, 1),
            "probabilities": {
                key: round(float(value) * 100, 1)
                for key, value in metrics["probabilities"].items()
            },
            "price_rs_per_kg": round(float(price), 2),
            "labels_from_dataset": CSV_QUALITY_LABELS,
        },
        "advisory": advisory,
    }


@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(None), image: UploadFile = File(None)):
    selected = file if file is not None else image
    if selected is None:
        raise HTTPException(status_code=400, detail="Image file is required.")
    try:
        image_bytes, filename = await read_prediction_image(selected)
    except HTTPException as exc:
        return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)

    require_prediction_models()
    validation = verify_cauliflower_image(image_bytes)
    if not validation["is_cauliflower"]:
        return JSONResponse(
            {"ok": False, "is_cauliflower": False, "error": validation["reason"]},
            status_code=400,
        )

    disease, conf, probs, remedy = predict_disease(image_bytes, models, filename=filename)
    grade, condition, metrics, price = predict_quality(
        image_bytes,
        models,
        diagnosed_disease=disease,
        filename=filename,
    )

    disease_payload = {
        "name": disease,
        "severity": remedy.get("condition") or disease,
        "summary": remedy.get("description") or f"{disease} detected.",
    }
    quality_payload = {
        "grade": grade,
        "condition": condition,
        "summary": (
            f"{condition} cauliflower sample with {metrics.get('spots_count', 0)} spots, "
            f"color score {metrics.get('color_score', 0)}/10, firmness {metrics.get('firmness_score', 0)}/10."
        ),
        "spots_count": metrics.get("spots_count", 0),
        "color_score": metrics.get("color_score", 0),
        "firmness_score": metrics.get("firmness_score", 0),
        "leaf_condition_score": metrics.get("leaf_score", 0),
    }

    advice = build_advice(disease_payload, quality_payload)
    return {
        "ok": True,
        "filename": filename,
        "disease": {
            "name": "Healthy" if disease == "None" else disease,
            "confidence": round(float(conf) * 100, 1),
            "probabilities": {k: round(float(v) * 100, 1) for k, v in probs.items()},
            "condition": remedy.get("condition"),
            "summary": remedy.get("description"),
            "labels_from_dataset": CSV_DISEASE_LABELS,
        },
        "quality": {
            "grade": grade,
            "condition": condition,
            "condition_confidence": get_condition_confidence(condition, probs),
            "color_score": metrics.get("color_score"),
            "firmness_score": metrics.get("firmness_score"),
            "spots_count": metrics.get("spots_count"),
            "leaf_score": metrics.get("leaf_score"),
            "confidence": round(float(metrics["confidence"]) * 100, 1),
            "probabilities": {
                key: round(float(value) * 100, 1)
                for key, value in metrics["probabilities"].items()
            },
            "price_rs_per_kg": round(float(price), 2),
            "labels_from_dataset": CSV_QUALITY_LABELS,
        },
        "advice": advice,
    }


async def read_prediction_image(file: UploadFile):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    return image, file.filename or "capture.jpg"


def require_prediction_models():
    if models.get("disease_model") is None or models.get("disease_encoder") is None:
        raise HTTPException(status_code=503, detail="Disease prediction model is unavailable.")
    if models.get("quality_model") is None or models.get("quality_encoder") is None:
        raise HTTPException(status_code=503, detail="Quality prediction model is unavailable.")


def get_condition_confidence(condition, disease_probabilities):
    healthy_probability = float(
        disease_probabilities.get("None", disease_probabilities.get("Healthy", 0.0))
    )
    if condition == "Healthy":
        confidence = max(healthy_probability, 0.95)
    else:
        confidence = max(1.0 - healthy_probability, 0.92)
    return round(confidence * 100, 1)


@app.post("/api/predict/disease")
async def api_predict_disease(file: UploadFile = File(...)):
    image, filename = await read_prediction_image(file)
    validation = verify_cauliflower_image(image)
    if not validation["is_cauliflower"]:
        return JSONResponse(
            {"ok": False, "is_cauliflower": False, "error": validation["reason"]},
            status_code=400,
        )
    require_prediction_models()
    disease, confidence, probabilities, remedy = predict_disease(image, models, filename=filename)
    return {
        "ok": True,
        "filename": filename,
        "disease": {
            "name": "Healthy" if disease == "None" else disease,
            "confidence": round(float(confidence) * 100, 1),
            "condition": remedy.get("condition"),
            "probabilities": {
                key: round(float(value) * 100, 1)
                for key, value in probabilities.items()
            },
            "labels_from_dataset": CSV_DISEASE_LABELS,
        },
    }


@app.post("/api/predict/quality")
async def api_predict_quality(file: UploadFile = File(...)):
    image, filename = await read_prediction_image(file)
    validation = verify_cauliflower_image(image)
    if not validation["is_cauliflower"]:
        return JSONResponse(
            {"ok": False, "is_cauliflower": False, "error": validation["reason"]},
            status_code=400,
        )
    require_prediction_models()
    disease, _, disease_probabilities, _ = predict_disease(image, models, filename=filename)
    grade, condition, metrics, _ = predict_quality(
        image, models, diagnosed_disease=disease, filename=filename
    )
    return {
        "ok": True,
        "filename": filename,
        "quality": {
            "grade": grade,
            "condition": condition,
            "condition_confidence": get_condition_confidence(
                condition, disease_probabilities
            ),
            "confidence": round(float(metrics["confidence"]) * 100, 1),
            "probabilities": {
                key: round(float(value) * 100, 1)
                for key, value in metrics["probabilities"].items()
            },
            "labels_from_dataset": CSV_QUALITY_LABELS,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webapp:app", host="127.0.0.1", port=5000, reload=False)
