from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import pytz
from datetime import datetime
import uvicorn
import os

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.h5")

# FastAPI app setup
app = FastAPI(title="🇦🇪 UAE Diabetes Doctor", version="2.2.0")

# Mount static files only if directory exists
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    print(f"⚠️ Warning: Static directory not found at {STATIC_DIR}. Skipping static mount.")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Load trained model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)# Run the app locally
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
else:
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")

# Diabetes advice based on risk level
DIABETES_KNOWLEDGE = {
    "low": [
        "Maintain traditional Emirati diet with portion control",
        "Limit luqaimat to 1-2 pieces weekly",
        "Daily 30-minute walks in cooler hours"
    ],
    "high": [
        "Immediate consultation at DHA-approved clinic",
        "Strict monitoring of dates consumption",
        "Ramadan fasting guidance required"
    ]
}

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_date": datetime.now(pytz.timezone("Asia/Dubai")).strftime("%d %B %Y")
    })

# Analyze route for diabetes prediction
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request,
                 age: int = Form(...),
                 bmi: float = Form(...),
                 glucose: int = Form(...),
                 hba1c: float = Form(...)):
    try:
        # Validate inputs
        if not 18 <= age <= 120:
            raise ValueError("Age must be between 18 and 120 (per DHA guidelines).")
        if not 15 <= bmi <= 45:
            raise ValueError("BMI must be between 15 and 45.")

        # Make prediction
        input_data = np.array([[age, bmi, glucose, hba1c]])
        risk = float(model.predict(input_data)[0][0]) * 100

        return templates.TemplateResponse("results.html", {
            "request": request,
            "risk": f"{risk:.1f}%",
            "advice": DIABETES_KNOWLEDGE["low" if risk < 50 else "high"],
            "emergency": "800-DHA (342)",
            "clinics": [
                "Dubai Diabetes Center - 04 219 2000",
                "Rashid Hospital - 04 219 3000"
            ]
        })

    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e),
            "dha_contact": "800342"
        })
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
