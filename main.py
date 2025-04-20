from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import tensorflow as tf
import numpy as np
import pytz
from datetime import datetime
import uvicorn

app = FastAPI(title="🇦🇪 UAE Diabetes Doctor", version="2.2.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load UAE-trained model
model = tf.keras.models.load_model("diabetes_model.h5")

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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "current_date": datetime.now(pytz.timezone("Asia/Dubai")).strftime("%d %B %Y")
    })

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request,
                 age: int = Form(...),
                 bmi: float = Form(...),
                 glucose: int = Form(...),
                 hba1c: float = Form(...)):
    
    try:
        # Validate UAE-specific ranges
        if not 18 <= age <= 120:
            raise ValueError("Age must be 18-120 (DHA guidelines)")
        if not 15 <= bmi <= 45:
            raise ValueError("BMI must be 15-45")
        
        # Predict risk
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