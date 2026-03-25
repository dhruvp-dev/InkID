import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn

from analyzer import analyze

# Ensure dirs exist
for p in ["static", "templates"]:
    if not os.path.exists(p): os.makedirs(p)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_endpoint(request: Request):
    # We parse the form manually because the JS sends FormData
    form = await request.form()
    known_texts = form.getlist("known_texts")
    test_text = form.get("test_text")

    # Filter empties
    known_texts = [t.strip() for t in known_texts if t.strip()]

    if not known_texts or not test_text:
        return JSONResponse({"error": "Missing input data"}, status_code=400)

    try:
        # Run the updated analyzer
        result = analyze(known_texts, test_text)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)