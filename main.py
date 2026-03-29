import os
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from analyzer import analyze

# Ensure environment is ready
for p in ["static", "templates"]:
    if not os.path.exists(p): os.makedirs(p)

app = FastAPI(title="InkID")

# Advanced CORS to prevent browser blocks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_endpoint(request: Request):
    try:
        form = await request.form()
        known_texts = form.getlist("known_texts")
        test_text = form.get("test_text")

        # Basic filtering
        known_texts = [t.strip() for t in known_texts if t.strip()]

        if not known_texts or not test_text:
            return JSONResponse({"error": "Baseline samples or test text is missing."}, status_code=400)

        # Run analysis
        result = analyze(known_texts, test_text)
        return JSONResponse(result)

    except Exception as e:
        # CRITICAL: This prints the real error to your terminal for debugging
        print(f"--- BACKEND CRASH ERROR ---")
        print(str(e))
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Internal Server Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    # Binding to 127.0.0.1 is much more stable for Localhost development
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)