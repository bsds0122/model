from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.model import router

app = FastAPI(
    title="Cardiovascular Risk Assessment API"
)

# ----------------------------------------------------
# Register Routes
# ----------------------------------------------------
app.include_router(router)

# ----------------------------------------------------
# Health Check
# ----------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Cardiovascular Risk Assessment API is running."
    }