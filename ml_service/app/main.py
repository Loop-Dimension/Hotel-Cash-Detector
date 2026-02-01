"""
ML Detection Service - FastAPI Application

This microservice handles all ML-based detection operations:
- Cash transaction detection
- Violence detection
- Fire/smoke detection
- Gemini AI validation
"""
import time
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings, get_gpu_info
from app.api.routes import router
from app.services.detector_manager import detector_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    print("=" * 50)
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print("=" * 50)

    # Log GPU status
    gpu_info = get_gpu_info()
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"GPU: {gpu_info['gpu_name']}")
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")

    print(f"Models Directory: {settings.MODELS_DIR}")
    print(f"USE_GPU Setting: {settings.USE_GPU}")
    print("=" * 50)

    yield

    print("Shutting down ML Detection Service...")
    # Cleanup detectors
    detector_manager.cleanup_all()


app = FastAPI(
    title=settings.APP_NAME,
    description="Microservice for ML-based detection (cash, violence, fire)",
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
