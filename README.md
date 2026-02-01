# Hotel Cash Detector - AI-Powered CCTV Monitoring

Real-time AI detection for cash transactions, violence, and fire in hotel environments.

## Quick Start

- [Full Documentation](docs/00-overview.md)
- [Setup Guide](docs/02-setup.md)
- [한국어 문서](docs/ko/)

## Key Features

### AI Detection Models
- **Cash Transaction Detection**: Hand-to-hand exchange recognition (YOLOv8-Pose)
- **Violence/Disturbance Detection**: Aggressive behavior and close combat
- **Fire/Smoke Detection**: Real-time fire and smoke detection

### System Capabilities
- Multi-camera support (8+ simultaneous streams)
- Real-time RTSP video processing
- Background detection workers
- Event logging with video clip recording
- Multi-language support (EN, KO, TH, VI, ZH)
- Developer mode for debugging and tuning

## Technology

- **Framework**: Django 5.2.7
- **AI Models**: YOLOv8, YOLOv8-Pose (PyTorch)
- **Computer Vision**: OpenCV, FFmpeg
- **Video**: RTSP streaming (TCP), H.264 encoding
- **Database**: SQLite (dev), PostgreSQL (production)

## Integration

This system integrates with:
- [HotelPMS](../HotelPMS) - Central authentication and project management
- **RTSP Cameras** - 8+ multi-camera support
- **GPU** - CUDA acceleration for real-time detection

See [docs/06-integrations.md](docs/06-integrations.md) for integration details.

## Hardware Requirements

**Minimum**:
- i5 CPU or equivalent
- 8 GB RAM
- 50 GB storage
- CUDA-capable GPU (GTX 1650+)

**Recommended (Production)**:
- i7+ CPU or AWS g4dn.xlarge
- 16 GB+ RAM
- GPU (GTX 1660+, Tesla T4, or better)
- 100+ GB SSD storage

## Quick Links

- **Dashboard**: http://localhost:8000
- **Admin**: http://localhost:8000/admin
- **API**: http://localhost:8000/api/
- **Live Monitoring**: http://localhost:8000/monitor/all/

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download AI models (auto-downloads on first run)
python -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); YOLO('yolov8s-pose.pt')"

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver 0.0.0.0:8000
```

## Docker Deployment (Microservices)

The system supports a microservices architecture with ML detection separated from the Django backend:

```
┌─────────────────┐     ┌─────────────────┐
│  Django Backend │────▶│   ML Service    │
│   (port 8000)   │     │   (port 8001)   │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   PostgreSQL    │     │   AI Models     │
└─────────────────┘     └─────────────────┘
```

### Quick Start with Docker

```bash
# 1. Copy environment file
cp .env.docker .env

# 2. Edit .env with your settings
#    - Set GEMINI_API_KEY
#    - Set DB_PASSWORD
#    - Set SECRET_KEY

# 3. Start all services
docker-compose up -d --build

# 4. View logs
docker-compose logs -f

# 5. Access the application
#    - Dashboard: http://localhost:80
#    - ML Service: http://localhost:8001/health
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| nginx | 80, 443 | Reverse proxy |
| django-backend | 8000 | Django web app |
| ml-service | 8001 | FastAPI ML detection |
| postgres | 5432 | PostgreSQL database |

### Environment Variables

```env
# Enable ML microservice mode
USE_ML_SERVICE=True
ML_SERVICE_URL=http://ml-service:8001
ML_SERVICE_TIMEOUT=30
```

For local development without Docker, set `USE_ML_SERVICE=False` to use local detectors.

## Documentation

For complete documentation, see:

- [Overview](docs/00-overview.md) - System overview and features
- [Architecture](docs/01-architecture.md) - AI models and technical design
- [Setup](docs/02-setup.md) - Development and production setup
- [Environment Variables](docs/03-env.md) - Configuration reference
- [Features](docs/04-features.md) - Detection features and capabilities
- [Data Models](docs/05-data-models.md) - Database schema
- [Integrations](docs/06-integrations.md) - PMS, RTSP, GPU integration
- [Flows](docs/07-flows.md) - Detection workflows
- [Deployment](docs/08-deployment.md) - AWS, Docker, GPU deployment
- [Troubleshooting](docs/09-troubleshooting.md) - Common issues

## License

Proprietary - All rights reserved
