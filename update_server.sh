#!/bin/bash

# Hotel CCTV - Server Update Script
# Updates both Django backend and ML detection service

set -e  # Exit on any error

echo "========================================="
echo "Hotel CCTV - Server Update Script"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/var/www/Hotel-Cash-Detector"
ML_SERVICE_DIR="$PROJECT_DIR/ml_service"
DJANGO_SERVICE="hotel-cctv"
ML_SERVICE="hotel-ml-service"

echo -e "${YELLOW}1. Navigating to project directory...${NC}"
cd "$PROJECT_DIR"
pwd
echo ""

echo -e "${YELLOW}2. Fetching latest code from Git (force update)...${NC}"
# Discard all local changes to tracked files only
# Media, static, .env, and other .gitignore files are preserved
git fetch origin main
git reset --hard origin/main
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] Git update successful (code files reset, media/static preserved)${NC}"
else
    echo -e "${RED}[FAIL] Git update failed${NC}"
    exit 1
fi
echo ""

# ==============================================================================
# Django Backend
# ==============================================================================

echo -e "${YELLOW}3. Updating Django backend...${NC}"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] Django virtual environment activated${NC}"
else
    echo -e "${RED}[FAIL] Failed to activate Django virtual environment${NC}"
    exit 1
fi

pip install --upgrade pip -q
pip install -r requirements.txt -q
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] Django dependencies installed${NC}"
else
    echo -e "${RED}[FAIL] Failed to install Django dependencies${NC}"
    exit 1
fi

python manage.py migrate --noinput
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] Migrations applied${NC}"
else
    echo -e "${RED}[FAIL] Migration failed${NC}"
    exit 1
fi

python manage.py collectstatic --noinput --clear -q
echo -e "${GREEN}[OK] Static files collected${NC}"
deactivate
echo ""

# ==============================================================================
# ML Detection Service
# ==============================================================================

echo -e "${YELLOW}4. Updating ML service...${NC}"
if [ -d "$ML_SERVICE_DIR" ]; then
    cd "$ML_SERVICE_DIR"

    # Create venv if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "  Creating ML service virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] ML service virtual environment activated${NC}"
    else
        echo -e "${RED}[FAIL] Failed to activate ML service virtual environment${NC}"
        exit 1
    fi

    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] ML service dependencies installed${NC}"
    else
        echo -e "${RED}[FAIL] Failed to install ML service dependencies${NC}"
        exit 1
    fi
    deactivate
    cd "$PROJECT_DIR"
else
    echo -e "${YELLOW}[SKIP] ML service directory not found${NC}"
fi
echo ""

# ==============================================================================
# Restart Services
# ==============================================================================

echo -e "${YELLOW}5. Restarting services...${NC}"

# Restart ML service first (Django depends on it)
if systemctl is-enabled "$ML_SERVICE" &>/dev/null; then
    systemctl restart "$ML_SERVICE"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[OK] ML service restarted${NC}"
    else
        echo -e "${RED}[FAIL] Failed to restart ML service${NC}"
    fi
    # Wait for ML service to be ready
    echo "  Waiting for ML service to start..."
    sleep 5
else
    echo -e "${YELLOW}[SKIP] ML service not enabled${NC}"
fi

# Restart Django service
systemctl restart "$DJANGO_SERVICE"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] Django service restarted${NC}"
else
    echo -e "${RED}[FAIL] Failed to restart Django service${NC}"
    exit 1
fi
echo ""

# ==============================================================================
# Post-update Checks
# ==============================================================================

echo -e "${YELLOW}6. Setting execute permissions on update script...${NC}"
chmod +x "$PROJECT_DIR/update_server.sh"
echo -e "${GREEN}[OK] Execute permissions set${NC}"
echo ""

echo -e "${YELLOW}7. Checking service status...${NC}"
sleep 3

echo ""
echo -e "${YELLOW}--- ML Service ---${NC}"
if systemctl is-enabled "$ML_SERVICE" &>/dev/null; then
    systemctl status "$ML_SERVICE" --no-pager -l | head -10
    echo ""
    # Health check
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK] ML service health check passed${NC}"
    else
        echo -e "${RED}[WARN] ML service health check failed (may still be starting)${NC}"
    fi
else
    echo -e "${YELLOW}[SKIP] ML service not enabled${NC}"
fi

echo ""
echo -e "${YELLOW}--- Django Service ---${NC}"
systemctl status "$DJANGO_SERVICE" --no-pager -l | head -10
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Update completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

echo -e "${YELLOW}Quick commands:${NC}"
echo "  Django logs:     journalctl -u $DJANGO_SERVICE -f"
echo "  ML service logs: journalctl -u $ML_SERVICE -f"
echo "  Django status:   systemctl status $DJANGO_SERVICE"
echo "  ML status:       systemctl status $ML_SERVICE"
echo "  Restart all:     systemctl restart $ML_SERVICE $DJANGO_SERVICE"
echo "  ML health:       curl http://localhost:8001/health"
echo ""
