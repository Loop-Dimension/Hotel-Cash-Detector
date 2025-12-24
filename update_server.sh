#!/bin/bash

# Hotel CCTV - Server Update Script
# This script pulls latest code, installs dependencies, and restarts the service

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
SERVICE_NAME="hotel-cctv"

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
    echo -e "${GREEN}✓ Git update successful (code files reset, media/static preserved)${NC}"
else
    echo -e "${RED}✗ Git update failed${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}3. Activating virtual environment...${NC}"
source venv/bin/activate
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Failed to activate virtual environment${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}4. Installing/updating Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}5. Running database migrations...${NC}"
python manage.py migrate --noinput
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Migrations applied${NC}"
else
    echo -e "${RED}✗ Migration failed${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}6. Collecting static files...${NC}"
python manage.py collectstatic --noinput
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Static files collected${NC}"
else
    echo -e "${RED}✗ Failed to collect static files${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}7. Restarting service...${NC}"
sudo systemctl restart "$SERVICE_NAME"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Service restarted${NC}"
else
    echo -e "${RED}✗ Failed to restart service${NC}"
    exit 1
fi
echo ""

echo -e "${YELLOW}9. Setting execute permissions on update script...${NC}"
chmod +x update_server.sh
echo -e "${GREEN}✓ Execute permissions set${NC}"
echo ""

echo -e "${YELLOW}9. Checking service status...${NC}"
sleep 3
sudo systemctl status "$SERVICE_NAME" --no-pager
echo ""

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Update completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

echo -e "${YELLOW}Quick commands:${NC}"
echo "  View logs:      sudo journalctl -u $SERVICE_NAME -f"
echo "  Service status: sudo systemctl status $SERVICE_NAME"
echo "  Restart:        sudo systemctl restart $SERVICE_NAME"
echo ""
