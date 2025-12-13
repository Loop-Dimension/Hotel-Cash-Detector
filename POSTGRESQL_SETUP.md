# PostgreSQL Setup for AWS EC2

## Install PostgreSQL on Ubuntu 24.04

```bash
# Update package list
sudo apt update

# Install PostgreSQL and contrib package
sudo apt install -y postgresql postgresql-contrib

# Start and enable PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Verify installation
sudo systemctl status postgresql
```

## Configure PostgreSQL Database

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user (run these commands in psql)
CREATE DATABASE cctv;
CREATE USER orange WITH PASSWORD '00oo00oo';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE cctv TO orange;

# For PostgreSQL 15+ (Ubuntu 24.04), also grant schema privileges
\c cctv
GRANT ALL ON SCHEMA public TO orange;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO orange;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO orange;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO orange;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO orange;

# Exit psql
\q
```

## Configure PostgreSQL for Django

```bash
# Edit PostgreSQL config to allow password authentication
sudo nano /etc/postgresql/16/main/pg_hba.conf

# Find this line:
# local   all             all                                     peer

# Change to:
# local   all             all                                     md5

# Also ensure this line exists for TCP/IP connections:
# host    all             all             127.0.0.1/32            md5

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## Test Database Connection

```bash
# Test connection with psql
psql -h localhost -U orange -d cctv -W
# Enter password: 00oo00oo

# If successful, you'll see:
# cctv=>

# Exit with \q
```

## Update Django Project

```bash
cd ~/Hotel-Cash-Detector

# Activate virtual environment
source venv/bin/activate

# Install PostgreSQL adapter (already in requirements.txt)
pip install psycopg2-binary

# Update .env file
nano .env

# Ensure these lines are set:
# DB_ENGINE=postgresql
# DB_NAME=cctv
# DB_USER=orange
# DB_PASSWORD=00oo00oo
# DB_HOST=localhost
# DB_PORT=5432

# Run migrations to create tables
python manage.py migrate

# Create admin user
python manage.py createsuperuser
# Or use seed command with .env credentials

# Restart service
sudo systemctl restart hotel-cctv
```

## Verify Django Connection

```bash
cd ~/Hotel-Cash-Detector
source venv/bin/activate

# Test database connection
python manage.py dbshell

# This should connect to PostgreSQL
# Exit with \q
```

## Common PostgreSQL Commands

```bash
# Switch to postgres user
sudo -u postgres psql

# List databases
\l

# Connect to database
\c cctv

# List tables
\dt

# List users
\du

# Show table structure
\d cctv_camera

# Exit
\q
```

## Backup and Restore

```bash
# Backup database
sudo -u postgres pg_dump cctv > cctv_backup.sql

# Restore database
sudo -u postgres psql cctv < cctv_backup.sql

# Backup specific table
sudo -u postgres pg_dump -t cctv_event cctv > events_backup.sql
```

## Performance Tuning (Optional)

```bash
# Edit PostgreSQL config
sudo nano /etc/postgresql/16/main/postgresql.conf

# Recommended settings for 16GB RAM (g4dn.xlarge)
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 1GB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
work_mem = 10MB
min_wal_size = 1GB
max_wal_size = 4GB

# Restart PostgreSQL
sudo systemctl restart postgresql
```

## Troubleshooting

**Connection Refused:**
```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check PostgreSQL is listening
sudo netstat -plnt | grep 5432

# Check logs
sudo tail -f /var/log/postgresql/postgresql-16-main.log
```

**Permission Denied:**
```bash
# Re-grant privileges
sudo -u postgres psql
\c cctv
GRANT ALL ON SCHEMA public TO orange;
```

**Django Migration Errors:**
```bash
# Drop and recreate database (CAUTION: deletes all data)
sudo -u postgres psql
DROP DATABASE cctv;
CREATE DATABASE cctv;
GRANT ALL PRIVILEGES ON DATABASE cctv TO orange;
\c cctv
GRANT ALL ON SCHEMA public TO orange;
\q

# Re-run migrations
python manage.py migrate
```

## Migration from SQLite to PostgreSQL

If you already have data in SQLite:

```bash
cd ~/Hotel-Cash-Detector
source venv/bin/activate

# Dump SQLite data
python manage.py dumpdata --natural-foreign --natural-primary \
  --exclude contenttypes --exclude auth.Permission \
  --exclude sessions.session > data_backup.json

# Update .env to use PostgreSQL
nano .env
# Set DB_ENGINE=postgresql

# Run migrations on empty PostgreSQL
python manage.py migrate

# Load data
python manage.py loaddata data_backup.json

# Restart service
sudo systemctl restart hotel-cctv
```
