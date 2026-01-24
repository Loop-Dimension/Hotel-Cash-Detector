# Data Models - Hotel Cash Detector

Complete database schema and data model documentation.

## Table of Contents

1. [Entity Relationship Diagram](#entity-relationship-diagram)
2. [Model Definitions](#model-definitions)
3. [Database Indexes](#database-indexes)
4. [Model Relationships](#model-relationships)
5. [Query Optimization](#query-optimization)

---

## Entity Relationship Diagram

### Complete Schema

```mermaid
erDiagram
    User ||--o{ Branch : manages
    Region ||--o{ Branch : contains
    Branch ||--o{ Camera : has
    Camera ||--o{ Event : generates
    User ||--o{ Event : reviews

    User {
        int id PK
        string username UK
        string email UK
        string password
        string role
        string phone
        datetime date_joined
        datetime last_login
        boolean is_active
        boolean is_staff
        boolean is_superuser
    }

    Region {
        int id PK
        string name UK
        string code UK
        datetime created_at
    }

    Branch {
        int id PK
        string name
        int region_id FK
        string address
        string status
        datetime created_at
        datetime updated_at
    }

    Camera {
        int id PK
        int branch_id FK
        string camera_id UK
        string name
        string rtsp_url
        string status
        int cashier_zone_x
        int cashier_zone_y
        int cashier_zone_width
        int cashier_zone_height
        boolean cashier_zone_enabled
        float cash_confidence
        float violence_confidence
        float fire_confidence
        int hand_touch_distance
        boolean detect_cash
        boolean detect_violence
        boolean detect_fire
        datetime created_at
        datetime updated_at
    }

    Event {
        int id PK
        int branch_id FK
        int camera_id FK
        string event_type
        string status
        float confidence
        int frame_number
        int bbox_x1
        int bbox_y1
        int bbox_x2
        int bbox_y2
        string clip_path
        string thumbnail_path
        text notes
        int reviewed_by FK
        datetime created_at
        datetime updated_at
    }
```

### Domain Organization

**User Management Domain**:
- `User` - System users with role-based access

**Geographic Domain**:
- `Region` - Geographic regions (Bangkok, Phuket, etc.)
- `Branch` - Hotels/properties within regions

**CCTV Domain**:
- `Camera` - RTSP cameras with detection settings
- `Event` - Detected events (cash, violence, fire)

---

## Model Definitions

### User Model

**Purpose**: Authentication, authorization, and event review tracking.

**Definition**:
```python
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    """Extended Django user model with role-based access."""

    ROLE_CHOICES = [
        ('admin', 'Admin (Master)'),
        ('project_manager', 'Project Manager'),
    ]

    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='project_manager'
    )
    phone = models.CharField(
        max_length=20,
        blank=True,
        null=True
    )

    class Meta:
        db_table = 'cctv_user'
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return f"{self.username} ({self.get_role_display()})"
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | AutoField | PK | Primary key |
| `username` | CharField(150) | UNIQUE, NOT NULL | Login username |
| `email` | EmailField | UNIQUE, NOT NULL | Email address |
| `password` | CharField(128) | NOT NULL | Hashed password (bcrypt) |
| `role` | CharField(20) | NOT NULL | `admin` or `project_manager` |
| `phone` | CharField(20) | NULL | Phone number (optional) |
| `date_joined` | DateTimeField | NOT NULL, auto | Account creation timestamp |
| `last_login` | DateTimeField | NULL | Last login timestamp |
| `is_active` | BooleanField | NOT NULL, default=True | Account active status |
| `is_staff` | BooleanField | NOT NULL, default=False | Django admin access |
| `is_superuser` | BooleanField | NOT NULL, default=False | Superuser permissions |

**Role-Based Access**:
- **Admin (Master)**: Full access to all regions, branches, cameras
- **Project Manager**: Access only to assigned branches

**Password Storage**: Uses Django's `make_password()` with PBKDF2-SHA256 hashing.

---

### Region Model

**Purpose**: Geographic organization of hotel properties.

**Definition**:
```python
class Region(models.Model):
    """Geographic region containing multiple branches."""

    name = models.CharField(
        max_length=50,
        unique=True,
        help_text="Region name (e.g., 'Bangkok', 'Phuket')"
    )
    code = models.CharField(
        max_length=10,
        unique=True,
        help_text="Short code (e.g., 'BKK', 'HKT')"
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'cctv_region'
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.code})"
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | AutoField | PK | Primary key |
| `name` | CharField(50) | UNIQUE, NOT NULL | Region name (e.g., "Bangkok") |
| `code` | CharField(10) | UNIQUE, NOT NULL | Short code (e.g., "BKK") |
| `created_at` | DateTimeField | NOT NULL, auto | Creation timestamp |

**Examples**:
```python
Region.objects.create(name='Bangkok', code='BKK')
Region.objects.create(name='Phuket', code='HKT')
Region.objects.create(name='Chiang Mai', code='CNX')
```

---

### Branch Model

**Purpose**: Individual hotel properties within regions.

**Definition**:
```python
class Branch(models.Model):
    """Hotel property with assigned managers."""

    STATUS_CHOICES = [
        ('confirmed', 'Confirmed'),
        ('reviewing', 'Reviewing'),
        ('pending', 'Pending'),
    ]

    name = models.CharField(max_length=100)
    region = models.ForeignKey(
        Region,
        on_delete=models.CASCADE,
        related_name='branches'
    )
    address = models.TextField(blank=True, null=True)
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='confirmed'
    )
    managers = models.ManyToManyField(
        User,
        related_name='managed_branches',
        blank=True
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cctv_branch'
        ordering = ['region', 'name']
        verbose_name_plural = 'Branches'

    def __str__(self):
        return f"{self.name} ({self.region.code})"
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | AutoField | PK | Primary key |
| `name` | CharField(100) | NOT NULL | Branch name (e.g., "Seoul Grand Hotel") |
| `region_id` | ForeignKey | FK → Region, NOT NULL | Parent region |
| `address` | TextField | NULL | Physical address |
| `status` | CharField(20) | NOT NULL | `confirmed`, `reviewing`, `pending` |
| `managers` | ManyToManyField | → User | Assigned project managers |
| `created_at` | DateTimeField | NOT NULL, auto | Creation timestamp |
| `updated_at` | DateTimeField | NOT NULL, auto | Last update timestamp |

**Relationships**:
- **Region** (Many-to-One): Each branch belongs to one region
- **Managers** (Many-to-Many): Multiple managers can be assigned to a branch

---

### Camera Model

**Purpose**: RTSP camera configuration and detection settings.

**Definition**:
```python
class Camera(models.Model):
    """RTSP camera with detection configuration."""

    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('maintenance', 'Maintenance'),
    ]

    branch = models.ForeignKey(
        Branch,
        on_delete=models.CASCADE,
        related_name='cameras'
    )
    camera_id = models.CharField(
        max_length=50,
        unique=True,
        help_text="Unique camera identifier"
    )
    name = models.CharField(max_length=100)
    rtsp_url = models.CharField(
        max_length=500,
        help_text="rtsp://username:password@ip:port/stream/path"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='online'
    )

    # Cashier zone coordinates
    cashier_zone_x = models.IntegerField(default=0)
    cashier_zone_y = models.IntegerField(default=0)
    cashier_zone_width = models.IntegerField(default=640)
    cashier_zone_height = models.IntegerField(default=480)
    cashier_zone_enabled = models.BooleanField(default=False)

    # Detection thresholds (0.0-1.0)
    cash_confidence = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    violence_confidence = models.FloatField(
        default=0.6,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    fire_confidence = models.FloatField(
        default=0.5,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    hand_touch_distance = models.IntegerField(
        default=100,
        help_text="Maximum pixel distance between hands for cash detection"
    )

    # Detection toggles
    detect_cash = models.BooleanField(default=True)
    detect_violence = models.BooleanField(default=True)
    detect_fire = models.BooleanField(default=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cctv_camera'
        ordering = ['branch', 'name']

    def __str__(self):
        return f"{self.name} ({self.branch.name})"
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | AutoField | PK | Primary key |
| `branch_id` | ForeignKey | FK → Branch, NOT NULL | Parent branch |
| `camera_id` | CharField(50) | UNIQUE, NOT NULL | Unique identifier (e.g., "CAM001") |
| `name` | CharField(100) | NOT NULL | Display name (e.g., "Front Desk Camera") |
| `rtsp_url` | CharField(500) | NOT NULL | RTSP stream URL |
| `status` | CharField(20) | NOT NULL | `online`, `offline`, `maintenance` |
| `cashier_zone_x` | Integer | NOT NULL, default=0 | Zone top-left X coordinate |
| `cashier_zone_y` | Integer | NOT NULL, default=0 | Zone top-left Y coordinate |
| `cashier_zone_width` | Integer | NOT NULL, default=640 | Zone width (pixels) |
| `cashier_zone_height` | Integer | NOT NULL, default=480 | Zone height (pixels) |
| `cashier_zone_enabled` | Boolean | NOT NULL, default=False | Zone detection enabled |
| `cash_confidence` | Float | 0.0-1.0, default=0.5 | Cash detection threshold |
| `violence_confidence` | Float | 0.0-1.0, default=0.6 | Violence detection threshold |
| `fire_confidence` | Float | 0.0-1.0, default=0.5 | Fire detection threshold |
| `hand_touch_distance` | Integer | NOT NULL, default=100 | Max pixels between hands |
| `detect_cash` | Boolean | NOT NULL, default=True | Enable cash detection |
| `detect_violence` | Boolean | NOT NULL, default=True | Enable violence detection |
| `detect_fire` | Boolean | NOT NULL, default=True | Enable fire detection |
| `created_at` | DateTimeField | NOT NULL, auto | Creation timestamp |
| `updated_at` | DateTimeField | NOT NULL, auto | Last update timestamp |

**RTSP URL Format**:
```
rtsp://username:password@ip:port/stream/path
```

**Examples**:
```
rtsp://admin:password@192.168.1.64:554/Streaming/Channels/101
rtsp://admin:adminadmin!@175.213.55.16:554
```

---

### Event Model

**Purpose**: Detected events (cash transactions, violence, fire) with video clips.

**Definition**:
```python
class Event(models.Model):
    """Detection event with video clip and metadata."""

    TYPE_CHOICES = [
        ('cash', 'Cash Transaction'),
        ('fire', 'Fire'),
        ('violence', 'Violence/Disturbance'),
    ]

    STATUS_CHOICES = [
        ('pending', 'Pending Review'),
        ('reviewing', 'Under Review'),
        ('confirmed', 'Confirmed'),
        ('false_positive', 'False Positive'),
    ]

    branch = models.ForeignKey(
        Branch,
        on_delete=models.CASCADE,
        related_name='events'
    )
    camera = models.ForeignKey(
        Camera,
        on_delete=models.CASCADE,
        related_name='events'
    )
    event_type = models.CharField(
        max_length=20,
        choices=TYPE_CHOICES
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    confidence = models.FloatField(
        default=0.0,
        help_text="Detection confidence (0.0-1.0)"
    )
    frame_number = models.IntegerField(
        default=0,
        help_text="Frame number in stream"
    )

    # Bounding box coordinates
    bbox_x1 = models.IntegerField(default=0)
    bbox_y1 = models.IntegerField(default=0)
    bbox_x2 = models.IntegerField(default=0)
    bbox_y2 = models.IntegerField(default=0)

    # Media files
    clip_path = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Path to 30-second video clip"
    )
    thumbnail_path = models.CharField(
        max_length=500,
        blank=True,
        null=True,
        help_text="Path to thumbnail image"
    )

    # Review
    notes = models.TextField(blank=True, null=True)
    reviewed_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='reviewed_events'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'cctv_event'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['camera', '-created_at']),
            models.Index(fields=['event_type', 'status']),
            models.Index(fields=['branch', '-created_at']),
        ]

    def __str__(self):
        return f"{self.get_event_type_display()} - {self.camera.name} ({self.created_at})"
```

**Fields**:

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `id` | AutoField | PK | Primary key |
| `branch_id` | ForeignKey | FK → Branch, NOT NULL | Parent branch |
| `camera_id` | ForeignKey | FK → Camera, NOT NULL | Source camera |
| `event_type` | CharField(20) | NOT NULL | `cash`, `violence`, `fire` |
| `status` | CharField(20) | NOT NULL | `pending`, `reviewing`, `confirmed`, `false_positive` |
| `confidence` | Float | NOT NULL, default=0.0 | Detection confidence (0.0-1.0) |
| `frame_number` | Integer | NOT NULL, default=0 | Frame number in stream |
| `bbox_x1` | Integer | NOT NULL, default=0 | Bounding box top-left X |
| `bbox_y1` | Integer | NOT NULL, default=0 | Bounding box top-left Y |
| `bbox_x2` | Integer | NOT NULL, default=0 | Bounding box bottom-right X |
| `bbox_y2` | Integer | NOT NULL, default=0 | Bounding box bottom-right Y |
| `clip_path` | CharField(500) | NULL | Relative path to video clip |
| `thumbnail_path` | CharField(500) | NULL | Relative path to thumbnail |
| `notes` | TextField | NULL | Reviewer notes |
| `reviewed_by` | ForeignKey | FK → User, NULL | Reviewing user |
| `created_at` | DateTimeField | NOT NULL, auto | Event timestamp |
| `updated_at` | DateTimeField | NOT NULL, auto | Last update timestamp |

**Event Lifecycle**:
1. **pending** → Event just detected, awaiting review
2. **reviewing** → Assigned to reviewer
3. **confirmed** → Verified as real event
4. **false_positive** → Not a real event (can be deleted)

---

## Database Indexes

### Primary Indexes

**Automatic Indexes** (Django creates these):
- `cctv_user.id` (PK)
- `cctv_region.id` (PK)
- `cctv_branch.id` (PK)
- `cctv_camera.id` (PK)
- `cctv_event.id` (PK)

**Unique Indexes**:
- `cctv_user.username` (UNIQUE)
- `cctv_user.email` (UNIQUE)
- `cctv_region.name` (UNIQUE)
- `cctv_region.code` (UNIQUE)
- `cctv_camera.camera_id` (UNIQUE)

### Foreign Key Indexes

**Automatic Indexes** on foreign keys:
- `cctv_branch.region_id`
- `cctv_camera.branch_id`
- `cctv_event.branch_id`
- `cctv_event.camera_id`
- `cctv_event.reviewed_by_id`

### Composite Indexes

**Event Model** (for efficient queries):
```python
indexes = [
    models.Index(fields=['camera', '-created_at']),      # Query events by camera
    models.Index(fields=['event_type', 'status']),       # Filter by type and status
    models.Index(fields=['branch', '-created_at']),      # Query events by branch
]
```

**Query Performance**:
```python
# Efficient: Uses index on (camera_id, -created_at)
events = Event.objects.filter(camera_id=1).order_by('-created_at')[:10]

# Efficient: Uses index on (event_type, status)
cash_pending = Event.objects.filter(event_type='cash', status='pending')

# Efficient: Uses index on (branch_id, -created_at)
branch_events = Event.objects.filter(branch_id=5).order_by('-created_at')
```

---

## Model Relationships

### One-to-Many Relationships

```mermaid
graph LR
    Region -->|has many| Branch
    Branch -->|has many| Camera
    Camera -->|generates many| Event
    User -->|reviews many| Event

    style Region fill:#e3f2fd
    style Branch fill:#fff3e0
    style Camera fill:#f3e5f5
    style Event fill:#fce4ec
    style User fill:#e8f5e9
```

**Access Patterns**:
```python
# Region → Branches
region = Region.objects.get(code='BKK')
branches = region.branches.all()

# Branch → Cameras
branch = Branch.objects.get(name='Seoul Hotel')
cameras = branch.cameras.all()

# Camera → Events
camera = Camera.objects.get(camera_id='CAM001')
events = camera.events.filter(event_type='cash').order_by('-created_at')

# User → Reviewed Events
user = User.objects.get(username='admin')
reviewed = user.reviewed_events.filter(status='confirmed')
```

### Many-to-Many Relationships

**Branch ↔ Managers** (User):

```python
# Assign manager to branch
branch = Branch.objects.get(id=1)
manager = User.objects.get(username='john')
branch.managers.add(manager)

# Query manager's branches
user = User.objects.get(username='john')
managed_branches = user.managed_branches.all()

# Query branch managers
branch = Branch.objects.get(id=1)
managers = branch.managers.all()
```

---

## Query Optimization

### Select Related (Foreign Keys)

**Problem**: N+1 queries
```python
# BAD: 1 + N queries (N = number of events)
events = Event.objects.all()
for event in events:
    print(event.camera.name)       # Query per event
    print(event.branch.name)       # Query per event
```

**Solution**: Use `select_related()`
```python
# GOOD: 1 query with JOINs
events = Event.objects.select_related('camera', 'branch', 'reviewed_by').all()
for event in events:
    print(event.camera.name)       # No additional query
    print(event.branch.name)       # No additional query
```

### Prefetch Related (Many-to-Many)

**Problem**: N+1 queries
```python
# BAD: 1 + N queries
branches = Branch.objects.all()
for branch in branches:
    print(branch.managers.count())  # Query per branch
```

**Solution**: Use `prefetch_related()`
```python
# GOOD: 2 queries total (1 for branches, 1 for managers)
branches = Branch.objects.prefetch_related('managers').all()
for branch in branches:
    print(branch.managers.count())  # No additional query
```

### Efficient Event Queries

**Filter events by region**:
```python
# Filter by region with joins
events = Event.objects.filter(
    branch__region__code='BKK',
    event_type='cash',
    created_at__gte=start_date
).select_related('camera', 'branch', 'branch__region').order_by('-created_at')
```

**Count events by type**:
```python
from django.db.models import Count

# Aggregate counts
stats = Event.objects.filter(
    camera_id=1
).values('event_type').annotate(
    count=Count('id')
).order_by('event_type')

# Result: [{'event_type': 'cash', 'count': 150}, ...]
```

---

## Related Documentation

- [01 - Architecture](01-architecture.md) - System architecture and database design
- [04 - Features](04-features.md) - Detection features and event types
- [06 - Integrations](06-integrations.md) - PMS integration (project sync)
- [07 - Flows](07-flows.md) - Event creation workflows

---

**Previous**: [← 04 - Features](04-features.md) | **Next**: [06 - Integrations →](06-integrations.md)
