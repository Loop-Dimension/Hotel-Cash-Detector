# Templates Guide - Hotel Security System

## 📁 All Templates Created

Your complete UI system is ready! Here's what's been created:

---

## 🔐 Authentication

### 1. `login.html`
**URL:** `/login`

**Features:**
- Username & password fields
- Remember me checkbox
- Demo accounts (Admin / Manager)
- Responsive design
- Password visibility toggle

**Demo Credentials:**
```
Admin:
- Username: admin
- Password: admin123

Manager:
- Username: manager
- Password: manager123
```

---

## 👑 Admin Templates

### 2. `admin_dashboard.html`
**URL:** `/admin/dashboard`  
**User Type:** Admin Only

**Features:**
- Stats overview (Hotels, Managers, Cameras, Detections)
- Quick action buttons
- Recent hotels table
- Sidebar navigation
- Real-time clock

**Sidebar Menu:**
- Dashboard
- Hotels
- Managers
- Cameras
- All Detections
- Settings

---

### 3. `hotels_list.html`
**URL:** `/admin/hotels`  
**User Type:** Admin Only

**Features:**
- Grid view of all hotels
- Hotel cards with:
  - Name & location
  - Camera count
  - Manager assigned
  - Today's detections by type
  - Status (Active/Inactive)
- Filters (Search, Status, Sort)
- Add/Edit/Delete actions

**Actions per Hotel:**
- View details
- Edit configuration
- Delete hotel

---

## 👨‍💼 Manager Templates

### 4. `manager_dashboard.html`
**URL:** `/manager/dashboard`  
**User Type:** Manager Only

**Features:**
- Hotel-specific dashboard
- Stats overview (Cameras, Detections, Incidents)
- Detection by type breakdown:
  - Cash Detection
  - Fire Detection
  - Fight Detection
  - Violence Detection
- Recent activity feed
- Camera status grid

**Sidebar Menu:**
- Dashboard
- My Cameras
- Detections
- Reports
- Profile

---

## 🎥 Camera Management

### 5. `cameras_config.html`
**URL:** `/manager/cameras` or `/admin/cameras`  
**User Type:** Manager & Admin

**Features:**
- Camera selection grid
- Configuration form for selected camera:
  - Basic settings (Name, Location)
  - Detection types toggle (Cash, Fire, Fight, Violence)
  - Detection parameters (sliders):
    - Hand Touch Distance
    - Pose Confidence
    - Minimum Transaction Frames
    - Cashier Overlap Threshold
  - Cashier Zone definition (X, Y, Width, Height)
- Visual Zone Editor button
- Save/Reset options

---

## 🔍 Detection Management

### 6. `detections_list.html`
**URL:** `/manager/detections` or `/admin/detections`  
**User Type:** Manager & Admin

**Features:**
- Stats overview (Total, High Priority, Reviewed, Pending)
- Advanced filters:
  - Detection Type
  - Camera
  - Priority
  - Status
- Detection cards with:
  - Thumbnail preview
  - Type icon & badge
  - Camera & timestamp
  - People involved
  - Confidence score
  - Priority level
  - Review status
- View/Download actions per detection
- Pagination

**Detection Types Displayed:**
- 💰 Cash Detection (Green)
- 🔥 Fire Detection (Red)
- 👊 Fight Detection (Orange)
- 😡 Violence Detection (Purple)

---

### 7. `detection_detail.html`
**URL:** `/manager/detections/{id}` or `/admin/detections/{id}`  
**User Type:** Manager & Admin

**Features:**
- Full video player with controls
- Video information:
  - File name
  - Duration
  - Start/End time
  - Resolution & FPS
- Transaction analysis:
  - People involved
  - Hand distance metrics
  - Transaction duration
  - Confidence score (with progress bar)
- Detection details sidebar:
  - ID & Type
  - Priority & Status
  - Camera & Location
  - Date & Time
  - Reviewed by
- Quick actions:
  - Download video
  - Share with team
  - Export report
  - Flag for review
  - Delete detection
- Notes section (textarea to add comments)
- Related detections

---

## 🎨 Design System

### Color Scheme
```
Primary: Blue-600 (#2563eb)
Success: Green-600 (#16a34a)
Warning: Orange-600 (#ea580c)
Danger: Red-600 (#dc2626)
Info: Purple-600 (#9333ea)

Header: Slate-800 (#1e293b)
Background: Gray-100 (#f3f4f6)
```

### Icons (Font Awesome 6.4.0)
All templates use Font Awesome icons for consistency.

### Responsive Design
- Mobile-first approach
- Tailwind CSS breakpoints:
  - sm: 640px
  - md: 768px
  - lg: 1024px
  - xl: 1280px

---

## 📊 User Flow

### Admin Flow
```
Login (admin/admin123)
  ↓
Admin Dashboard
  ├─ View Hotels → Hotels List → Hotel Detail
  ├─ View Managers → Managers List → Manager Detail
  ├─ View Cameras → Cameras Config
  └─ View Detections → Detections List → Detection Detail
```

### Manager Flow
```
Login (manager/manager123)
  ↓
Manager Dashboard (for specific hotel)
  ├─ View My Cameras → Cameras Config
  ├─ View Detections → Detections List → Detection Detail
  └─ View Reports
```

---

## 🔗 URL Structure

### Authentication
- `/login` - Login page

### Admin URLs
- `/admin/dashboard` - Admin dashboard
- `/admin/hotels` - Hotels list
- `/admin/hotels/add` - Add hotel (to be implemented)
- `/admin/hotels/{id}` - Hotel details
- `/admin/hotels/{id}/edit` - Edit hotel
- `/admin/managers` - Managers list
- `/admin/managers/add` - Add manager
- `/admin/cameras` - All cameras
- `/admin/cameras/add` - Add camera
- `/admin/detections` - All detections
- `/admin/detections/{id}` - Detection detail
- `/admin/settings` - System settings

### Manager URLs
- `/manager/dashboard` - Manager dashboard
- `/manager/cameras` - My cameras
- `/manager/detections` - My detections
- `/manager/detections/{id}` - Detection detail
- `/manager/reports` - Reports
- `/manager/profile` - Profile settings

### Common URLs
- `/logout` - Logout

---

## 🎯 Features by Template

| Template | Video Player | Filters | Stats | Actions | Notes |
|----------|-------------|---------|-------|---------|-------|
| login.html | ❌ | ❌ | ❌ | ✅ | Demo login |
| admin_dashboard.html | ❌ | ❌ | ✅ | ✅ | Quick actions |
| hotels_list.html | ❌ | ✅ | ❌ | ✅ | CRUD operations |
| manager_dashboard.html | ❌ | ❌ | ✅ | ✅ | Hotel-specific |
| cameras_config.html | ❌ | ❌ | ❌ | ✅ | Full config |
| detections_list.html | ❌ | ✅ | ✅ | ✅ | Pagination |
| detection_detail.html | ✅ | ❌ | ❌ | ✅ | Full detail |

---

## 🚀 Next Steps for Backend Implementation

### 1. Authentication
```python
# Flask-Login or JWT tokens
@app.route('/login', methods=['POST'])
def login():
    # Verify credentials
    # Create session
    # Redirect based on role
```

### 2. Role-Based Access Control
```python
# Decorator for protected routes
@admin_required
def admin_dashboard():
    # Admin only
    
@manager_required
def manager_dashboard():
    # Manager only
```

### 3. API Endpoints
```python
# Hotels
GET    /api/hotels
POST   /api/hotels
GET    /api/hotels/{id}
PUT    /api/hotels/{id}
DELETE /api/hotels/{id}

# Managers
GET    /api/managers
POST   /api/managers
GET    /api/managers/{id}
PUT    /api/managers/{id}

# Cameras
GET    /api/cameras
POST   /api/cameras
GET    /api/cameras/{id}
PUT    /api/cameras/{id}/config

# Detections
GET    /api/detections
GET    /api/detections/{id}
PUT    /api/detections/{id}/status
POST   /api/detections/{id}/notes
```

### 4. Database Models
```python
class User:
    id, username, password_hash, role, hotel_id
    
class Hotel:
    id, name, location, status, manager_id
    
class Camera:
    id, name, hotel_id, location, config, status
    
class Detection:
    id, type, camera_id, timestamp, video_path,
    status, priority, confidence, reviewed_by
```

---

## 💡 Template Integration Tips

### Using with Flask
```python
from flask import render_template

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    # Get data from database
    stats = get_dashboard_stats()
    return render_template('admin_dashboard.html', stats=stats)

@app.route('/manager/detections/<detection_id>')
def detection_detail(detection_id):
    detection = get_detection(detection_id)
    return render_template('detection_detail.html', 
                         detection=detection)
```

### Dynamic Data
Replace hardcoded values with template variables:
```html
<!-- Before -->
<p class="text-3xl font-bold text-gray-800">12</p>

<!-- After -->
<p class="text-3xl font-bold text-gray-800">{{ hotel_count }}</p>
```

---

## 🎨 Customization

### Colors
Update Tailwind colors in template `<head>`:
```html
<script>
  tailwind.config = {
    theme: {
      extend: {
        colors: {
          primary: '#your-color',
        }
      }
    }
  }
</script>
```

### Branding
Replace placeholder text:
- "Hotel Security System" → Your brand name
- Icons and logos
- Color scheme

---

## ✅ What's Ready

- ✅ 7 complete HTML templates
- ✅ Responsive design (mobile-friendly)
- ✅ Consistent styling with Tailwind CSS
- ✅ Font Awesome icons
- ✅ Interactive elements (buttons, forms, filters)
- ✅ Two user roles (Admin & Manager)
- ✅ Detection type support (Cash, Fire, Fight, Violence)
- ✅ Video player integration
- ✅ Camera configuration interface

---

## 🔄 What Needs Backend

- ❌ Authentication logic
- ❌ Database integration
- ❌ API endpoints
- ❌ File upload handling
- ❌ Video processing
- ❌ Real-time updates
- ❌ Notifications
- ❌ User management

---

## 📖 Usage Example

**For Admin:**
1. Open `/login`
2. Enter `admin` / `admin123`
3. View dashboard with all hotels
4. Click "Hotels" to manage properties
5. Click on a hotel to see its cameras and detections
6. View/download detection clips

**For Manager:**
1. Open `/login`
2. Enter `manager` / `manager123`
3. View dashboard for their assigned hotel
4. Monitor detections by type
5. Configure cameras
6. Review and download clips

---

**Your templates are ready to use! Just connect them to your backend! 🚀**

