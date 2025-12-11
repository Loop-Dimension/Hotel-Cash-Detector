# Upcoming Updates & Enhancements

## Planned Features

### 1. Dual-Zone Detection Enhancement (v1.1.0)

**Current Limitation:**
- Only one cashier zone is defined
- Detection requires person center point inside zone
- Single-zone approach may miss some transaction patterns

**Planned Enhancement:**
- Add **Cash Exchange Area** as a separate zone (similar to cashier zone)
- Two-phase detection algorithm:
  1. **Phase 1: Hand Touch Detection** - Detect when cashier and customer hands touch (exchange area)
  2. **Phase 2: Hand Movement Tracking** - Track if cashier's hand then moves into cashier zone (cash drawer/register/safe)
- **Complete Transaction Pattern:** Touch with customer → Put money in cash drawer = Confirmed cash transaction
- This will significantly improve detection accuracy by capturing the full transaction flow

**Implementation Details:**

#### New Database Fields (Camera Model)
```python
# Cash exchange zone coordinates
cash_exchange_zone_x = models.IntegerField(default=0)
cash_exchange_zone_y = models.IntegerField(default=0)
cash_exchange_zone_width = models.IntegerField(default=640)
cash_exchange_zone_height = models.IntegerField(default=480)
cash_exchange_zone_enabled = models.BooleanField(default=False)
```

#### Detection Logic Flow
```
┌────────────────────────────────────────────────────────────┐
│         ENHANCED DUAL-ZONE CASH DETECTION                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  1. DETECT HAND TOUCH IN CASH EXCHANGE AREA                │
│     └── Cashier hand + Customer hand                       │
│     └── Distance < hand_touch_distance                     │
│     └── Both hands within cash_exchange_zone               │
│     └── Record: touch_point, timestamp, cashier_hand_id    │
│                                                            │
│  2. TRACK CASHIER HAND MOVEMENT TO CASH DRAWER             │
│     └── Monitor the cashier's hand only                    │
│     └── Check if hand moves into cashier_zone (drawer)     │
│     └── Time window: 1-5 seconds after touch               │
│     └── Direction: exchange_area → cash_drawer             │
│                                                            │
│  3. CONFIRM CASH TRANSACTION                               │
│     └── ✓ Cashier touched customer's hand (exchange)      │
│     └── ✓ Cashier put hand in cash drawer/register        │
│     └── ✓ Complete transaction flow captured              │
│     └── ✓ Confidence threshold met                        │
│     └── → TRIGGER EVENT (Confirmed Cash Transaction)      │
│                                                            │
│  BENEFITS:                                                 │
│  • Captures complete transaction: touch → drawer          │
│  • Reduces false positives (requires 2-phase validation)  │
│  • Confirms money was actually stored in cash drawer      │
│  • Works with various counter layouts                     │
│  • Much higher accuracy than single-zone detection        │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Configuration UI Changes
- Add second zone drawing tool (similar to cashier zone)
- Visual distinction between zones:
  - **Cashier Zone** (Blue rectangle): The backend area where cashier stores money (cash drawer/register/safe)
  - **Cash Exchange Area** (Green rectangle): The counter surface where customer and cashier hands meet during transaction
- Enable/disable toggle for dual-zone mode
- Settings page will show both zones with independent controls

#### JSON Metadata Enhancement
```json
{
  "event_type": "cash",
  "detection_method": "dual_zone",
  
  "phase_1_touch": {
    "location": "cash_exchange_area",
    "touch_point": [640, 450],
    "timestamp": "2025-12-10T14:30:45.123",
    "hand1_id": "person_0_right",
    "hand2_id": "person_1_left",
    "distance": 85.5
  },
  
  "phase_2_movement": {
    "tracked_hand": "cashier_right",
    "entry_point": [720, 380],
    "cash_drawer_entry_time": "2025-12-10T14:30:46.456",
    "movement_duration_ms": 1333,
    "movement_path": [[640, 450], [680, 420], [720, 380]],
    "action": "put_money_in_drawer"
  },
  
  "cash_exchange_zone": [100, 300, 600, 400],
  "cashier_zone": [2, 354, 1273, 359],
  
  "confidence_score": 0.92,
  "pattern_matched": "exchange_to_cashier"
}
```

#### Performance Considerations
- Hand tracking requires storing hand positions for 5-second window
- Memory usage: ~150 frames × 17 keypoints × 2-4 people = ~10KB per camera
- CPU overhead: Minimal (keypoint comparison only)
- GPU overhead: None (uses existing pose estimation)

#### Migration Path
- Existing single-zone detection remains default
- Dual-zone is opt-in per camera
- Backward compatible with current database
- No changes required for existing deployments



---

**Last Updated:** December 10, 2025
