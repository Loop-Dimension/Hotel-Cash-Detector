# Model Examine - CCTV Detection Testing Tool

YOLO 모델 테스트용 독립 실행 도구입니다.

## 기능

- **Zone 설정**: 마우스로 Cashier Zone, Cash Drawer Zone 지정
- **2-Track 현금 감지**:
  - Track 1 (potential_cash): 손 접촉만으로 즉시 이벤트
  - Track 2 (cash): 손 접촉 → 박스 입금 확인
- **Violence 감지**: Bounding Box 겹침 기반
- **Fire 감지**: YOLO fire_smoke 모델 활용
- **Split View**: 왼쪽 원본 / 오른쪽 Detection Overlay
- **자동 클립 녹화**: 감지 시 앞뒤 6초씩 총 12초 클립 저장

## 사용법

### 1. Zone 설정 후 스트리밍

```bash
# 비디오 파일로 Zone 설정
python -m cctv.model_examine.examine_stream --video test.mp4 --setup-zones

# RTSP 스트림으로 Zone 설정
python -m cctv.model_examine.examine_stream --rtsp "rtsp://admin:pass@192.168.1.100:554/stream1" --setup-zones
```

### 2. 기존 Zone 설정으로 스트리밍

```bash
# zones.json 파일 사용
python -m cctv.model_examine.examine_stream --video test.mp4 --zone-config zones.json

# RTSP 스트림
python -m cctv.model_examine.examine_stream --rtsp "rtsp://..." --zone-config my_zones.json
```

### 3. 특정 감지만 활성화

```bash
# 현금 감지만
python -m cctv.model_examine.examine_stream --video test.mp4 --no-violence --no-fire

# 폭력 감지만
python -m cctv.model_examine.examine_stream --video test.mp4 --no-cash --no-fire
```

### 4. 클립 저장 경로 지정

```bash
python -m cctv.model_examine.examine_stream --video test.mp4 --output ./my_clips
```

## 키보드 단축키

| 키 | 기능 |
|---|------|
| Q | 종료 |
| P | 일시정지/재개 |
| Z | Zone 설정 모드 |
| D | Cash 감지 토글 |

## Zone 설정 모드

| 키 | 기능 |
|---|------|
| 좌클릭 | 꼭짓점 추가 |
| 우클릭 | 마지막 꼭짓점 삭제 |
| TAB | Cashier Zone ↔ Drawer Zone 전환 |
| S | 저장 후 종료 |
| R | 모든 점 초기화 |
| Q | 저장 없이 종료 |

## Zone 설정 파일 형식

```json
{
  "cashier_zone": [[100, 100], [300, 100], [300, 300], [100, 300]],
  "cash_drawer_zone": [[150, 200], [250, 200], [250, 280], [150, 280]]
}
```

## 클립 저장 위치

기본 경로: `cctv/model_examine_record/`

파일 형식: `YYYYMMDD_HHMMSS_이벤트타입.mp4`

예시:
- `20240115_143022_potential_cash.mp4`
- `20240115_143156_cash.mp4`
- `20240115_144012_violence.mp4`

## Interactive Mode (Option Select)

```bash
# Run without --rtsp/--video to pick input source
python -m cctv.model_examine.examine_stream
```

## Cooldown Options (seconds)

```bash
python -m cctv.model_examine.examine_stream --cash-cooldown 10 --cash-predict-cooldown 6 --violence-cooldown 15 --fire-cooldown 8
```
