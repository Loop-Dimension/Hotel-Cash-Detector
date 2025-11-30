"""
Video Processor for Hotel CCTV Detection System

Process videos from the input folder and generate detection reports.
Can be used for batch processing without the web interface.
"""
import cv2
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config import INPUT_DIR, OUTPUT_DIR, MODELS_DIR, DetectionConfig
from detectors import UnifiedDetector


class VideoProcessor:
    """
    Process video files for detection of cash transactions, violence, and fire.
    """
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.detector = None
        self.output_dir = OUTPUT_DIR
        
    def initialize(self, cashier_zone=None):
        """Initialize the detector"""
        detector_config = {
            'models_dir': str(MODELS_DIR),
            'cashier_zone': cashier_zone or DetectionConfig.DEFAULT_CASHIER_ZONE,
            'hand_touch_distance': DetectionConfig.HAND_TOUCH_DISTANCE,
            'pose_confidence': DetectionConfig.POSE_CONFIDENCE,
            'detect_cash': self.config.get('detect_cash', True),
            'detect_violence': self.config.get('detect_violence', True),
            'detect_fire': self.config.get('detect_fire', True)
        }
        
        self.detector = UnifiedDetector(detector_config)
        return self.detector.initialize()
    
    def load_camera_config(self, camera_folder: Path) -> dict:
        """Load camera-specific configuration"""
        config_file = camera_folder / 'config.json'
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def process_video(self, video_path: str, output_video: bool = True, 
                     show_preview: bool = False) -> dict:
        """
        Process a single video file
        
        Args:
            video_path: Path to the video file
            output_video: Whether to save the annotated video
            show_preview: Whether to show live preview window
            
        Returns:
            Detection report dictionary
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"❌ Video not found: {video_path}")
            return None
        
        print(f"\n{'=' * 60}")
        print(f"📹 Processing: {video_path.name}")
        print(f"{'=' * 60}")
        
        # Try to load camera config from parent folder
        camera_config = self.load_camera_config(video_path.parent)
        if camera_config.get('CASHIER_ZONE'):
            print(f"✅ Loaded cashier zone from config: {camera_config['CASHIER_ZONE']}")
            self.detector.set_cashier_zone(camera_config['CASHIER_ZONE'])
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📊 Video info: {width}x{height} @ {fps}fps, {total_frames} frames ({duration:.1f}s)")
        
        # Setup output video writer
        out_writer = None
        if output_video:
            self.output_dir.mkdir(exist_ok=True)
            output_path = self.output_dir / f"detected_{video_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"💾 Output video: {output_path}")
        
        # Process frames
        all_detections = []
        frame_skip = DetectionConfig.FRAME_SKIP
        
        print(f"\n🔍 Analyzing video (processing every {frame_skip} frames)...")
        
        pbar = tqdm(total=total_frames, desc="Processing", unit="frames")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            pbar.update(1)
            
            # Skip frames for performance
            if frame_count % frame_skip != 0:
                if out_writer:
                    out_writer.write(frame)
                continue
            
            # Process frame
            result = self.detector.process_frame(frame, draw_overlay=True)
            
            # Collect detections
            for det in result['detections']:
                det['frame_number'] = frame_count
                det['timestamp_seconds'] = frame_count / fps
                all_detections.append(det)
            
            # Show preview if enabled
            if show_preview:
                # Resize for display
                display = cv2.resize(result['frame'], (1280, 720))
                cv2.imshow('Processing', display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️ Processing stopped by user")
                    break
            
            # Write output frame
            if out_writer:
                out_writer.write(result['frame'])
        
        pbar.close()
        
        # Cleanup
        cap.release()
        if out_writer:
            out_writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Calculate processing stats
        elapsed = time.time() - start_time
        fps_achieved = frame_count / elapsed
        
        # Generate report
        report = self.generate_report(video_path, all_detections, {
            'duration_seconds': duration,
            'total_frames': total_frames,
            'processing_time': elapsed,
            'fps_achieved': fps_achieved
        })
        
        # Save report
        report_path = self.output_dir / f"report_{video_path.stem}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Report saved: {report_path}")
        
        # Print summary
        self.print_summary(report)
        
        return report
    
    def generate_report(self, video_path: Path, detections: list, stats: dict) -> dict:
        """Generate a detection report"""
        
        # Count by type
        by_type = {
            'CASH': [],
            'VIOLENCE': [],
            'FIRE': []
        }
        
        for det in detections:
            label = det.get('label', 'UNKNOWN')
            if label in by_type:
                by_type[label].append(det)
        
        report = {
            'video': {
                'name': video_path.name,
                'path': str(video_path),
                'duration_seconds': stats['duration_seconds'],
                'total_frames': stats['total_frames']
            },
            'processing': {
                'timestamp': datetime.now().isoformat(),
                'processing_time_seconds': stats['processing_time'],
                'fps_achieved': stats['fps_achieved']
            },
            'summary': {
                'total_detections': len(detections),
                'cash_transactions': len(by_type['CASH']),
                'violence_incidents': len(by_type['VIOLENCE']),
                'fire_alerts': len(by_type['FIRE'])
            },
            'detections': {
                'cash': by_type['CASH'],
                'violence': by_type['VIOLENCE'],
                'fire': by_type['FIRE']
            }
        }
        
        return report
    
    def print_summary(self, report: dict):
        """Print a summary of the detection report"""
        print(f"\n{'=' * 60}")
        print("📊 DETECTION SUMMARY")
        print(f"{'=' * 60}")
        
        summary = report['summary']
        
        print(f"\n🎬 Video: {report['video']['name']}")
        print(f"⏱️  Duration: {report['video']['duration_seconds']:.1f}s")
        print(f"⚡ Processing speed: {report['processing']['fps_achieved']:.1f} fps")
        
        print(f"\n📈 Detections:")
        print(f"   💵 Cash Transactions: {summary['cash_transactions']}")
        print(f"   ⚠️  Violence Incidents: {summary['violence_incidents']}")
        print(f"   🔥 Fire/Smoke Alerts: {summary['fire_alerts']}")
        print(f"   ───────────────────")
        print(f"   📊 Total: {summary['total_detections']}")
        
        # Print timeline of important events
        if summary['violence_incidents'] > 0 or summary['fire_alerts'] > 0:
            print(f"\n🚨 IMPORTANT EVENTS:")
            
            for det in report['detections']['violence']:
                t = det.get('timestamp_seconds', 0)
                print(f"   ⚠️  Violence at {t:.1f}s (confidence: {det['confidence']:.2f})")
            
            for det in report['detections']['fire']:
                t = det.get('timestamp_seconds', 0)
                print(f"   🔥 Fire at {t:.1f}s (confidence: {det['confidence']:.2f})")
        
        print(f"\n{'=' * 60}\n")
    
    def process_folder(self, folder_path: str, output_video: bool = True):
        """Process all videos in a folder"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Folder not found: {folder}")
            return []
        
        # Find all video files
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        videos = []
        for ext in video_extensions:
            videos.extend(folder.glob(ext))
        
        if not videos:
            print(f"❌ No videos found in: {folder}")
            return []
        
        print(f"\n📁 Found {len(videos)} videos in {folder}")
        
        reports = []
        for i, video in enumerate(videos, 1):
            print(f"\n[{i}/{len(videos)}] Processing: {video.name}")
            report = self.process_video(str(video), output_video)
            if report:
                reports.append(report)
        
        # Generate combined report
        if reports:
            combined_report = {
                'folder': str(folder),
                'processed_at': datetime.now().isoformat(),
                'total_videos': len(reports),
                'summary': {
                    'cash_transactions': sum(r['summary']['cash_transactions'] for r in reports),
                    'violence_incidents': sum(r['summary']['violence_incidents'] for r in reports),
                    'fire_alerts': sum(r['summary']['fire_alerts'] for r in reports)
                },
                'videos': [r['video']['name'] for r in reports]
            }
            
            combined_path = self.output_dir / f"combined_report_{folder.name}.json"
            with open(combined_path, 'w') as f:
                json.dump(combined_report, f, indent=2)
            
            print(f"\n📄 Combined report saved: {combined_path}")
        
        return reports


def main():
    """Main entry point for video processing"""
    parser = argparse.ArgumentParser(
        description='Hotel CCTV Detection System - Video Processor'
    )
    parser.add_argument('input', nargs='?', default=str(INPUT_DIR),
                       help='Video file or folder to process')
    parser.add_argument('--no-output', action='store_true',
                       help='Do not save annotated video')
    parser.add_argument('--preview', action='store_true',
                       help='Show live preview while processing')
    parser.add_argument('--no-cash', action='store_true',
                       help='Disable cash detection')
    parser.add_argument('--no-violence', action='store_true',
                       help='Disable violence detection')
    parser.add_argument('--no-fire', action='store_true',
                       help='Disable fire detection')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("🏨 HOTEL CCTV DETECTION SYSTEM")
    print("=" * 60)
    print("\nDetection Types:")
    print("  💵 Cash Transactions" + (" (disabled)" if args.no_cash else ""))
    print("  ⚠️  Violence" + (" (disabled)" if args.no_violence else ""))
    print("  🔥 Fire/Smoke" + (" (disabled)" if args.no_fire else ""))
    print("=" * 60)
    
    # Initialize processor
    processor = VideoProcessor({
        'detect_cash': not args.no_cash,
        'detect_violence': not args.no_violence,
        'detect_fire': not args.no_fire
    })
    
    if not processor.initialize():
        print("❌ Failed to initialize detector")
        return
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Process single video
        processor.process_video(str(input_path), 
                               output_video=not args.no_output,
                               show_preview=args.preview)
    elif input_path.is_dir():
        # Process folder
        processor.process_folder(str(input_path), 
                                output_video=not args.no_output)
    else:
        print(f"❌ Input not found: {input_path}")


if __name__ == '__main__':
    main()
