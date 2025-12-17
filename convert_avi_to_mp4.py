"""
Convert AVI videos to MP4 format using FFmpeg
Usage: python convert_avi_to_mp4.py <input_file.avi> [output_file.mp4]
"""
import sys
import subprocess
from pathlib import Path

def convert_avi_to_mp4(input_path, output_path=None):
    """
    Convert AVI video to MP4 format
    
    Args:
        input_path: Path to input AVI file
        output_path: Path to output MP4 file (optional, defaults to same name with .mp4)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if not input_path.suffix.lower() in ['.avi', '.AVI']:
        print(f"Warning: Input file doesn't have .avi extension: {input_path}")
    
    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.with_suffix('.mp4')
    else:
        output_path = Path(output_path)
    
    # Make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting: {input_path}")
    print(f"Output: {output_path}")
    
    # FFmpeg command for high-quality MP4 conversion
    # -i: input file
    # -c:v libx264: use H.264 video codec
    # -preset fast: encoding speed preset (fast/medium/slow)
    # -crf 23: quality (lower = better, 18-28 is reasonable, 23 is default)
    # -c:a aac: use AAC audio codec
    # -b:a 128k: audio bitrate
    command = [
        'ffmpeg',
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-y',  # Overwrite output file if exists
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
        print(f"✓ Conversion successful!")
        print(f"Output file: {output_path}")
        print(f"Size: {output_path.stat().st_size / (1024*1024):.2f} MB")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion:")
        if e.stderr:
            print(e.stderr)
        return False
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg:")
        print("  Windows: choco install ffmpeg")
        print("  Or download from: https://ffmpeg.org/download.html")
        return False

def convert_directory(input_dir, output_dir=None, recursive=False):
    """
    Convert all AVI files in a directory
    
    Args:
        input_dir: Directory containing AVI files
        output_dir: Output directory (optional, defaults to same directory)
        recursive: Search subdirectories (default: False)
    """
    input_dir = Path(input_dir)
    
    if not input_dir.is_dir():
        print(f"Error: Not a directory: {input_dir}")
        return
    
    # Find all AVI files
    if recursive:
        avi_files = list(input_dir.rglob('*.avi')) + list(input_dir.rglob('*.AVI'))
    else:
        avi_files = list(input_dir.glob('*.avi')) + list(input_dir.glob('*.AVI'))
    
    # Also find other common video formats
    video_extensions = ['*.mov', '*.MOV', '*.mkv', '*.MKV', '*.wmv', '*.WMV', '*.flv', '*.FLV']
    for ext in video_extensions:
        if recursive:
            avi_files.extend(input_dir.rglob(ext))
        else:
            avi_files.extend(input_dir.glob(ext))
    
    if not avi_files:
        print(f"No video files found in: {input_dir}")
        return
    
    print(f"Found {len(avi_files)} video file(s)")
    print()
    
    success_count = 0
    failed_count = 0
    
    for i, video_file in enumerate(avi_files, 1):
        print(f"[{i}/{len(avi_files)}] Processing: {video_file.name}")
        
        if output_dir:
            output_path = Path(output_dir) / video_file.with_suffix('.mp4').name
        else:
            output_path = video_file.with_suffix('.mp4')
        
        # Skip if MP4 already exists
        if output_path.exists():
            print(f"  ⊙ Skipping - MP4 already exists")
            print()
            continue
        
        if convert_avi_to_mp4(video_file, output_path):
            success_count += 1
        else:
            failed_count += 1
        print()
    
    print(f"=" * 60)
    print(f"Conversion complete!")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total: {len(avi_files)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python convert_avi_to_mp4.py <input.avi> [output.mp4]")
        print("  Directory:   python convert_avi_to_mp4.py <directory> [-r]")
        print()
        print("Options:")
        print("  -r    Recursive - search subdirectories")
        print()
        print("Examples:")
        print('  python convert_avi_to_mp4.py video.avi')
        print('  python convert_avi_to_mp4.py video.avi converted.mp4')
        print('  python convert_avi_to_mp4.py media/uploads/test/')
        print('  python convert_avi_to_mp4.py testing/ -r')
        sys.exit(1)
    
    input_arg = sys.argv[1]
    input_path = Path(input_arg)
    
    # Check if it's a directory or file
    if input_path.is_dir():
        # Check for recursive flag
        recursive = '-r' in sys.argv or '--recursive' in sys.argv
        output_dir = None
        
        # Check if second arg is output dir (not a flag)
        if len(sys.argv) > 2 and not sys.argv[2].startswith('-'):
            output_dir = sys.argv[2]
        
        convert_directory(input_path, output_dir, recursive)
    else:
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        convert_avi_to_mp4(input_path, output_path)
