# Installing FFmpeg for Browser Video Playback

## Why FFmpeg?

FFmpeg is required to convert videos to browser-compatible MP4 format (H.264 codec).

**Without FFmpeg:**
- Videos saved as AVI (Motion JPEG)
- Won't play in browsers ❌
- Must download and use VLC

**With FFmpeg:**
- Videos automatically converted to MP4 ✅
- Plays directly in browsers
- H.264 codec (universal support)

---

## Installation

### Windows

**Option 1: Winget (Recommended)**
```bash
winget install ffmpeg
```

**Option 2: Chocolatey**
```bash
choco install ffmpeg
```

**Option 3: Manual Download**
1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
2. Extract to `C:\ffmpeg`
3. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" → Add: `C:\ffmpeg\bin`
4. Restart terminal

**Verify Installation:**
```bash
ffmpeg -version
```

---

### Linux

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**CentOS/RHEL:**
```bash
sudo yum install ffmpeg
```

---

### macOS

```bash
brew install ffmpeg
```

---

## Testing

After installing FFmpeg:

1. **Restart the Flask app:**
   ```bash
   python run_production.py
   ```

2. **Look for this message:**
   ```
   ✅ FFmpeg detected - will use for video conversion
   ```

3. **Upload a video**

4. **Check console output:**
   ```
   📎 Clip 1: 현금 거래 - video_CASH_EXCHANGE_5s.avi (85 frames)
   🔄 Converting to MP4...
   ✅ Converted to MP4
   📎 Clip 1: 현금 거래 - video_CASH_EXCHANGE_5s.mp4 (85 frames)
   ```

5. **Click "미리보기" (Preview)** → Video plays in browser! 🎬

---

## Troubleshooting

### FFmpeg not detected

**Check if in PATH:**
```bash
where ffmpeg     # Windows
which ffmpeg     # Linux/Mac
```

**If not found:**
- Make sure you restarted terminal after installation
- Check PATH environment variable
- Try absolute path to ffmpeg

### Conversion fails

**Check FFmpeg version:**
```bash
ffmpeg -version
```

**Minimum version:** 4.0+

**If conversion fails:**
- Videos will stay as .avi
- Download button will still work
- Open with VLC player

---

## Alternative: No FFmpeg

If you can't install FFmpeg:

1. Videos will be saved as `.avi` files
2. **Click "다운로드" (Download)** button
3. Open with **VLC Media Player** or similar
4. All annotations will still be visible

**VLC Download:** https://www.videolan.org/vlc/

---

## Summary

| Format | Browser | Download | Quality |
|--------|---------|----------|---------|
| MP4 (with FFmpeg) | ✅ Perfect | ✅ Works | High |
| AVI (no FFmpeg) | ❌ Doesn't play | ✅ Works | High |

**Recommendation:** Install FFmpeg for best experience! 🚀

