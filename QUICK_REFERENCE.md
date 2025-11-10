# Quick Reference - LPGA Face Recognition

## Setup (One Time)
```bash
# Linux/Mac
./setup.sh
source venv/bin/activate

# Windows
setup.bat
venv\Scripts\activate.bat
```

## Common Commands

### Generate Embeddings
```bash
python golf_player_face_recognition.py --generate_embeddings
```

### Run Recognition
```bash
# Webcam (default)
python golf_player_face_recognition.py --recognize

# Specific source
python golf_player_face_recognition.py --recognize --source 10

# With threshold
python golf_player_face_recognition.py --recognize --source 10 --threshold 0.7
```

### Change Model
```bash
# Balanced (default)
python golf_player_face_recognition.py --recognize --model buffalo_l

# More accurate
python golf_player_face_recognition.py --recognize --model buffalo_sc

# Faster
python golf_player_face_recognition.py --recognize --model buffalo_s
```

### Help
```bash
python golf_player_face_recognition.py --help
```

## Output Format
```
[14:23:45] Recognized: Leona Maguire (confidence: 0.78)
```

## File Structure
```
project/
├── golf_player_face_recognition.py    # Main script
├── requirements.txt            # Dependencies
├── setup.sh / setup.bat        # Setup scripts
├── dataset/                    # Player images
│   └── Player_Name/
│       ├── image1.jpg
│       └── image2.jpg
└── player_embeddings.pkl       # Generated database
```

## Thresholds
- **0.50-0.60**: More detections, more false positives
- **0.65**: Default, balanced
- **0.70-0.80**: Fewer false positives, stricter

## Troubleshooting
```bash
# Embeddings not found
python golf_player_face_recognition.py --generate_embeddings

# Can't open video source
python golf_player_face_recognition.py --recognize --source 0  # Try 0, 1, 2...

# Check Python version
python --version  # Should be 3.10+

# Reinstall dependencies
pip install -r requirements.txt
```

## Performance Tips
- Use GPU (CUDA) for faster processing
- Lower `processing_fps` in code for real-time performance
- Use `buffalo_s` model for speed
- Use `buffalo_sc` model for accuracy

## Web Server Integration (Future)
```bash
python golf_player_face_recognition.py --recognize --output_url http://localhost:8000/api
```

Expected POST format:
```json
{
    "player": "Player Name",
    "confidence": 0.78,
    "timestamp": "14:23:45"
}
```
