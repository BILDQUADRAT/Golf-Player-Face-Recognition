# LPGA Player Face Recognition System

A real-time face recognition system for identifying LPGA golfers in video streams, updated for Python 3.13 with modern best practices.

## Features

- **Real-time face recognition** from video sources (webcam, video files, or streams)
- **Latest InsightFace models** (buffalo_l) for accurate face detection and recognition
- **Simplified output** - recognizes players and outputs their names with confidence scores
- **Web server integration ready** - prepared for sending data to external systems
- **Easy deployment** with virtual environment and requirements management
- **Python 3.13 compatible**

## Requirements

- Python 3.13 (or 3.10+)
- Webcam or video source
- ~2GB disk space for models and dataset

## Quick Start

### 1. Initial Setup

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```batch
setup.bat
```

**Or manually:**
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Dataset

First, you need to build a dataset of player images:

```bash
python cutlist_generator_v3.py --build_dataset
```

**Note:** The dataset building function is a placeholder. You'll need to add your own image source/download logic in the `build_dataset_with_progress()` method.

Alternatively, manually create the dataset structure:
```
dataset/
├── Player_Name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Player_Name_2/
│   └── ...
```

### 3. Generate Face Embeddings

Once you have images, generate the face embeddings database:

```bash
python cutlist_generator_v3.py --generate_embeddings
```

This creates `player_embeddings.pkl` containing facial features for each player.

### 4. Run Real-Time Recognition

**From webcam (default):**
```bash
python cutlist_generator_v3.py --recognize
```

**From specific video source:**
```bash
python cutlist_generator_v3.py --recognize --source 10
```

**With custom confidence threshold:**
```bash
python cutlist_generator_v3.py --recognize --source 0 --threshold 0.7
```

## Advanced Options

### Command Line Arguments

```
--build_dataset          Build player image dataset (requires implementation)
--generate_embeddings    Generate face embeddings from dataset
--recognize              Run real-time face recognition
--source INT             Video source ID (default: 0)
--threshold FLOAT        Recognition confidence threshold (default: 0.65)
--model NAME             InsightFace model (buffalo_l, buffalo_sc, buffalo_s)
--output_url URL         Web server URL for results (future feature)
```

### Model Selection

The system supports multiple InsightFace models:

- **buffalo_l** (default) - Balanced speed and accuracy, recommended
- **buffalo_sc** - More accurate but slower
- **buffalo_s** - Faster but less accurate

```bash
python cutlist_generator_v3.py --recognize --model buffalo_sc
```

### Adjusting Recognition Threshold

The threshold (0.0 to 1.0) determines how confident the system must be:

- **0.5-0.6**: More detections, but more false positives
- **0.65** (default): Good balance
- **0.7-0.8**: Fewer false positives, but might miss some matches

## Output Format

When a player is recognized, the system outputs:

```
[14:23:45] Recognized: Leona Maguire (confidence: 0.78)
```

## Web Server Integration (Coming Soon)

The system is prepared for web server integration. To enable:

1. Set up your web server endpoint
2. Use the `--output_url` parameter:

```bash
python cutlist_generator_v3.py --recognize --output_url http://your-server/api/player
```

The system will POST JSON data in this format:
```json
{
    "player": "Leona Maguire",
    "confidence": 0.78,
    "timestamp": "14:23:45"
}
```

## Project Structure

```
.
├── cutlist_generator_v3.py    # Main application
├── requirements.txt            # Python dependencies
├── setup.sh                    # Linux/Mac setup script
├── setup.bat                   # Windows setup script
├── README.md                   # This file
├── dataset/                    # Player images (you create)
├── player_embeddings.pkl       # Generated embeddings database
└── progress.json               # Dataset building progress
```

## Troubleshooting

### "Embeddings file not found"
Run `--generate_embeddings` after building your dataset.

### "Cannot open video source"
- Check your video source ID (try 0, 1, 2, etc.)
- Ensure webcam permissions are granted
- For video files, use the full path

### Low recognition accuracy
- Increase dataset size (more images per player)
- Adjust threshold value
- Use higher quality images
- Try the `buffalo_sc` model for better accuracy

### Performance issues
- Reduce processing FPS in the Config class
- Use `buffalo_s` model for faster processing
- Ensure GPU support is available (CUDA)

## Technical Details

### Models and Updates

This version uses:
- **InsightFace 0.7.3+** with buffalo_l model (latest stable)
- **Python 3.13 compatible** (uses modern type hints and dataclasses)
- **ONNX Runtime 1.18+** for efficient inference

### Key Changes from v2

1. ✅ Removed graphics overlay API dependency
2. ✅ Simplified output to console (with web server hooks)
3. ✅ Updated to latest InsightFace models
4. ✅ Python 3.13 compatibility
5. ✅ Better code organization with classes
6. ✅ Type hints for better IDE support
7. ✅ Virtual environment support
8. ✅ Cross-platform setup scripts

## Performance

On typical hardware:
- **CPU**: ~2-3 FPS processing
- **GPU (CUDA)**: ~10-15 FPS processing
- **Memory**: ~1-2GB during recognition

## License

[Your license here]

## Contributing

[Your contribution guidelines here]
