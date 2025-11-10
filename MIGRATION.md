# Migration Guide: v2 → v3

## Summary of Changes

### 1. ✅ Project Setup (Virtual Environment)

**Before (v2):**
- No standardized setup
- Manual dependency installation
- System-wide Python packages

**After (v3):**
- Virtual environment support
- `requirements.txt` with pinned versions
- Setup scripts for Linux/Mac/Windows
- Isolated dependencies

**Action Required:**
```bash
# Run setup script
./setup.sh        # Linux/Mac
setup.bat         # Windows
```

---

### 2. ✅ Simplified Output (Removed Graphics API)

**Before (v2):**
```python
def grafik_overlay_api_call(name):
    response = requests.post(API_URL, json={'text': name})
```

**After (v3):**
```python
def output_recognition(self, player_name: str, confidence: float):
    timestamp = time.strftime("%H:%M:%S")
    output_msg = f"[{timestamp}] Recognized: {player_name} (confidence: {confidence:.2f})"
    print(output_msg)
    
    # Ready for web server integration:
    # if self.config.output_url:
    #     requests.post(self.config.output_url, json={...})
```

**Output format:**
```
[14:23:45] Recognized: Leona Maguire (confidence: 0.78)
```

**Action Required:**
- No API_URL needed anymore
- Use `--output_url` parameter when ready for web server

---

### 3. ✅ Latest Models (InsightFace)

**Before (v2):**
- Unspecified model version
- No model selection options

**After (v3):**
- **buffalo_l** (default) - Latest balanced model
- **buffalo_sc** - High accuracy model
- **buffalo_s** - Fast model
- ONNX Runtime 1.18+ for optimization

**Model Performance:**
| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| buffalo_s | ⚡⚡⚡ | ⭐⭐ | Real-time, low latency |
| buffalo_l | ⚡⚡ | ⭐⭐⭐ | **Recommended balance** |
| buffalo_sc | ⚡ | ⭐⭐⭐⭐ | Best accuracy |

**Action Required:**
- None, buffalo_l is used by default
- Optional: Try `--model buffalo_sc` for better accuracy

---

### 4. ✅ Python 3.13 Compatibility

**Before (v2):**
- Unclear Python version requirement
- No type hints
- Legacy code patterns

**After (v3):**
- Python 3.13 compatible (works with 3.10+)
- Modern type hints (`Optional`, `Tuple`, `Dict`)
- Dataclasses for configuration
- Pathlib instead of os.path

**Code Improvements:**
```python
# v2 style
def recognize_face(embedding):
    return player_name, score

# v3 style
def recognize_face(self, face_embedding: np.ndarray) -> Tuple[Optional[str], float]:
    return player_name, score
```

**Action Required:**
- Use Python 3.10 or higher
- Recommended: Python 3.13

---

## Function Mapping

| v2 Function | v3 Method | Changes |
|-------------|-----------|---------|
| `build_dataset_with_progress()` | `PlayerRecognitionSystem.build_dataset_with_progress()` | Now a class method |
| `generate_embeddings()` | `PlayerRecognitionSystem.generate_embeddings()` | Now a class method |
| `real_time_api_update()` | `PlayerRecognitionSystem.real_time_recognition()` | Renamed, simplified |
| `grafik_overlay_api_call()` | `PlayerRecognitionSystem.output_recognition()` | Replaced, no API |
| `process_frame_with_ocr()` | Removed | Optional feature, removed |

---

## Configuration Changes

**Before (v2):**
```python
# Hardcoded in script
DATASET_DIR = "dataset"
EMBEDDINGS_FILE = "player_embeddings.pkl"
API_URL = "http://..."
```

**After (v3):**
```python
# Config class with defaults
@dataclass
class Config:
    dataset_dir: str = "dataset"
    embeddings_file: str = "player_embeddings.pkl"
    model_name: str = "buffalo_l"
    threshold: float = 0.65
    output_url: Optional[str] = None
```

**Action Required:**
- Use command-line arguments or config file
- No hardcoded URLs in source code

---

## Command-Line Interface

**Before (v2):**
```python
if __name__ == "__main__":
    real_time_api_update(10, 0.65)  # Hardcoded
```

**After (v3):**
```bash
# Much more flexible
python cutlist_generator_v3.py --recognize --source 10 --threshold 0.65

# Or with help
python cutlist_generator_v3.py --help
```

---

## Dependencies Updated

| Package | v2 Version | v3 Version | Change |
|---------|-----------|-----------|--------|
| opencv-python | Unspecified | ≥4.10.0 | Latest stable |
| insightface | Unspecified | ≥0.7.3 | Latest stable |
| numpy | Unspecified | ≥1.26.0 | Python 3.13 compatible |
| onnxruntime | Missing | ≥1.18.0 | **Added** for model optimization |

---

## Migration Checklist

- [ ] Run setup script to create virtual environment
- [ ] Activate virtual environment
- [ ] Install dependencies from requirements.txt
- [ ] Copy your existing dataset (if any)
- [ ] Generate new embeddings with v3
- [ ] Test recognition with your video source
- [ ] Configure web server URL when ready (optional)

---

## Breaking Changes

⚠️ **These require attention:**

1. **API removed**: `grafik_overlay_api_call()` no longer exists
   - **Solution**: Use `--output_url` parameter for web integration

2. **OCR removed**: `process_frame_with_ocr()` removed
   - **Solution**: Re-implement if needed, or use separate OCR tool

3. **Main execution changed**: No longer hardcoded `real_time_api_update(10, 0.65)`
   - **Solution**: Use command-line arguments

4. **Class-based structure**: Functions are now methods
   - **Solution**: Use the PlayerRecognitionSystem class

---

## Quick Start (Migrating from v2)

```bash
# 1. Setup new environment
./setup.sh

# 2. Copy your old dataset (if you have one)
cp -r old_project/dataset ./dataset

# 3. Generate new embeddings
python cutlist_generator_v3.py --generate_embeddings

# 4. Run recognition (same video source as before)
python cutlist_generator_v3.py --recognize --source 10 --threshold 0.65

# 5. When ready for web server
python cutlist_generator_v3.py --recognize --source 10 --output_url http://your-server/api
```

---

## Questions?

If you need help with:
- **Web server integration**: Check the commented code in `output_recognition()`
- **Model selection**: See the README model comparison table
- **Performance tuning**: Adjust `processing_fps` in Config class

---

## Advantages of v3

✅ Modern Python (3.13 compatible)  
✅ Better code organization (classes, type hints)  
✅ Virtual environment isolation  
✅ Latest InsightFace models  
✅ Flexible CLI interface  
✅ Easier deployment  
✅ Simplified output  
✅ Ready for web server integration  
✅ Better error handling  
✅ Cross-platform support
