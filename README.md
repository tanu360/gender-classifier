<div align="center">
  <h1>AI Gender Classifier</h1>

  <h3>Batch image classification with DeepFace, threaded processing, and clean output folders</h3>

  <p>
    <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" /></a>
    <a href="https://github.com/serengil/deepface"><img alt="DeepFace" src="https://img.shields.io/badge/DeepFace-AI-8B5CF6?style=for-the-badge" /></a>
    <a href="https://www.tensorflow.org/"><img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" /></a>
    <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-4.13-22C55E?style=for-the-badge&logo=opencv&logoColor=white" /></a>
    <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-64748B?style=for-the-badge" /></a>
  </p>

  <p>
    <a href="#-features">Features</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-output-structure">Output Structure</a> •
    <a href="#-model-cache">Model Cache</a> •
    <a href="#-troubleshooting">Troubleshooting</a> •
    <a href="#-license">License</a>
  </p>
</div>

---

## 🌟 Overview

AI Gender Classifier is a Python CLI tool that scans a folder of images, uses **DeepFace** to predict visible gender presentation, and copies images into organized output folders.

It supports recursive folder scanning, image validation, DeepFace detector fallback, multi-threaded processing, category-prefixed filenames, and a colorful terminal summary.

> Important: this tool predicts model labels from image appearance. Do not use it for identity verification, access control, sensitive decisions, or assumptions about a person's actual gender identity.

---

## ✨ Features

- **DeepFace gender analysis** with `actions=["gender"]`
- **OpenCV detector first**, then **RetinaFace fallback**
- **Recursive image scanning** for nested folders
- **Skips `classified_images`** on reruns to prevent re-processing output
- **Thread-safe counters** for reliable parallel processing
- **Overwrite-safe filenames** like `male-1.jpg`, `female-1.jpg`
- **Existing output detection** so numbering continues after old files
- **Image validation** using Pillow before AI analysis
- **Original files preserved** because images are copied, not deleted
- **No-human and error buckets** for safer sorting
- **Colorful terminal output** with progress and final stats
- **Optional auto-install** of missing Python packages at runtime

---

## 🧱 Tech Stack

| Layer | Tech |
| ----- | ---- |
| Language | Python 3.11+ |
| AI model wrapper | DeepFace |
| ML runtime | TensorFlow, tf-keras |
| Face detector fallback | RetinaFace |
| Image handling | Pillow, OpenCV |
| Terminal UI | Colorama, Emoji |
| Parallelism | `ThreadPoolExecutor` |

---

## 🚀 Quick Start

### Requirements

- Python 3.11+
- pip
- Internet connection on first run for model downloads

### Install

```bash
git clone <your-repo-url>
cd gender-classifier
python3 -m pip install -r requirements.txt
```

The script can also auto-install missing packages when it starts, but installing from `requirements.txt` first is still the cleanest setup.

### Run

```bash
python3 main.py
```

Then enter the folder path that contains your images:

```txt
Enter the folder path containing images:
```

Confirm with:

```txt
y
```

---

## 📁 Output Structure

The script creates a `classified_images` folder inside your selected input folder:

```txt
your-image-folder/
  image-1.jpg
  image-2.png
  classified_images/
    male/
      male-1.jpg
      male-2.jpg
    female/
      female-1.jpg
      female-2.jpg
    no_human/
      no-human-1.jpg
    errors/
      error-1.jpg
```

Generated output files continue from existing numbers. For example, if `female-7.jpg` already exists, the next copied female image becomes `female-8.jpg`.

---

## 🧠 How It Works

```txt
Input folder
  |
  v
Find supported images
  |
  v
Skip classified_images output folder
  |
  v
Validate image with Pillow
  |
  v
DeepFace analyze gender
  |
  +--> man   -> classified_images/male/male-N.ext
  +--> woman -> classified_images/female/female-N.ext
  +--> none  -> classified_images/no_human/no-human-N.ext
  +--> error -> classified_images/errors/error-N.ext
```

Supported image formats:

```txt
.jpg, .jpeg, .png, .bmp, .tiff, .webp
```

---

## ⚙️ Configuration

You can edit these values in `main.py`:

```python
SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
MAX_WORKERS = 4
```

Filename prefixes are also configurable:

```python
CATEGORY_FILENAME_PREFIXES = {
    "male": "male",
    "female": "female",
    "no_human": "no-human",
    "error": "error",
}
```

---

## 📦 Model Cache

DeepFace downloads model weights into your home directory:

```txt
~/.deepface/weights
```

Common files include:

```txt
gender_model_weights.h5
retinaface.h5
age_model_weights.h5
```

If you delete these files, DeepFace will download them again on a future run.

---

## 📜 Commands

| Command | Description |
| ------- | ----------- |
| `python3 main.py` | Start the classifier |
| `python3 -m pip install -r requirements.txt` | Install dependencies |
| `python3 -m py_compile main.py` | Check Python syntax |

---

## 🧯 Troubleshooting

| Problem | Fix |
| ------- | --- |
| `python: command not found` | Use `python3 main.py` |
| Package import error | Run `python3 -m pip install -r requirements.txt` |
| First run is slow | DeepFace may be downloading model weights |
| No images found | Check supported extensions and folder path |
| Output images appear again on rerun | Make sure you are running the latest code; `classified_images` is skipped |
| TensorFlow warnings | Usually safe if classification still runs |

---

## 🔒 Notes

- Original images are preserved.
- Output files are copied into category folders.
- `classified_images` is ignored during input scanning.
- Model predictions can be wrong, especially for low-quality, stylized, occluded, or non-frontal images.
- Avoid using this tool for sensitive or high-stakes decisions.

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a focused feature branch.
3. Keep changes small and readable.
4. Run `python3 -m py_compile main.py`.
5. Open a pull request with a clear summary.

---

## 📄 License

MIT License. See [LICENSE](./LICENSE).
