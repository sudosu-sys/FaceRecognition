
# Facial Recognition System with Python 🧠📸

## Overview
This project is a Python-based facial recognition system that can:
- Train itself on images.
- Recognize faces in **images**, **live-stream videos**, and **pre-recorded videos**.

Built with **OpenCV**, **Dlib**, and **deep learning**, it supports GPU acceleration for faster processing.

---

## Features
- **Image Recognition:** Identify faces in static images.
- **Live Video Stream Recognition:** Process and recognize faces in real-time using webcam input.
- **Pre-Recorded Video Recognition:** Detect faces in video files.
- **Optimized for GPU:** Supports CUDA for faster performance.

---

## Requirements
1. Python 3.x
2. Libraries:
   - `opencv-python`
   - `imutils`
   - `face-recognition`
   - `pickle`
   - `numpy`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Encode Faces
```bash
python encode_faces.py -i <path_to_dataset> -e encodings.pickle -d cnn
```
- **`-i`**: Path to the directory containing training images.
- **`-e`**: Path to save the serialized encodings.
- **`-d`**: Face detection model (`hog` or `cnn`).

---

### 2. Recognize Faces in Images
```bash
python recognize_faces_image.py -e encodings.pickle -i <path_to_image> -d cnn
```
- **`-e`**: Path to encodings file.
- **`-i`**: Path to input image.
- **`-d`**: Detection model (`hog` or `cnn`).

---

### 3. Recognize Faces in Live Video Stream
```bash
python recognize_faces_video.py -e encodings.pickle -d cnn -y 1
```
- **`-e`**: Path to encodings file.
- **`-d`**: Detection model (`hog` or `cnn`).
- **`-y`**: Display output frame (1 to show, 0 to hide).

---

### 4. Recognize Faces in Pre-Recorded Videos
```bash
python recognize_faces_video_file.py -e encodings.pickle -v <path_to_video> -o output.avi -d cnn -y 1
```
- **`-e`**: Path to encodings file.
- **`-v`**: Path to input video file.
- **`-o`**: Path to save the output video.
- **`-d`**: Detection model (`hog` or `cnn`).
- **`-y`**: Display output frame (1 to show, 0 to hide).

---

## Notes
- Test images/videos should be **different from the training dataset** for unbiased results.
- Use **CNN** detection for better accuracy (requires GPU).
- Press **'q'** to quit while processing videos.

---

## Future Plans
- Expand functionality to support mobile applications.
- Optimize performance for larger datasets.
- Add real-time alerts and logging systems.

---

## Contributions
Pull requests are welcome! Feel free to open an issue for feedback or suggestions.

---

## License
This project is licensed under the MIT License.
