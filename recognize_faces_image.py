import cv2
import argparse
import pickle
import face_recognition
import numpy as np

# Argument parser setup
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-d", "--detection-method", type=str, default="hog",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# Load encodings and image
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
image = cv2.imread(args["image"])
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

print("[INFO] recognizing faces...")
boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
encodings = face_recognition.face_encodings(rgb, boxes)

names = []
for encoding in encodings:
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"
    if True in matches:
        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}
        for i in matchedIdxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1
        name = max(counts, key=counts.get)
    names.append(name)

# Draw boxes and names
for ((top, right, bottom, left), name) in zip(boxes, names):
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15
    cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# ---- Zoom and Pan Setup ---- #
zoom_scale = 1.0
pan_x, pan_y = 0, 0
is_dragging = False
start_x, start_y = 0, 0

def update_display():
    """Update the displayed image with zoom and pan."""
    global zoom_scale, pan_x, pan_y
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Zoom and crop
    cropped = cv2.resize(image, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)
    ch, cw = cropped.shape[:2]
    x1 = max(0, center_x - pan_x)
    y1 = max(0, center_y - pan_y)
    x2 = min(cw, x1 + w)
    y2 = min(ch, y1 + h)

    display_img = cropped[y1:y2, x1:x2]
    cv2.imshow("Image", display_img)

def mouse_events(event, x, y, flags, param):
    """Handle zoom, pan, and drag actions."""
    global zoom_scale, pan_x, pan_y, is_dragging, start_x, start_y

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll Up
            zoom_scale = min(zoom_scale * 1.1, 10.0)
        else:  # Scroll Down
            zoom_scale = max(zoom_scale / 1.1, 0.1)
        update_display()

    elif event == cv2.EVENT_LBUTTONDOWN:  # Start Dragging
        is_dragging = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:  # Drag
        dx, dy = x - start_x, y - start_y
        pan_x += dx
        pan_y += dy
        start_x, start_y = x, y
        update_display()

    elif event == cv2.EVENT_LBUTTONUP:  # Stop Dragging
        is_dragging = False

# ---- Display Image with Zoom and Pan ---- #
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_events)
update_display()

# Exit with Esc key
while True:
    if cv2.waitKey(1) & 0xFF == 27:  # Esc key
        break

cv2.destroyAllWindows()
