

# 🤖 Real-Time Object Detection with MediaPipe and OpenCV (Python)

This Python script implements a real-time object detection pipeline using `mediapipe` for high-performance detection and `opencv` for webcam capture and visualization.

The application captures frames from your webcam, runs object detection using the lightweight **EfficientDet-Lite** model, and draws bounding boxes and labels directly onto the video feed.

## ✨ Features

  * **Real-Time Processing:** Captures and processes video from the default webcam (`cv2.VideoCapture(0)`).
  * **MediaPipe Integration:** Uses the MediaPipe Tasks Vision API for efficient object detection.
  * **Lightweight Model:** Utilizes the `efficientdet_lite0.tflite` model, suitable for CPU-based real-time inference.
  * **Visual Feedback:** Draws bounding boxes, category labels, and confidence scores directly on the detected objects.
  * **Console Output:** Prints the detection results (label and score) to the console for monitoring.

## 🛠️ Prerequisites

To run this script, you need to have Python installed along with the following libraries:

1.  `numpy`
2.  `opencv-python` (cv2)
3.  `mediapipe`

### Installation

You can install the necessary dependencies using `pip`:

```bash
pip install numpy opencv-python mediapipe
```

### Model File

You must have the TensorFlow Lite model file in the same directory as your Python script.

  * **Model Name:** `efficientdet_lite0.tflite`
  * **Source:** This model is typically downloaded automatically when setting up MediaPipe examples, or you can find it in the official TensorFlow/MediaPipe model repositories.

## 🚀 How to Run

1.  **Save the Files:**

      * Save the provided Python code as a file named `object_detector.py`.
      * Ensure the `efficientdet_lite0.tflite` model file is in the same directory.

2.  **Execute the Script:**
    Open your terminal or command prompt, navigate to the directory where you saved the files, and run the script:

    ```bash
    python object_detector.py
    ```

3.  **View Output:**
    Two windows will appear:

      * `Frame` (This window will quickly disappear or show the unannotated frame, as it's immediately overwritten in the loop, this can be removed or simplified).
      * `abc` (This is the primary output window showing the live webcam feed with **real-time object detection annotations**).

4.  **Exit:**
    Press the **`ESC`** key to gracefully exit the application and close all windows.

## 💻 Code Overview

### 1\. Visualization Function

The `visualize` function handles the drawing logic using `cv2`:

```python
def visualize(image, detection_result) -> np.ndarray:
    # ... logic to iterate through detections ...
    # Draws a green rectangle around the object
    cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3) 
    # Adds the label and score text (e.g., 'person (0.95)')
    cv2.putText(image, result_text, text_location, ...) 
    # Prints to console
    print(result_text)
    return image
```

### 2\. Detector Initialization

The `ObjectDetector` is set up using the provided TFLite model:

```python
base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5) # Minimum confidence to show a detection
detector = vision.ObjectDetector.create_from_options(options)
```

### 3\. Main Loop

The main loop performs the real-time processing:

```python
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # 1. Save frame to disk (a temporary and inefficient step)
    cv2.imwrite('image.jpg', frame)
    
    # 2. Load the image using MediaPipe's format
    image = mp.Image.create_from_file("image.jpg")

    # 3. Perform the detection
    detection_result = detector.detect(image)

    # 4. Annotate and display
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    # Convert from RGB (MediaPipe) back to BGR (OpenCV for display)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("abc", rgb_annotated_image)
    
    # Exit condition
    if cv2.waitKey(1) & 0xFF == 27:
        break
```

-----

## 💡 Note on Efficiency

The current implementation writes the frame to disk (`image.jpg`) and immediately reads it back for MediaPipe processing in every single frame.

For a more robust and faster real-time application, the frame conversion should be done directly in memory without involving disk I/O, typically using `mp.Image.create_from_numpy()`:

```python
# To improve performance, replace this block:
# cv2.imwrite('image.jpg', frame)
# image = mp.Image.create_from_file("image.jpg")

# with this in-memory conversion:
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
```
