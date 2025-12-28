# Human Pose Estimation and Fall Detection with YOLO-Pose

## 🤖 Overview

This project implements a real-time human pose estimation and fall detection system using the state-of-the-art YOLOv11-Pose model. It processes a video stream to identify individuals, track their key body joints, and analyze the orientation of their torso to classify their state as **Steady**, **Fallen**, or **Unknown**.

The system enhances robustness by categorizing a person's state as **Unknown** (indicated by a gray bounding box) if critical joint points (both shoulders or both hips) are not visible, preventing unreliable fall detection analysis.

## ✨ Features

* **Human Pose Estimation:** Uses YOLOv11-Pose to accurately detect 17 key points on the human body.
* **Fall Detection:** Analyzes the slope of the line connecting the center of the upper body (shoulders) and the lower body (hips/knees). A slope close to zero indicates a horizontal posture (Fallen state).
* **Robust State Classification:**
    * **Steady** (Green Box) 🟢
    * **Fallen** (Red Box) 🔴
    * **Unknown** (Gray Box) ⚪: Triggered if both shoulder joints OR both hip joints are obscured, making the fall detection logic unreliable.
* **Visual Alerts:** When a fallen state is detected, a cropped, highlighted alert window is displayed in the corner of the video.
* **Video Output:** Generates an annotated video file (`yolov11_pose_estimator_detected.mp4`) showing the results.

## 📺 Demonstration

The GIF below illustrates the system's ability to track multiple people and accurately detect falls, highlighting the "Fallen" status in red and triggering the alert window.

![System Demonstration](output\accidentpeople01_output-ezgif.com-optimize.gif)

![System Demonstration](output\accidentpeople02_output-ezgif.com-optimize.gif)


``

## 🛠️ Prerequisites

To run this code, you need Python and the following libraries:

* **Python 3.x**
* **Ultralytics YOLO:** For the pose detection model.
* **OpenCV:** For video processing and drawing.
* **imageio:** For writing the output video file.

## ⚙️ Installation

1.  **Clone the repository (or save the code):**
    ```bash
    git clone <your-repo-link>
    cd <your-repo-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install ultralytics opencv-python imageio
    ```

3.  **Download the YOLOv11-Pose Model:**
    The script uses `yolo11x-pose.pt`. This file will typically be downloaded automatically by the `ultralytics` library when the code runs, but you can download it manually for offline use.

4.  **Place the Input Video:**
    Ensure your video file is named `accidentpeople01.mp4` and is located in the same directory as the Python script, or update the `VIDEO_PATH` variable in the script.

## 🚀 How to Run

Execute the Python script:

```bash
python your_script_name.py
```