# Human Pose Estimation and Fall Detection with YOLO-Pose

## 🤖 Overview

This project implements a real-time human pose estimation and fall detection system by utilizing the state-of-the-art **YOLOv11-Pose** model. It processes a video stream to identify individuals, track their 17 key body joints, and analyze the orientation of their torso to classify their state as **Steady**, **Fallen**, or **Unknown**.

The system incorporates robust logic:
1.  **Slope Analysis:** The primary fall detection is based on the slope of the line connecting the center of the upper body (shoulders) and the lower body (hips/knees). A slope close to zero indicates a horizontal posture (Fallen state).
2.  **Inverted Posture Check:** A dedicated check identifies and flags an inverted or upside-down posture as a `Fallen` state.
3.  **Unknown State:** The state is classified as **Unknown** (indicated by a gray bounding box) if critical joint points needed for reliable slope calculation are missing, preventing unreliable analysis.

## ✨ Features

* **Human Pose Estimation:** Uses `yolo11x-pose.pt` to accurately detect 17 key points on the human body.
* **Fall Detection Logic:**
    * **Slope-based:** Detects falls when the vertical alignment (slope) of the torso line falls below a threshold (absolute slope $< 1.0$).
    * **Inverted Posture:** Detects when the knees or ankles are vertically above the shoulders/hips.
* **Robust State Classification and Visuals:**
    * **Steady** (Green Box) 🟢
    * **Fallen** (Red Box) 🔴: Triggered by either the slope or inverted posture check.
    * **Unknown** (Gray Box) ⚪: Triggered if both shoulder joints OR both hip joints are not visible.
* **Visual Alerts:** When a fallen state is detected, a cropped, highlighted alert window (with a red border) is displayed in the corner of the video.
* **Video Output:** Generates an annotated video file (`output/accidentpeople01_output.mp4`) showing the results.

## 📺 Demonstration

The GIFs below illustrate the system's ability to track multiple people and accurately detect falls, highlighting the "Fallen" status in red and triggering the alert window.

![System Demonstration](output\accidentpeople01_output-ezgif.com-optimize.gif)

![System Demonstration](output\accidentpeople02_output-ezgif.com-optimize.gif)


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
    The script is configured to use the video file named `accidentpeople01.mp4`. Ensure this file is located in the same directory as the Python script, or update the `VIDEO_PATH` variable in the script.

## 🚀 How to Run

Execute the Python script:

```bash
python YOLO_Falling_Detection.py