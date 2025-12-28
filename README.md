# 🤖 Real-Time Human Pose Estimation and Fall Detection with YOLO-Pose

## ⭐ Project Summary

This project delivers a highly robust system for real-time human pose estimation and automated fall detection using the state-of-the-art **YOLOv11-Pose** model. By analyzing the vertical alignment of a person's torso, the system can reliably classify the person's state as **Steady**, **Fallen**, or **Unknown**, providing immediate visual alerts for critical situations.

---

## 💡 Conceptual Overview: How Fall Detection Works

The core of the fall detection logic relies on analyzing the person's body posture relative to the ground plane, specifically by calculating the slope of the line connecting their upper and lower body centers.

1.  **Key Point Averaging:**
    * **Upper Center ($P_A$):** Calculated as the midpoint between the Left Shoulder (5) and Right Shoulder (6).
    * **Lower Center ($P_B$):** Calculated as the midpoint between the Left Knee (13) and Right Knee (14), or if knees are obscured, the midpoint between the Left Hip (11) and Right Hip (12).
2.  **Slope Calculation:** The slope ($m$) of the line segment $\overline{P_A P_B}$ is calculated as:
    $$m = \frac{P_B(y) - P_A(y)}{P_B(x) - P_A(x)}$$
    
3.  **Classification Threshold:**
    * **Steady:** A person standing or walking upright will have a very steep slope (large absolute value).
    * **Fallen:** A person lying horizontally will have a slope close to zero. The system classifies the posture as **Fallen** if the absolute slope is less than **1.0** ($|m| < 1.0$).
4.  **Inverted Posture Check:** A dedicated check flags the posture as **Fallen** if the lower body joints (knees or ankles) are vertically higher than the upper body joints (shoulders/hips).
5.  **Robustness Check:** If critical joints (both shoulders and both hips) are not visible, the posture is classified as **Unknown** to prevent false positives from unreliable data.

---

## ✨ Features & Visual Classification

| Status | Color | Description | Code Logic |
| :--- | :--- | :--- | :--- |
| **Steady** | Green Box 🟢 | The person is upright and moving normally. | $|m| \ge 1.0$ |
| **Fallen** | Red Box 🔴 | The person is lying down (horizontal) or in an inverted position. | $|m| < 1.0$ OR `is_inverted` is True |
| **Unknown** | Gray Box ⚪ | The pose data is incomplete (missing key joints) and cannot be reliably analyzed. | Insufficient visibility of shoulders or hips. |

* **Model:** Uses the powerful `yolo11x-pose.pt` for highly accurate 17-point keypoint detection.
* **Visual Alert:** A dynamic, cropped alert window of the fallen person is displayed in the corner of the output video upon fall detection.
* **Scalability:** Designed to track and analyze multiple individuals simultaneously.

## 📺 Demonstration

The GIF below illustrates the system's ability to track multiple people and accurately detect falls, highlighting the "Fallen" status in red and triggering the alert window.

![System Demonstration](output/accidentpeople01_output-ezgif.com-optimize.gif)

![System Demonstration](output/accidentpeople02_output-ezgif.com-optimize.gif)

---

## 🛠️ Prerequisites

To run this code, you need Python and the following libraries:

* **Python 3.x**
* **Ultralytics YOLO:** For the pose detection model and utilities.
* **OpenCV (`opencv-python`):** For video processing, frame manipulation, and drawing.
* **`imageio`:** For writing the high-quality output video file.

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

3.  **Model & Input Setup:**
    * The script automatically downloads `yolo11x-pose.pt` upon first execution.
    * Place your input video file, named `accidentpeople01.mp4`, in the project root directory.

## 🚀 How to Run

Execute the main Python script:

```bash
python YOLO_Falling_Detection.py
```

**Output:** The annotated video, `output/accidentpeople01_output.mp4`, will be saved after processing is complete.

---

## 🧑‍💻 Code Structure & Key Functions

The `YOLO_Falling_Detection.py` script is modularized with the following key functions:

* **`analyze_pose_and_draw_keypoints(image, keypoints_xy, keypoints_conf)`:**
* This is the primary function for fall detection.
* It calculates , , and the slope .
* It determines the person's status (`Fallen`, `Steady`, `Unknown`) and draws the skeleton and the torso line on the frame.


* **`check_inverted_posture(kpts, conf)`:**
* A specialized function that checks if lower body joints are above a certain threshold relative to the reference joints (shoulders/hips).


* **`draw_boxes_with_annotator(annotator, boxes, person_status_details)`:**
* Applies the appropriate bounding box color (Red, Green, or Gray) and status label based on the results from `analyze_pose_and_draw_keypoints`.


* **`Falling_Alert_Window(clean_image, box)`:**
* Responsible for cropping the detected fallen person from the original frame and resizing it to create the visual alert overlay.