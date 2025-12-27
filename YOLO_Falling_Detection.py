import imageio
from ultralytics import YOLO
import cv2
import math
from ultralytics.utils.plotting import Annotator

VIDEO_PATH = "accidentpeople01.mp4"
dataset = cv2.VideoCapture(VIDEO_PATH)

if not dataset.isOpened():
    print(f"Error: Could not open video file {VIDEO_PATH}")
    exit()

model = YOLO("yolo11x-pose.pt")
draw_pose_flag = True
draw_boxes_flag = True

success, first_frame = dataset.read()
if not success:
    print("Error: Could not read first frame.")
    dataset.release()
    exit()

dataset.set(cv2.CAP_PROP_POS_FRAMES, 0)

h, w = first_frame.shape[:2]
TARGET_WIDTH = 1080
aspect_ratio = w / h
TARGET_HEIGHT = int(TARGET_WIDTH / aspect_ratio)
RESIZE_DIM = (TARGET_WIDTH, TARGET_HEIGHT)

try:
    writer = imageio.get_writer("output/yolov11_pose_estimator_detected.mp4", 
                                fps=dataset.get(cv2.CAP_PROP_FPS),
                                codec='libx264', quality=8)
except Exception as e:
    print(f"Error initializing video writer: {e}")
    dataset.release()
    exit()

frame_count = 0

def analyze_pose_and_draw_keypoints(image, keypoints_xy, keypoints_conf):
    
    joint_labels = {
        5: "L Shoulder", 6: "R Shoulder", 
        7: "L Elbow", 8: "R Elbow", 9: "L Wrist", 10: "R Wrist", 
        11: "L Hip", 12: "R Hip", 13: "L Knee", 14: "R Knee", 
        15: "L Ankle", 16: "R Ankle"
    }
    
    SHOULDER_JOINTS = [5, 6]
    HIP_JOINTS = [11, 12]
    KNEE_JOINTS = [13, 14]

    skeleton = [
        (5,6), (5,7), (7,9), (6,8), (8,10),
        (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)
    ]
    person_status_details = []

    for kpts, conf in zip(keypoints_xy, keypoints_conf):
        kpts = kpts.cpu().numpy()
        conf = conf.cpu().numpy()
        falling_flag = False
        slope_value = 0.0
        is_unknown = False
        
        P_A_source_indices = []
        P_B_source_indices = []

        for (start, end) in skeleton:
            if conf[start] > 0.7 and conf[end] > 0.7 and draw_pose_flag:
                pt1 = (int(kpts[start][0]), int(kpts[start][1]))
                pt2 = (int(kpts[end][0]), int(kpts[end][1]))
                cv2.line(image, pt1, pt2, (255, 100, 0), 2)

        is_both_shoulders_visible = conf[5] > 0.7 and conf[6] > 0.7
        is_both_hips_visible = conf[11] > 0.7 and conf[12] > 0.7
        
        if (not is_both_shoulders_visible) or (not is_both_hips_visible):
             is_unknown = True
        
        P_A = None
        
        if is_both_shoulders_visible:
            P_A = (kpts[5] + kpts[6]) / 2
            P_A_source_indices = [5, 6]
        else:
            visible_shoulders = [i for i in SHOULDER_JOINTS if conf[i] > 0.7]
            if visible_shoulders:
                P_A_idx = min(visible_shoulders, key=lambda i: kpts[i][1])
                P_A = kpts[P_A_idx]
                P_A_source_indices = [P_A_idx]

        is_both_knees_visible = conf[13] > 0.7 and conf[14] > 0.7
        P_B = None

        if is_both_knees_visible:
            P_B = (kpts[13] + kpts[14]) / 2
            P_B_source_indices = [13, 14]
        elif is_both_hips_visible:
            P_B = (kpts[11] + kpts[12]) / 2
            P_B_source_indices = [11, 12]
        else:
            visible_bottom_joints = [i for i in HIP_JOINTS + KNEE_JOINTS if conf[i] > 0.7]
            if visible_bottom_joints:
                P_B_idx = max(visible_bottom_joints, key=lambda i: kpts[i][1])
                P_B = kpts[P_B_idx]
                P_B_source_indices = [P_B_idx]


        if not is_unknown and P_A is not None and P_B is not None:
            
            dx = P_B[0] - P_A[0]
            dy = P_B[1] - P_A[1]

            if abs(dx) > 1:
                slope_value = dy / dx 
                
                if abs(slope_value) < 1.0:
                    falling_flag = True
                    
            if draw_pose_flag:
                pt_pA = (int(P_A[0]), int(P_A[1]))
                pt_pB = (int(P_B[0]), int(P_B[1]))
                
                line_color = (0, 0, 255) if falling_flag else (0, 255, 0)
                line_text = "FALLING DETECTED" if falling_flag else "Steady Posture"
                
                cv2.line(image, pt_pA, pt_pB, line_color, 4)
                cv2.circle(image, pt_pA, 6, (0, 255, 255), -1)
                cv2.circle(image, pt_pB, 6, (0, 255, 255), -1)
                
                cv2.putText(image, line_text, (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2, cv2.LINE_AA)
        
        analysis_joints = P_A_source_indices + P_B_source_indices
        
        for i, (x, y) in enumerate(kpts):
            if conf[i] > 0.7 and i in joint_labels:
                if draw_pose_flag:
                    if i in analysis_joints:
                        joint_color = (0, 0, 255)
                    else:
                        joint_color = (255, 0, 0)

                    cv2.circle(image, (int(x), int(y)), 5, joint_color, -1)
                    
                    label = joint_labels[i]
                    text_pos = (int(x) + 7, int(y) - 5)
                    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        person_status_details.append((falling_flag, slope_value, is_unknown))
    
    return person_status_details

def draw_boxes_with_annotator(annotator, boxes, person_status_details):
    
    for box_idx, box in enumerate(boxes):
        
        falling_flag, slope_value, is_unknown = person_status_details[box_idx] 
        
        if is_unknown:
            box_color = (128, 128, 128)
            status_label = "Unknown"
            slope_text = ""
        elif falling_flag:
            box_color = (0, 0, 255)
            status_label = "Fallen"
            slope_text = f" | Slope: {slope_value:.2f}"
        else:
            box_color = (0, 255, 0)
            status_label = "Steady"
            slope_text = f" | Slope: {slope_value:.2f}"
            
        main_label = status_label + slope_text
        
        annotator.box_label(box.xyxy[0], main_label, color=box_color)


def Falling_Alert_Window(clean_image, box):
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    cropped_image = clean_image[y1:y2, x1:x2].copy()
    
    if cropped_image.size != 0:
        
        cv2.rectangle(cropped_image, (0, 0), (cropped_image.shape[1], cropped_image.shape[0]), (0, 0, 255), 5)
        
        main_h, main_w = clean_image.shape[:2]
        target_h = main_h // 5
        
        crop_h, crop_w = cropped_image.shape[:2]
        aspect_ratio = crop_w / crop_h
        
        resized_h = target_h
        resized_w = int(target_h * aspect_ratio)
        
        resized_crop = cv2.resize(cropped_image, (resized_w, resized_h))
        
        pos_x = main_w - resized_w - 10
        pos_y = 10 
        
        if pos_x >= 0 and (pos_x + resized_w) <= main_w and pos_y >= 0 and (pos_y + resized_h) <= main_h:
            return resized_crop, pos_x, pos_y, resized_w, resized_h
        
    return None, None, None, None, None

while True:
    ret, frame = dataset.read()
    if not ret:
        break
    
    frame_count += 1
    
    frame = cv2.resize(frame, RESIZE_DIM)
    
    clean_frame = frame.copy() 
    
    results = model(frame, classes=[0], verbose=False)
    
    if results[0].boxes.xyxy.numel() == 0:
        person_status_details = []
        keypoints_xy = []
        keypoints_conf = []
    else:
        keypoints_xy = results[0].keypoints.xy
        keypoints_conf = results[0].keypoints.conf
        
    person_status_details = analyze_pose_and_draw_keypoints(frame, keypoints_xy, keypoints_conf)
    
    if draw_boxes_flag:
        line_thickness = 2
        annotator = Annotator(frame, line_width=line_thickness)
        
        draw_boxes_with_annotator(annotator, results[0].boxes, person_status_details)
        
        frame = annotator.im
        
        for box_idx, box in enumerate(results[0].boxes):
            falling_flag, slope_value, is_unknown = person_status_details[box_idx]
            
            if not is_unknown and falling_flag:
                resized_crop, pos_x, pos_y, resized_w, resized_h = Falling_Alert_Window(clean_frame, box)
                
                if resized_crop is not None:
                    frame[pos_y : pos_y + resized_h, pos_x : pos_x + resized_w] = resized_crop
                break

    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.imshow("Input", frame)
    
    key = cv2.waitKey(1)
    if key == 13 or key == 27:
        break

dataset.release()
writer.close()
cv2.destroyAllWindows()

print("Video processing complete and saved to output/yolov11_pose_estimator_detected.mp4")