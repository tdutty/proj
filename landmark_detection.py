import dlib
import cv2
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Define the indices for the facial landmarks corresponding to different regions
LEFT_FACE_POINTS = list(range(0, 17))  # Left face
RIGHT_FACE_POINTS = list(range(17, 27))  # Right face
TOP_HEAD_POINTS = [27, 28, 29, 30, 31, 32, 33, 34, 35]  # Top of the head
BACK_HEAD_POINTS = [8, 9, 10]  # Back of the head

# Function to detect facial landmarks
def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
    
    landmarks = predictor(gray, faces[0])
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
    return landmarks

# Function to check if a 360-degree view is detected
def detect_360_view(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    prev_landmarks = None
    complete_views = 0
    
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks = detect_landmarks(frame)
        
        if landmarks is not None:
            if prev_landmarks is not None:
                left_face_diff = np.linalg.norm(prev_landmarks[LEFT_FACE_POINTS] - landmarks[LEFT_FACE_POINTS])
                right_face_diff = np.linalg.norm(prev_landmarks[RIGHT_FACE_POINTS] - landmarks[RIGHT_FACE_POINTS])
                top_head_diff = np.linalg.norm(prev_landmarks[TOP_HEAD_POINTS] - landmarks[TOP_HEAD_POINTS])
                back_head_diff = np.linalg.norm(prev_landmarks[BACK_HEAD_POINTS] - landmarks[BACK_HEAD_POINTS])
                
                if left_face_diff < 100 and right_face_diff < 100 and top_head_diff < 100 and back_head_diff < 100:
                    complete_views += 1
            prev_landmarks = landmarks
    
    cap.release()
    
    completeness_ratio = complete_views / frame_count
    if completeness_ratio >= 0.9:  # You may need to adjust this threshold based on your data
        return True
    else:
        return False

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with the path to your input video file
    
    if detect_360_view(video_path):
        print("360-degree view of a person's head detected.")
    else:
        print("360-degree view of a person's head not detected.")
