import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
import time

video = cv2.VideoCapture(0)
blazeface_tf = tf.keras.models.load_model("./keras_models/blazeface_tf.h5")
facemesh_tf = tf.keras.models.load_model("./keras_models/facemesh_tf.h5")
mappings = open("landmark_contours.txt").readlines()
contours = {}
for line in mappings:
    line = line.strip().split(" ")
    contours[line[0]] = [int(i) for i in line[1:]]

    
def predict_frame(orig_frame):
    orig_h, orig_w = orig_frame.shape[0:2]
    frame = create_letterbox_image(orig_frame, 128)
    h,w = frame.shape[0:2]
    input_frame = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(input_frame.astype(np.float32), 0) / 127.5 - 1
    result = blazeface_tf.predict(input_tensor)[0]
    final_boxes, landmarks_proposals = process_detections(result,(orig_h, orig_w),5, 0.75, 0.5, pad_ratio=0.5)
    if len(final_boxes) == 0: return orig_frame
    landmarks_input = get_landmarks_crop(orig_frame, landmarks_proposals, (192, 192))
    landmarks_result = facemesh_tf.predict(landmarks_input)[:, :-1] 
    final_landmarks = process_landmarks(landmarks_result, landmarks_proposals, (orig_h, orig_w), 192)
    assert len(final_boxes) == len(final_landmarks)
    for (bx,land) in zip(final_boxes, final_landmarks):
        cv2.rectangle(orig_frame, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 255), 1)
        cv2.drawContours(orig_frame, land[np.newaxis, contours["face"], :], -1, (255, 0, 0), thickness=2)
        for pt in land:
            cv2.circle(orig_frame, (pt[0], pt[1]), 1, (0, 0, 255), -1)
    return orig_frame
    
while True:
    ret, frame = video.read() 
    if ret:
        cv2.imshow("image", predict_frame(frame))
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
video.release()
cv2.destroyAllWindows()