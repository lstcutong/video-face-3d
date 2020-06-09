import dlib
import cv2
import numpy as np

def detect_landmark_dlib_2D(image_path, predictor):
    image = cv2.imread(image_path)
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    landmarks = []

    dets = detector(gray, 1)
    for face in dets:
        single_face = []
        shape = predictor(image, face)
        for pt in shape.parts():
            pt_pos = (pt.x, pt.y)

            single_face.append(np.array(pt_pos))

        landmarks.append(np.array(single_face))
    return landmarks