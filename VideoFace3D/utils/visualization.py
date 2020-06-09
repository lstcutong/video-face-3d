import cv2
import random
import numpy as np

def draw_landmarks(image, landmarks, plot_index=False):
    image1 = image.copy()
    for i in range(len(landmarks)):
        lm_num = len(landmarks[i])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for j in range(lm_num):
            x, y = int(landmarks[i][j][0]), int(landmarks[i][j][1])
            image1 = cv2.circle(image1, (x, y), radius=3, thickness=2, color=color)
            if plot_index:
                image1 = cv2.putText(image1, str(j), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            color,
                                            thickness=1)
    return image1



