import argparse
import os
from time import time
import sys

#print(sys.path)
#import MorphableModelFitting as mmf

from MorphableModelFitting.face_track.align import detect_face
import cv2
import numpy as np
import tensorflow as tf
from MorphableModelFitting.face_track.lib.face_utils import judge_side_face
from MorphableModelFitting.face_track.lib.utils import Logger, mkdir
from MorphableModelFitting.utils.Global import project_dir
from MorphableModelFitting.face_track.src.sort import Sort
import copy


class FaceTracker():
    def __init__(self, scale_rate=1.0, detect_interval=1, face_score_threshold=0.85, margin=15):
        self.scale_rate = scale_rate
        self.detect_interval = detect_interval
        self.face_score_threshold = face_score_threshold
        self.margin = margin

        self.tracker = Sort()
        self.minsize = 40  # minimum size of face for mtcnn to detect
        self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        self.factor = 0.709  # scale factor

    def start_track(self, video_path):
        with tf.Graph().as_default():
            with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                  log_device_placement=False)) as sess:
                self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(sess, os.path.join(project_dir,
                                                                                              "../face_track/align"))

                cam = cv2.VideoCapture(video_path)
                c = 0
                all_result = []
                while True:
                    final_faces = []
                    addtional_attribute_list = []
                    ret, frame = cam.read()
                    if not ret:
                        break

                    frame = cv2.resize(frame, (0, 0), fx=self.scale_rate, fy=self.scale_rate)
                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if c % self.detect_interval == 0:
                        img_size = np.asarray(frame.shape)[0:2]
                        mtcnn_starttime = time()
                        faces, points = detect_face.detect_face(r_g_b_frame, self.minsize, self.pnet, self.rnet, self.onet,
                                                                self.threshold,
                                                                self.factor)
                        face_sums = faces.shape[0]
                        if face_sums > 0:
                            face_list = []
                            for i, item in enumerate(faces):
                                score = round(faces[i, 4], 6)
                                if score > self.face_score_threshold:
                                    det = np.squeeze(faces[i, 0:4])

                                    # face rectangle
                                    det[0] = np.maximum(det[0] - self.margin, 0)
                                    det[1] = np.maximum(det[1] - self.margin, 0)
                                    det[2] = np.minimum(det[2] + self.margin, img_size[1])
                                    det[3] = np.minimum(det[3] + self.margin, img_size[0])
                                    face_list.append(item)

                                    # face cropped
                                    bb = np.array(det, dtype=np.int32)

                                    # use 5 face landmarks  to judge the face is front or side
                                    squeeze_points = np.squeeze(points[:, i])
                                    tolist = squeeze_points.tolist()
                                    facial_landmarks = []
                                    for j in range(5):
                                        item = [tolist[j], tolist[(j + 5)]]
                                        facial_landmarks.append(item)

                                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                        np.array(facial_landmarks))

                                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                    item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                    addtional_attribute_list.append(item_list)

                            final_faces = np.array(face_list)

                    trackers = self.tracker.update(final_faces, img_size, None, addtional_attribute_list, self.detect_interval)

                    people = []
                    for d in trackers:
                        det = np.array([0,0,0,0])
                        d = d.astype(np.int32)
                        det[0] = np.maximum(d[0] - self.margin, 0)
                        det[1] = np.maximum(d[1] - self.margin, 0)
                        det[2] = np.minimum(d[2] + self.margin, img_size[1])
                        det[3] = np.minimum(d[3] + self.margin, img_size[0])
                        bb = np.array(det, dtype=np.int32)
                        people.append((bb, d[4]))

                    all_result.append((frame, people))

        return all_result
