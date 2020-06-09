from MorphableModelFitting.utils.Global import *
import torch
from MorphableModelFitting.landmark_detect import ddfa_mobilenet as mobilenet_v1
import dlib
from MorphableModelFitting.landmark_detect.ddfa_landmarks import detect_landmark_ddfa_3D
from MorphableModelFitting.landmark_detect.dlib_landmark import detect_landmark_dlib_2D
import sys


class LandmarkDetector():
    def __init__(self, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")


class FaceLandmarkDetector(LandmarkDetector):
    def __init__(self, mode="2D", cuda=True):
        super(FaceLandmarkDetector, self).__init__(cuda)
        self.mode = mode

        if self.mode == "2D":
            self.__load_2D_static_data_file__()
        elif self.mode == "3D":
            self.__load_3D_static_data_file__()
        else:
            raise Exception("Please choose between '2D' or '3D'")

    def __load_2D_static_data_file__(self):
        dlib_landmark_model = DLIB_LANDMARK_MODEL_PATH
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)

    def __load_3D_static_data_file__(self):
        checkpoint_fp = CHECKPOINT_FP_PATH

        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        self.model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = self.model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.model.load_state_dict(model_dict)

        self.model.to(self.device)

        dlib_landmark_model = DLIB_LANDMARK_MODEL_PATH
        self.face_regressor = dlib.shape_predictor(dlib_landmark_model)

    def detect_face_landmark(self, image_path, rects=None):
        if self.mode == "3D":
            landmarks = detect_landmark_ddfa_3D(image_path, self.model, self.face_regressor, self.device, rects=rects)
        else:
            landmarks = detect_landmark_dlib_2D(image_path, self.face_regressor)

        if len(landmarks) == 0:
            return []
        else:
            return landmarks
