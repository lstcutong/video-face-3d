from VideoFace3D.utils.Global import *
import torch
from VideoFace3D.landmark_detect import ddfa_mobilenet as mobilenet_v1
import dlib
from VideoFace3D.models.ddfa_predict import *
import sys
from scipy import io


class FaceShapePredict():
    def __init__(self, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.__load_3D_static_data_file__()
        self.front_face_index = io.loadmat(FRONT_FACE_INDEX_PATH)["idx"][:, 0].astype(np.int) - 1

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

    def predict_shape(self, image_path, rects=None):
        vertices, color = detect_landmark_ddfa_3D_shape(image_path, self.model, self.face_regressor, self.device,
                                                        rects=rects)

        return vertices[:, self.front_face_index, :], color[:, self.front_face_index, :]
