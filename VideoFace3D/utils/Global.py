SMOOTH_METHODS_OPTIMIZE = 1
SMOOTH_METHODS_MEDUIM = 2
SMOOTH_METHODS_MEAN = 3
SMOOTH_METHODS_GAUSSIAN = 4
SMOOTH_METHODS_DCNN = 5

import os.path as osp
import os

def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

project_dir = os.path.dirname(os.path.abspath(__file__))

d = make_abs_path("../data")
#d = "/home/magic/lst/codes/MorphableFaceFitting/MorphableModel"

BFM_FRONT_MODEL_PATH             = osp.join(d, "BFM_model_front.mat")
SINGLE_IMAGE_RECON_MODEL_PATH    = osp.join(d, "FaceReconModel.pb")
SIMILARITY_LM3D_ALL_MODEL_PATH   = osp.join(d, "similarity_Lm3D_all.mat")
FRONT_FACE_INDEX_PATH            = osp.join(d, "BFM_front_idx.mat")
CHECKPOINT_FP_PATH               = osp.join(d, "phase1_wpdc_vdc.pth.tar")
DLIB_LANDMARK_MODEL_PATH         = osp.join(d, "shape_predictor_68_face_landmarks.dat")
TRI_PATH                         = osp.join(d, "tri.mat")
SFSNET_PATH                      = osp.join(d, "SfSNet.pth")
BISENET_MODEL_PATH               = osp.join(d, "bisenet.pth")