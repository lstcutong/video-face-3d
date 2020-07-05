import torch
import numpy as np
from scipy.io import loadmat
from MorphableModelFitting.utils.Global import *

class FaceModelBFM():
    def __init__(self):
        self.model_path = BFM_FRONT_MODEL_PATH
        model = loadmat(self.model_path)
        self.meanshape = model['meanshape']  # mean face shape
        self.idBase = model['idBase']  # identity basis
        self.exBase = model['exBase']  # expression basis
        self.meantex = model['meantex']  # mean face texture
        self.texBase = model['texBase']  # texture basis
        # point_buf can be used for calculating each vertex's norm once face's norm is computed
        self.point_buf = model[
            'point_buf']  # adjacent face index for each vertex, starts from 1 (only used for calculating face normal)
        # tri can be used for calculating each face's norm
        self.tri = model['tri'][:,::-1].copy()  # vertex index for each triangle face, starts from 1
        self.keypoints = np.squeeze(model['keypoints']).astype(np.int32) - 1  # 68 face landmark index, starts from 0
        self.transform_keypoints_index()

    def shape_formation(self, id_param, ex_param):
        '''

        :param id_param: [batch, dim]
        :param ex_param: [batch, dim]
        :return:
        '''
        batch = len(id_param)

        idBase = torch.from_numpy(self.idBase).float().to(id_param.device)
        ms = torch.from_numpy(self.meanshape.reshape(-1)).float().to(id_param.device)
        exBase = torch.from_numpy(self.exBase).float().to(id_param.device)


        shape = ms.unsqueeze(1) + idBase @ id_param.float().transpose(1, 0) + exBase @ ex_param.float().transpose(1, 0)
        shape = shape.transpose(1, 0).reshape((batch, -1, 3))

        face_shape = shape - torch.mean(torch.reshape(ms, [1, -1, 3]), dim=1)

        return face_shape

    def texture_formation(self, tex_param):
        batch = len(tex_param)
        texBase = torch.from_numpy(self.texBase).float().to(tex_param.device)
        mt = torch.from_numpy(self.meantex.reshape(-1)).float().to(tex_param.device)

        tex = mt.unsqueeze(1) + texBase @ tex_param.transpose(1, 0)
        tex = tex.transpose(1, 0)

        tex = tex.reshape((batch, -1, 3))
        return tex

    # keypoint index in BFM_front_face model is not the same as defined in dlib
    # this function makes them be same
    # default using dlib index
    def transform_keypoints_index(self):
        kp_idx = self.keypoints.reshape(-1)
        coutour = list(kp_idx[0:17][::-1])
        nose_u = list(kp_idx[27:31])
        nose_d = list(kp_idx[31:36][::-1])

        eye_cor_order = np.array([45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40])
        mouse_cor_order = np.array([54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65])

        mouse = list(kp_idx[mouse_cor_order])
        eyes = list(kp_idx[eye_cor_order])
        eye_brow = list(kp_idx[17:27][::-1])
        kp_idx = coutour + eye_brow + nose_u + nose_d + eyes + mouse

        self.keypoints = np.array(kp_idx).reshape(-1).astype(np.int64)

    def get_triangle_and_kp68_index(self):
        kp_idx = self.keypoints

        return torch.from_numpy(self.tri.astype(np.int32) - 1).unsqueeze(0).cuda(), torch.from_numpy(
            kp_idx.astype(np.int64)).cuda()