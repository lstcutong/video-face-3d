import glob
import os
import numpy as np
import cv2
import torch

from VideoFace3D.SFS.SFSNet.functions import create_shading_recon
from VideoFace3D.SFS.SFSNet.mask import MaskGenerator
from VideoFace3D.SFS.SFSNet.model import SfSNet
from VideoFace3D.SFS.SFSNet.utils import convert
from VideoFace3D.utils.Global import SFSNET_PATH
from VideoFace3D.utils.geometry import *
from PIL import Image

class SfSPipline():
    def __init__(self, cuda=True):
        self.device = torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")


class FaceSfSPipline(SfSPipline):
    def __init__(self, cuda=True):
        super(FaceSfSPipline, self).__init__(cuda)

        self.net = SfSNet()
        self.net.eval()
        self.net.load_state_dict(torch.load(SFSNET_PATH))
        self.net.to(self.device)

        self.M = 128

    def disentangle(self, image, ldmark=None):
        '''

        :param image: [h,w,3] bgr
        :param ldmark: [68,2]
        :return: norm[224,224,3], albedo[224,224,3], light[27]
        '''

        if ldmark is not None:
            cv2.imwrite("./cache.png", image)
            im, kp = alignment_and_crop("./cache.png", ldmark)

            os.remove("./cache.png")
        else:
            im = np.array([image])

        im = Image.fromarray(im[0]).resize((self.M, self.M), Image.ANTIALIAS)
        #im = cv2.resize(im[0], (self.M, self.M))

        im = np.float32(im) / 255.0
        # from (128, 128, 3) to (1, 3, 128, 128)
        im = np.transpose(im, [2, 0, 1])
        im = np.expand_dims(im, 0)

        # get the normal, albedo and light parameter
        normal, albedo, light = self.net(torch.from_numpy(im).to(self.device))

        n_out = normal.cpu().detach().numpy()
        al_out = albedo.cpu().detach().numpy()
        light_out = light.cpu().detach().numpy()

        n_out = np.squeeze(n_out, 0)
        n_out = np.transpose(n_out, [1, 2, 0])
        # from [1, 3, 128, 128] to [128, 128, 3]
        al_out = np.squeeze(al_out, 0)
        al_out = np.transpose(al_out, [1, 2, 0])
        # from [1, 27] to [27, 1]
        light_out = np.transpose(light_out, [1, 0])

        n_out2 = n_out[:, :, (2, 1, 0)]
        n_out2 = 2 * n_out2 - 1
        nr = np.sqrt(np.sum(n_out2 ** 2, axis=2))  # nr=sqrt(sum(n_out2.^2,3))
        nr = np.expand_dims(nr, axis=2)
        n_out2 = n_out2 / np.repeat(nr, 3, axis=2)

        al_out2 = al_out[:, :, (2, 1, 0)]
        # Note: n_out2, al_out2, light_out is the actual output

        al_out2 = cv2.cvtColor(al_out2, cv2.COLOR_RGB2BGR)
        n_out2 = cv2.cvtColor(n_out2, cv2.COLOR_RGB2BGR)

        #al_out2 = cv2.resize(al_out2, (224,224))
        #n_out2 = cv2.resize(n_out2, (224,224))

        return n_out2, al_out2, light_out