from VideoFace3D.segmentation.faceparsing.model import BiSeNet
import torch
from VideoFace3D.utils.Global import BISENET_MODEL_PATH
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

class FaceSegmentation():
    def __init__(self, cuda=True):
        self.device = torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")
        self.model = BiSeNet(n_classes=19)
        self.model.load_state_dict(torch.load(BISENET_MODEL_PATH))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.stride = 1

        self.bg = 0
        self.skin = 1
        self.l_brow = 2
        self.r_brow = 3
        self.l_eye = 4
        self.r_eye = 5
        self.eye_g = 6
        self.l_ear = 7
        self.r_ear = 8
        self.ear_r = 9
        self.nose = 10
        self.mouth = 11
        self.u_lip = 12
        self.l_lip = 13
        self.neck = 14
        self.neck_l = 15
        self.cloth = 16
        self.hair = 17
        self.hat = 18

        self.skins = [self.skin]
        self.eyes = [self.l_brow, self.r_brow, self.l_eye, self.r_eye, self.eye_g]
        self.noses = [self.nose]
        self.mouths = [self.mouth, self.u_lip, self.l_lip]
        self.ears = [self.l_ear, self.r_ear, self.ear_r]
        self.necks = [self.neck, self.neck_l]
        self.cloths = [self.cloth]
        self.hairs = [self.hair]
        self.hats = [self.hat]

    def create_face_mask(self, image_path,
                         skin=True,
                         eye=True,
                         nose=True,
                         mouth=True,
                         ear=False,
                         neck=False,
                         cloth=False,
                         hair=False,
                         hat=False
                         ):
        '''

        :return: mask, mask_probability
        '''
        img = Image.open(image_path)
        org_w, org_h = img.size
        image = img.resize((512, 512), Image.BILINEAR)
        img = self.to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)
        out = self.model(img)[0].squeeze(0).cpu().detach().numpy()

        parsing_anno = out.argmax(0)

        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, (org_h, org_w), fx=self.stride, fy=self.stride, interpolation=cv2.INTER_NEAREST)
        org_parsing = vis_parsing_anno.copy()
        mask_prob = out - out.min(0)
        mask_prob_exp = np.exp(mask_prob)
        mask_prob = mask_prob_exp / mask_prob_exp.sum(0)
        mask_prob = mask_prob.max(0)

        mask_prob = cv2.resize(mask_prob, (org_h, org_w), fx=self.stride, fy=self.stride)

        mask = np.zeros((org_w, org_h))

        if skin:
            for p in self.skins:
                mask += vis_parsing_anno == p
        if eye:
            for p in self.eyes:
                mask += vis_parsing_anno == p
        if nose:
            for p in self.noses:
                mask += vis_parsing_anno == p
        if mouth:
            for p in self.mouths:
                mask += vis_parsing_anno == p
        if ear:
            for p in self.ears:
                mask += vis_parsing_anno == p
        if neck:
            for p in self.necks:
                mask += vis_parsing_anno == p
        if cloth:
            for p in self.cloths:
                mask += vis_parsing_anno == p
        if hair:
            for p in self.hairs:
                mask += vis_parsing_anno == p
        if hat:
            for p in self.hats:
                mask += vis_parsing_anno == p

        return org_parsing, mask, mask_prob * mask

    def visualize(self, parsing_anno, image_path):
        stride = 1
        part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]


        im = cv2.imread(image_path)
        vis_im = im.copy().astype(np.uint8)
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        # print(vis_parsing_anno_color.shape, vis_im.shape)
        vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        return vis_im