import torch.nn as nn
import torch
from MorphableModelFitting.utils.geometry import *
from MorphableModelFitting.renderer.render import Renderer
from MorphableModelFitting.models.face_model import *
import cv2
import copy
from MorphableModelFitting.landmark_detect.detector import FaceLandmarkDetector
from MorphableModelFitting.utils.geometry import alignment_and_crop, estimate_affine_matrix_3d22d, matrix2angle, P2sRt
import sys
import tensorflow as tf
import numpy as np


def to_standard_tensor(p, device):
    if not torch.is_tensor(p):
        if isinstance(p, np.ndarray):
            return torch.from_numpy(p).float().to(device)
        else:
            assert Exception("unknown input type")
    else:
        return p.float().to(device)


class FaceWeakProjectionOptimizer(nn.Module):
    def __init__(self, init_pose, image_size, cuda=True):
        '''
        :param init_pose: [batch, 7]
        '''
        super(FaceWeakProjectionOptimizer, self).__init__()
        self.batch = len(init_pose)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.pose = nn.Parameter(torch.Tensor(init_pose))
        self.image_size = image_size
        self.renderer = Renderer(image_size=image_size, camera_mode="weak_projection")

    def forward(self, vertices):
        '''
        :param vertices: [batch, 68, 3]
        :return:
        '''
        assert len(vertices) == self.batch, "batch not equal!"
        scale = self.pose[:, 6].float().unsqueeze(1).unsqueeze(2)
        euler = self.pose[:, 0:3].float()
        trans = self.pose[:, 3:6].float().unsqueeze(1)

        K = torch.Tensor.repeat(torch.eye(3).unsqueeze(0), (self.batch, 1, 1)).to(self.device) * scale
        rot_mat = euler2rot(euler)

        project_points = self.renderer(vertices, mode="points", K=K, R=rot_mat, t=trans)
        return project_points


class FaceShapeOptimizer(nn.Module):
    def __init__(self, id_init, exp_init, cuda=True):
        super(FaceShapeOptimizer, self).__init__()
        self.batch = len(id_init)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.idi = nn.Parameter(torch.Tensor(id_init)).float()
        self.exp = nn.Parameter(torch.Tensor(exp_init)).float()

        self.facemodel = FaceModelBFM()

    def forward(self, x):
        return self.facemodel.shape_formation(self.idi, self.exp), self.idi, self.exp


class FaceLightOptimizer_SH(nn.Module):
    pass


class FaceLightOptimizer_phong(nn.Module):
    def __init__(self, init_direction, init_ambient, cuda=True):
        super(FaceLightOptimizer_phong, self).__init__()
        self.batch = len(init_direction)
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.direction = nn.Parameter(torch.Tensor(init_direction)).float()
        self.ambient = nn.Parameter(torch.Tensor(init_ambient)).float()
        self.facemodel = FaceModelBFM()

    def forward(self, s, R, t, idi, ex, tex, image_size):
        '''

        :param s: [batch,1]
        :param R: [batch,3]
        :param t: [batch,3]
        :param id: [batch,80]
        :param ex: [batch,64]
        :param tex: [batch,80]
        :return:
        '''
        scale = to_standard_tensor(s, self.device).unsqueeze(1)
        euler = to_standard_tensor(R, self.device)
        trans = to_standard_tensor(t, self.device).unsqueeze(1)
        ident = to_standard_tensor(idi, self.device)
        expre = to_standard_tensor(ex, self.device)
        textu = to_standard_tensor(tex, self.device)

        K = torch.Tensor.repeat(torch.eye(3).unsqueeze(0), (self.batch, 1, 1)).to(self.device) * scale
        rot_mat = euler2rot(euler)

        renderer = Renderer(image_size=image_size, K=K, R=rot_mat, t=trans, near=0.1, far=10000,
                            light_direction=self.direction, light_color_ambient=self.ambient)
        shape = self.facemodel.shape_formation(ident, expre)
        texture = self.facemodel.texture_formation(textu)
        triangles = torch.Tensor.repeat((torch.from_numpy(self.facemodel.tri) - 1).long().unsqueeze(0),
                                        (self.batch, 1, 1)).to(
            self.device)

        texture = texture_from_point2faces(triangles, texture) / 255

        rgb, depth, silh = renderer(shape, triangles, texture)
        return rgb


class FaceTextureOptimizer(nn.Module):
    def __init__(self, gamma_init, tex_init):
        super(FaceTextureOptimizer, self).__init__()
        pass


class FittingPipline():
    def __init__(self, cuda):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")


class FaceFittingPipline(FittingPipline):
    def __init__(self, type="accurate",
                 project_mode="weak_projection",
                 cuda=True,
                 show_mid=False,
                 checking=False):
        '''

        :param image_path: list for all image to annot
        :param type: 'accurate' or 'fast'
        :param project_mode: 'weak_projection' only for this version
        :param cuda:
        '''
        super(FaceFittingPipline, self).__init__(cuda)
        self.cuda = cuda
        self.show_mid = show_mid
        self.checking = checking
        # self.image_path = image_path
        self.type = type
        if not self.type in ['accurate', 'fast']:
            raise Exception("choose from 'accurate' or 'fast'")

        self.project_mode = project_mode
        self.facemodel = FaceModelBFM()
        self.crop_size = 224

        with tf.Graph().as_default() as self.graph:
            self.graph_def = self.__load_graph__(SINGLE_IMAGE_RECON_MODEL_PATH)
            self.sess = tf.Session()
            self.tf_images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
            tf.import_graph_def(self.graph_def, name='resnet', input_map={'input_imgs:0': self.tf_images})
            self.coeff = self.graph.get_tensor_by_name('resnet/coeff:0')

    def __check_kp68__(self, kp_gt, kp_align, im_size):
        batch = len(kp_gt)
        result = []
        for j in range(batch):
            img = np.ones((im_size, im_size, 3)) * 255
            img = img.astype(np.uint8)
            img_gt = copy.deepcopy(img)
            img_align = copy.deepcopy(img)
            for i in range(68):
                cv2.putText(img_gt, str(i), (int(kp_gt[j, i, 0]) - 5, int(kp_gt[j, i, 1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (255, 0, 0), thickness=1)
                cv2.circle(img_gt, (int(kp_gt[j, i, 0]), int(kp_gt[j, i, 1])), 1, (0, 0, 255))

                cv2.putText(img_align, str(i), (int(kp_align[j, i, 0]) - 5, int(kp_align[j, i, 1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 0, 0), thickness=1)
                cv2.circle(img_align, (int(kp_align[j, i, 0]), int(kp_align[j, i, 1])), 1, (0, 0, 255))
            result.append(np.column_stack([img_gt, img_align]))

        return np.row_stack(result)

    def __pose_fitting__(self, init_pose, gt_shape, gt_kp):
        tri = torch.Tensor(self.facemodel.keypoints).long().to(self.device)
        gt_shape = torch.Tensor(gt_shape).float().to(self.device)[:, tri, :]
        gt_kp = torch.Tensor(gt_kp).float().to(self.device)

        pose_opti = FaceWeakProjectionOptimizer(init_pose, self.crop_size, cuda=self.cuda)
        if self.cuda:
            pose_opti = pose_opti.cuda()

        optimizer = torch.optim.Adam(pose_opti.parameters(), lr=1)
        criterion = nn.MSELoss()

        loss_last = 99999999
        max_iteration = 3000
        iteration = 0
        coverage = False
        while not coverage:
            iteration += 1
            align_kp = pose_opti(gt_shape)[:, :, 0:2]
            loss = criterion(align_kp, gt_kp)

            if abs(loss.item() - loss_last) < 1e-3:
                coverage = True

            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_last = loss.item()
                if self.show_mid:
                    print("pose opti iter {} loss : {:.6f}".format(iteration, loss.item()))

            if iteration > max_iteration:
                break

            if self.checking:
                check_im = self.__check_kp68__(gt_kp.cpu().detach().numpy(), align_kp.cpu().detach().numpy(), 224)
                cv2.imwrite("./check.png", check_im)

        rts = pose_opti.pose.cpu().detach().numpy()
        r, t, s = rts[:, 0:3], rts[:, 3:6], rts[:, 6]
        if s.ndim == 1:
            s = np.expand_dims(s, axis=1)

        return r, t, s

    def __shape_fiiting__(self, init_id, init_exp, gt_kp, r, t, s):
        '''

        :param init_id:
        :param init_exp:
        :param gt_kp:
        :param r: [batch, 3]
        :param t: [batch, 3]
        :param s: [batch, ]
        :return:
        '''
        tri = torch.Tensor(self.facemodel.keypoints).long().to(self.device)
        gt_kp = torch.Tensor(gt_kp).float().to(self.device)

        init_pose = np.column_stack([r, t, s])

        shape_opti = FaceShapeOptimizer(init_id, init_exp, cuda=self.cuda)
        pose_opti = FaceWeakProjectionOptimizer(init_pose, self.crop_size, self.cuda)
        if self.cuda:
            shape_opti = shape_opti.cuda()
            pose_opti = pose_opti.cuda()

        optimizer = torch.optim.Adam(shape_opti.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        loss_last = 99999999
        max_iteration = 300
        iteration = 0
        coverage = False
        while not coverage:
            morph_shape, _idi, _exp = shape_opti(None)
            align_kp = pose_opti(morph_shape[:, tri, :])[:, :, 0:2]
            loss_reg = torch.sum(_idi ** 2) + torch.sum(_exp ** 2)
            loss_kp = criterion(align_kp, gt_kp)
            loss = loss_kp + 1e-2 * loss_reg

            if abs(loss_kp.item() - loss_last) < 1e-3:
                coverage = True

            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_last = loss.item()
                if self.show_mid:
                    print("{} shape opti loss_kp : {:.6f} loss_reg : {:.6f}".format(iteration, loss_kp.item(),
                                                                                    loss_reg.item()))

            iteration += 1
            if iteration > max_iteration:
                break

        idi, exp = shape_opti.idi.cpu().detach().numpy(), shape_opti.exp.cpu().detach().numpy()

        return idi, exp

    def __texture_fitting__(self):
        pass

    def __light_fitting__(self, reference_images, s, R, t, idi, ex, tex, mode="SH"):
        batch = len(reference_images)
        image_size = reference_images[0].shape[0]

        if mode == "SH":
            o_mizer = FaceLightOptimizer_SH()
        elif mode == "phong":
            init_direction = np.repeat(np.array([[0.0, 0.0, -1.0]]), batch, 0)
            init_ambient = np.repeat(np.array([[0.5, 0.5, 0.5]]), batch, 0)
            o_mizer = FaceLightOptimizer_phong(init_direction, init_ambient, self.cuda)
        else:
            return None

        if self.cuda:
            o_mizer = o_mizer.cuda()

        ref_img = torch.from_numpy(reference_images[:,:,:,::-1].transpose((0, 3, 1, 2)) / 255).to(self.device).float()

        optimizer = torch.optim.Adam(o_mizer.parameters(), lr=1e-2)
        criterion = nn.MSELoss()

        loss_last = 99999999
        max_iteration = 100
        iteration = 0
        coverage = False
        while not coverage:
            iteration += 1
            ren_im = o_mizer(s, R, t, idi, ex, tex, image_size)

            loss = criterion(ren_im, ref_img)

            #if abs(loss.item() - loss_last) < 1e-3:
            #    coverage = True

            #else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_last = loss.item()
            if self.show_mid:
                print("light opti iter {} loss : {:.6f}".format(iteration, loss.item()))

            if iteration > max_iteration:
                break

            if self.checking:
                check_im = ren_im.cpu().detach().numpy().transpose((0, 2, 3, 1))[:,:,:,::-1]

                ref_im = reference_images.reshape((-1, image_size, 3))
                check_im = (check_im.reshape((-1, image_size, 3)) * 255).clip(0, 255).astype(np.uint8)

                show_im = np.column_stack([ref_im, check_im])

                cv2.imshow("check", show_im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if mode == "SH":
            res = None
        elif mode == "phong":
            res = {"direction": o_mizer.direction.cpu().detach().numpy(),
                   "ambient": o_mizer.ambient.cpu().detach().numpy()}

        return res

    def __load_graph__(self, graph_filename):
        with tf.gfile.GFile(graph_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        return graph_def

    def __Split_coeff__(self, coeff):
        id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
        ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
        tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
        angles = coeff[:, 224:227]  # ruler angles(x,y,z) for rotation of dim 3
        gamma = coeff[:, 227:254]  # lighting coeff for 3 channel SH function of dim 27
        translation = coeff[:, 254:]  # translation coeff of dim 3

        return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation

    def __cnn_regression__(self, input_img):
        '''
        with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
            images = tf.placeholder(name='input_imgs', shape=[None, 224, 224, 3], dtype=tf.float32)
            graph_def = self.__load_graph__(SINGLE_IMAGE_RECON_MODEL_PATH)
            

            # output coefficients of R-Net (dim = 257)
            coeff = graph.get_tensor_by_name('resnet/coeff:0')

            #with tf.Session() as sess:
            coef = self.sess.run(coeff, feed_dict={images: input_img})
        '''
        coef = self.sess.run(self.coeff, feed_dict={self.tf_images: input_img})

        idi, ex, tex, angles, gamma, trans = self.__Split_coeff__(coef)
        return idi, ex, tex, angles, gamma, trans

    def __fast_fitting__(self, input_img, lm_inputs):
        keypoints_idx = torch.from_numpy(self.facemodel.keypoints).long() - 1
        init_idi, init_exp, tex, angles, gamma, trans = self.__cnn_regression__(input_img)

        shape = self.facemodel.shape_formation(torch.from_numpy(init_idi), torch.from_numpy(init_exp))

        # angles[:, 0] = -angles[:, 0]
        angles[:, 1] = angles[:, 1] + np.pi  # pitch, yaw, roll

        angles = angles % (2 * np.pi)
        # np.set_printoptions(suppress=True)
        # print(rad2degree(angles))

        s_s, eular_s, t_s = [], [], []
        for i in range(len(shape)):
            kp3d = shape[i][keypoints_idx, :].cpu().numpy()
            kp2d = lm_inputs[i]
            P = estimate_affine_matrix_3d22d(kp3d, kp2d)
            s, R, t = P2sRt(P)
            euler = matrix2angle(R)
            s_s.append(np.array([s]))
            eular_s.append(np.array([-euler[0], euler[1], euler[2] + np.pi]))
            t_s.append(np.array([t[0] / s, t[1] / s, 0]))

        # return init_idi, init_exp, tex, gamma, np.array(s_s), np.array(eular_s), np.array(t_s)
        return init_idi, init_exp, tex, gamma, np.array(s_s), np.array(angles), np.array(t_s)

    def start_fiiting(self, image_path, landmark_detect):
        import time
        # landmark_detect = FaceLandmarkDetector("3D")
        all_result = []

        for im_path in image_path:
            # print("process {}".format(im_path))
            t0 = time.time()
            ldmarks = landmark_detect.detect_face_landmark(im_path)
            t1 = time.time()

            single_batch = len(ldmarks)
            if single_batch == 0:
                print("process {} detect 0 face, detect time:{:.2f}s, annot time:0s".format(im_path, t1 - t0))
                continue

            img_inputs = []
            lm_inputs = []
            for i in range(single_batch):
                img, lm68 = alignment_and_crop(im_path, ldmarks[i])
                img_inputs.append(img)
                lm_inputs.append(lm68)

            img_inputs = np.row_stack(img_inputs)
            lm_inputs = np.row_stack(lm_inputs)

            t2 = time.time()
            if self.type == "accurate":
                idi, ex, tex, gamma, s, angles, trans = self.__fast_fitting__(img_inputs, lm_inputs)

                init_pose = np.column_stack([angles, trans, s])
                shape = self.facemodel.shape_formation(torch.from_numpy(idi), torch.from_numpy(ex))
                opt_r, opt_t, opt_s = self.__pose_fitting__(init_pose, shape, lm_inputs)

                opt_idi, opt_exp = self.__shape_fiiting__(idi, ex, lm_inputs, opt_r, opt_t, opt_s)

                all_result.append((img_inputs, lm_inputs,
                                   {"id": opt_idi, "exp": opt_exp, "tex": tex, "r": opt_r, "t": opt_t, "s": opt_s,
                                    "gamma": gamma}))

                # return

            if self.type == "fast":
                idi, ex, tex, gamma, s, angles, trans = self.__fast_fitting__(img_inputs, lm_inputs)

                all_result.append((img_inputs, lm_inputs,
                                   {"id": idi, "exp": ex, "tex": tex, "r": angles, "t": trans, "gamma": gamma, "s": s}))
                # return {"id": idi, "exp": ex, "tex": tex, "r": opt_r, "t": opt_t, "gamma": gamma, "s": opt_s}
            t3 = time.time()
            print("process {} detect {} face, detect time:{:.2f}s, annot time:{:.2f}s".format(im_path, single_batch,
                                                                                              t1 - t0, t3 - t2))

        return all_result

    def start_fitting_224(self, image_path, landmark_path):
        import time

        assert len(image_path) == len(landmark_path), "image list length != landmark list length"
        all_result = []

        for i in range(len(image_path)):
            img = cv2.imread(image_path[i])
            assert img.shape[0] == 224 and img.shape[1] == 224, "image size is not (224,224,3)"

            ldmarks = np.load(landmark_path[i])

            img_inputs = np.array([img])
            lm_inputs = np.array([ldmarks])

            t2 = time.time()
            if self.type == "accurate":
                idi, ex, tex, gamma, s, angles, trans = self.__fast_fitting__(img_inputs, lm_inputs)

                init_pose = np.column_stack([angles, trans, s])
                shape = self.facemodel.shape_formation(torch.from_numpy(idi), torch.from_numpy(ex))
                opt_r, opt_t, opt_s = self.__pose_fitting__(init_pose, shape, lm_inputs)

                opt_idi, opt_exp = self.__shape_fiiting__(idi, ex, lm_inputs, opt_r, opt_t, opt_s)

                all_result.append((img_inputs, lm_inputs,
                                   {"id": opt_idi, "exp": opt_exp, "tex": tex, "r": opt_r, "t": opt_t, "s": opt_s,
                                    "gamma": gamma}))

            if self.type == "fast":
                idi, ex, tex, gamma, s, angles, trans = self.__fast_fitting__(img_inputs, lm_inputs)

                all_result.append((img_inputs, lm_inputs,
                                   {"id": idi, "exp": ex, "tex": tex, "r": angles, "t": trans, "gamma": gamma, "s": s}))
                # return {"id": idi, "exp": ex, "tex": tex, "r": opt_r, "t": opt_t, "gamma": gamma, "s": opt_s}
            t3 = time.time()
            print("process {} annot time:{:.2f}s".format(image_path[i], t3 - t2))

        return all_result

    def start_fitting_light_224(self, image_path, s, R, t, idi, ex, tex, mode="phong"):
        reference_imgs = []
        for i in range(len(image_path)):
            img = cv2.imread(image_path[i])
            assert img.shape[0] == 224 and img.shape[1] == 224, "image size is not (224,224,3)"

            reference_imgs.append(img)

        reference_imgs = np.array(reference_imgs)

        return self.__light_fitting__(reference_imgs, s, R, t, idi, ex, tex, mode=mode)
