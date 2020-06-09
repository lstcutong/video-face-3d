from __future__ import division
import math

import torch
import torch.nn as nn
import numpy

import neural_renderer as nr
from MorphableModelFitting.renderer.weak_projection import weak_projection
from MorphableModelFitting.utils.geometry import texture_from_point2faces, euler2rot, compute_face_norm, compute_point_norm
from MorphableModelFitting.renderer.lightning import *
from MorphableModelFitting.models.face_model import FaceModelBFM

class Renderer(nn.Module):
    def __init__(self, image_size=256, anti_aliasing=True, background_color=[0, 0, 0],
                 fill_back=True, camera_mode='weak_projection',
                 K=None, R=None, t=None, dist_coeffs=None, orig_size=1024,
                 perspective=True, viewing_angle=30, camera_direction=[0, 0, 1],
                 near=0.1, far=100,light_mode="parallel",SH_Coeff=None,
                 light_intensity_ambient=0.5, light_intensity_directional=0.5,
                 light_color_ambient=[1, 1, 1], light_color_directional=[1, 1, 1],
                 light_direction=[0, 1, 0]):
        super(Renderer, self).__init__()
        # rendering
        self.image_size = image_size
        self.anti_aliasing = anti_aliasing
        self.background_color = background_color
        self.fill_back = fill_back

        self.light_mode= light_mode
        self.SH_coeff = SH_Coeff
        self.facemodel = FaceModelBFM()
        # camera
        self.camera_mode = camera_mode
        if self.camera_mode in ['projection', 'weak_projection']:
            self.K = K
            self.R = R
            self.t = t
            if isinstance(self.K, numpy.ndarray):
                self.K = torch.cuda.FloatTensor(self.K)
            if isinstance(self.R, numpy.ndarray):
                self.R = torch.cuda.FloatTensor(self.R)
            if isinstance(self.t, numpy.ndarray):
                self.t = torch.cuda.FloatTensor(self.t)
            self.dist_coeffs = dist_coeffs
            if dist_coeffs is None:
                self.dist_coeffs = torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]])
            self.orig_size = orig_size
        elif self.camera_mode in ['look', 'look_at']:
            self.perspective = perspective
            self.viewing_angle = viewing_angle
            self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
            self.camera_direction = [0, 0, 1]
        else:
            raise ValueError('Camera mode has to be one of projection, look or look_at')

        self.near = near
        self.far = far

        # light
        self.light_intensity_ambient = light_intensity_ambient
        self.light_intensity_directional = light_intensity_directional
        self.light_color_ambient = light_color_ambient
        self.light_color_directional = light_color_directional
        self.light_direction = light_direction

        # rasterization
        self.rasterizer_eps = 1e-3

    def forward(self, vertices, faces=None, textures=None, mode=None, K=None, R=None, t=None, dist_coeffs=None,
                orig_size=224):
        '''
        Implementation of forward rendering method
        The old API is preserved for back-compatibility with the Chainer implementation
        '''

        if mode is None:
            return self.render(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode is 'rgb':
            return self.render_rgb(vertices, faces, textures, K, R, t, dist_coeffs, orig_size)
        elif mode == 'silhouettes':
            return self.render_silhouettes(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'depth':
            return self.render_depth(vertices, faces, K, R, t, dist_coeffs, orig_size)
        elif mode == 'points':
            return self.project_points(vertices, K, R, t, dist_coeffs, orig_size)
        else:
            raise ValueError("mode should be one of None, 'silhouettes' or 'depth'")

    def render_silhouettes(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'weak_projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t

            vertices = weak_projection(vertices, K, R, t, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_silhouettes(faces, self.image_size, self.anti_aliasing)
        return images

    def render_depth(self, vertices, faces, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):

        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'weak_projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t

            vertices = weak_projection(vertices, K, R, t, orig_size)
        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render_rgb(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        if self.fill_back:
            faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
            textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)

        # lighting
        faces_lighting = nr.vertices_to_faces(vertices, faces)
        textures = nr.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'weak_projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t

            vertices = weak_projection(vertices, K, R, t, orig_size)

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        images = nr.rasterize(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return images

    def render(self, vertices, faces, textures, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        # fill back
        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'weak_projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t

            vertices = weak_projection(vertices, K, R, t, orig_size)
            # 正则化一下，让脸颊到鼻尖的向量和z轴负方向夹角小于90度 (a,b,c)*(0,0,-1)>0 -> c<0
            # 另外将z值全部归到正轴上去
            from MorphableModelFitting.models.face_model import FaceModelBFM
            front_idx, back_idx_1, back_idx_2 = FaceModelBFM().keypoints[30], FaceModelBFM().keypoints[0], FaceModelBFM().keypoints[16]
            for i in range(len(vertices)):
                back_z = (vertices[i, back_idx_1, 2] + vertices[i, back_idx_2, 2]) / 2
                if (vertices[i, front_idx, 2] - back_z) > 0:
                    vertices[i, :, 2] = -vertices[i, :, 2]
            vertices[:, :, 2] = vertices[:, :, 2] - torch.min(vertices[:, :, 2], dim=1)[0].unsqueeze(1) + 1

        # lighting
        if self.light_mode == "parallel":
            textures = texture_from_point2faces(faces, textures) / 255
            if self.fill_back:
                faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
            faces_lighting = nr.vertices_to_faces(vertices, faces)
            textures = nr.lighting(
                faces_lighting,
                textures,
                self.light_intensity_ambient,
                self.light_intensity_directional,
                self.light_color_ambient,
                self.light_color_directional,
                self.light_direction)
        elif self.light_mode == "SH":
            point_buf = torch.from_numpy(self.facemodel.point_buf).long() - 1
            point_norm = compute_point_norm(vertices, faces[0][:, list(reversed(range(faces.shape[-1])))], point_buf)

            # texture = texture_from_point2faces(triangles, texture).reshape((batch, -1, 3))
            textures, _ = Illumination_SH(textures, point_norm, self.SH_coeff)
            textures = texture_from_point2faces(faces, textures) / 255
            if self.fill_back:
                faces = torch.cat((faces, faces[:, :, list(reversed(range(faces.shape[-1])))]), dim=1).detach()
                textures = torch.cat((textures, textures.permute((0, 1, 4, 3, 2, 5))), dim=1)
        else:
            return None

        # rasterization
        faces = nr.vertices_to_faces(vertices, faces)
        out = nr.rasterize_rgbad(
            faces, textures, self.image_size, self.anti_aliasing, self.near, self.far, self.rasterizer_eps,
            self.background_color)
        return out['rgb'], out['depth'], out['alpha']

    def project_points(self, vertices, K=None, R=None, t=None, dist_coeffs=None, orig_size=None):
        if self.camera_mode == 'look_at':
            vertices = nr.look_at(vertices, self.eye)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'look':
            vertices = nr.look(vertices, self.eye, self.camera_direction)
            # perspective transformation
            if self.perspective:
                vertices = nr.perspective(vertices, angle=self.viewing_angle)
        elif self.camera_mode == 'projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t
            if dist_coeffs is None:
                dist_coeffs = self.dist_coeffs
            if orig_size is None:
                orig_size = self.orig_size
            vertices = nr.projection(vertices, K, R, t, dist_coeffs, orig_size)
        elif self.camera_mode == 'weak_projection':
            if K is None:
                K = self.K
            if R is None:
                R = self.R
            if t is None:
                t = self.t

            vertices = weak_projection(vertices, K, R, t, orig_size)

            vertices[:, :, 0] = (orig_size / 2) * vertices[:, :, 0] + (orig_size / 2)
            vertices[:, :, 1] = orig_size - ((orig_size / 2) * vertices[:, :, 1] + (orig_size / 2))
        return vertices
