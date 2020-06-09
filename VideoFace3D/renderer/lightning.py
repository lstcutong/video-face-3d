import torch
import math


def Illumination_SH(face_texture, norm, gamma):
    '''

    :param face_texture: [batch, face_num, 3]
    :param norm: [batch, face_num, 3]
    :param gamma: [batch, 27]
    :return:
    '''
    pi = 3.1415926
    num_vertex = face_texture.shape[1]
    batch = len(face_texture)

    init_lit = torch.Tensor([0.8, 0, 0, 0, 0, 0, 0, 0, 0]).to(gamma.device)
    gamma = torch.reshape(gamma, [-1, 3, 9])
    gamma = gamma + torch.reshape(init_lit, [1, 1, 9])

    # parameter of 9 SH function
    a0 = torch.Tensor([pi]).to(gamma.device)
    a1 = torch.Tensor([2 * pi / math.sqrt(3.0)]).to(gamma.device)
    a2 = torch.Tensor([2 * pi / math.sqrt(8.0)]).to(gamma.device)
    c0 = torch.Tensor([1 / math.sqrt(4 * pi)]).to(gamma.device)
    c1 = torch.Tensor([math.sqrt(3.0) / math.sqrt(4 * pi)]).to(gamma.device)
    c2 = torch.Tensor([3 * math.sqrt(5.0) / math.sqrt(12 * pi)]).to(gamma.device)

    Y0 = torch.Tensor.repeat(torch.reshape(a0 * c0, [1, 1, 1]), [batch, num_vertex, 1])
    Y1 = torch.reshape(-a1 * c1 * norm[:, :, 1], [batch, num_vertex, 1])
    Y2 = torch.reshape(a1 * c1 * norm[:, :, 2], [batch, num_vertex, 1])
    Y3 = torch.reshape(-a1 * c1 * norm[:, :, 0], [batch, num_vertex, 1])
    Y4 = torch.reshape(a2 * c2 * norm[:, :, 0] * norm[:, :, 1], [batch, num_vertex, 1])
    Y5 = torch.reshape(-a2 * c2 * norm[:, :, 1] * norm[:, :, 2], [batch, num_vertex, 1])
    Y6 = torch.reshape(a2 * c2 * 0.5 / math.sqrt(3.0) * (3 * norm[:, :, 2] ** 2 - 1), [batch, num_vertex, 1])
    Y7 = torch.reshape(-a2 * c2 * norm[:, :, 0] * norm[:, :, 2], [batch, num_vertex, 1])
    Y8 = torch.reshape(a2 * c2 * 0.5 * (norm[:, :, 0] ** 2 - norm[:, :, 1] ** 2), [batch, num_vertex, 1])

    Y = torch.cat([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], dim=2)

    # Y shape:[batch,N,9].

    # [batch,N,9] * [batch,9,1] = [batch,N]
    lit_r = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 0, :], 2)), 2)
    lit_g = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 1, :], 2)), 2)
    lit_b = torch.squeeze(torch.matmul(Y, torch.unsqueeze(gamma[:, 2, :], 2)), 2)

    # shape:[batch,N,3]

    face_color_r = (lit_r * face_texture[:, :, 0]).unsqueeze(2)
    face_color_g = (lit_g * face_texture[:, :, 1]).unsqueeze(2)
    face_color_b = (lit_b * face_texture[:, :, 2]).unsqueeze(2)

    face_color = torch.cat([face_color_r, face_color_g, face_color_b], dim=2)
    lighting = torch.cat([lit_r.unsqueeze(2), lit_g.unsqueeze(2), lit_b.unsqueeze(2)], dim=2) * 128
    return face_color, lighting


import torch
import torch.nn.functional as F
import numpy as np

def lighting_phong(faces, textures, intensity_ambient=0.5, intensity_directional=0.5,
             color_ambient=(1, 1, 1), color_directional=(1, 1, 1), direction=(0, 1, 0)):

    bs, nf = faces.shape[:2]
    device = faces.device

    # arguments
    # make sure to convert all inputs to float tensors
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = torch.tensor(color_ambient, dtype=torch.float32, device=device)
    elif isinstance(color_ambient, np.ndarray):
        color_ambient = torch.from_numpy(color_ambient).float().to(device)
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = torch.tensor(color_directional, dtype=torch.float32, device=device)
    elif isinstance(color_directional, np.ndarray):
        color_directional = torch.from_numpy(color_directional).float().to(device)
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
    elif isinstance(direction, np.ndarray):
        direction = torch.from_numpy(direction).float().to(device)
    if color_ambient.ndimension() == 1:
        color_ambient = color_ambient[None, :]
    if color_directional.ndimension() == 1:
        color_directional = color_directional[None, :]
    if direction.ndimension() == 1:
        direction = direction[None, :]

    # create light
    light = torch.zeros(bs, nf, 3, dtype=torch.float32).to(device)

    # ambient light
    if intensity_ambient != 0:
        light += intensity_ambient * color_ambient[:, None, :]

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        # pytorch normalize divides by max(norm, eps) instead of (norm+eps) in chainer
        normals = F.normalize(torch.cross(v10, v12), eps=1e-5)
        normals = normals.reshape((bs, nf, 3))

        if direction.ndimension() == 2:
            direction = direction[:, None, :]
        cos = F.relu(torch.sum(normals * direction, dim=2))
        # may have to verify that the next line is correct
        light += intensity_directional * (color_directional[:, None, :] * cos[:, :, None])

    # apply
    light = light[:,:,None, None, None, :]
    textures *= light
    return textures
