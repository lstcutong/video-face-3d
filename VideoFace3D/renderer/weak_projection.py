from __future__ import division

import torch


def weak_projection(vertices, K, R, t, orig_size):
    '''
    Calculate weak_projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 scale matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters

    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]

    vertices = torch.stack([x, y, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1,2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)
    return vertices
