#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import torch
import pickle
import scipy.io as sio


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))


def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()


def _load_gpu(fp):
    return torch.from_numpy(_load(fp)).cuda()


def load_bfm(model_path):
    suffix = _get_suffix(model_path)
    if suffix == 'mat':
        C = sio.loadmat(model_path)
        model = C['model_refine']
        model = model[0, 0]

        model_new = {}
        w_shp = model['w'].astype(np.float32)
        model_new['w_shp_sim'] = w_shp[:, :40]
        w_exp = model['w_exp'].astype(np.float32)
        model_new['w_exp_sim'] = w_exp[:, :10]

        u_shp = model['mu_shape']
        u_exp = model['mu_exp']
        u = (u_shp + u_exp).astype(np.float32)
        model_new['mu'] = u
        model_new['tri'] = model['tri'].astype(np.int32) - 1

        # flatten it, pay attention to index value
        keypoints = model['keypoints'].astype(np.int32) - 1
        keypoints = np.concatenate((3 * keypoints, 3 * keypoints + 1, 3 * keypoints + 2), axis=0)

        model_new['keypoints'] = keypoints.T.flatten()

        #
        w = np.concatenate((w_shp, w_exp), axis=1)
        w_base = w[keypoints]
        w_norm = np.linalg.norm(w, axis=0)
        w_base_norm = np.linalg.norm(w_base, axis=0)

        dim = w_shp.shape[0] // 3
        u_base = u[keypoints].reshape(-1, 1)
        w_shp_base = w_shp[keypoints]
        w_exp_base = w_exp[keypoints]

        model_new['w_norm'] = w_norm
        model_new['w_base_norm'] = w_base_norm
        model_new['dim'] = dim
        model_new['u_base'] = u_base
        model_new['w_shp_base'] = w_shp_base
        model_new['w_exp_base'] = w_exp_base

        _dump(model_path.replace('.mat', '.pkl'), model_new)
        return model_new
    else:
        return _load(model_path)


def load_obj(obj_file):
    sents = open(obj_file).read().split("\n")
    vertices, color, tri = [], [], []
    for s in sents:
        p = s.split(" ")
        if p[0] == 'v':
            vertices.append([float(p[1]), float(p[2]), float(p[3])])
            if len(p) == 7:
                color.append([int(p[4]), int(p[5]), int(p[6])])
        if p[0] == 'c':
            color.append([int(p[4]), int(p[5]), int(p[6])])
        if p[0] == 'f':
            tri.append([int(p[1]), int(p[2]), int(p[3])])

    return np.array(vertices), np.array(color)[:, ::-1], np.array(tri) - 1


def write_obj_with_colors(obj_name, vertices, triangles, colors):
    triangles = triangles.copy()  # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[0]):
            s = 'v {:.4f} {:.4f} {:.4f} {} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2], colors[i, 2],
                                                           colors[i, 1], colors[i, 0])
            f.write(s)

        # write f: ver ind/ uv ind
        for i in range(triangles.shape[0]):
            s = 'f {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2])
            f.write(s)


_load_cpu = _load
_numpy_to_tensor = lambda x: torch.from_numpy(x)
_tensor_to_numpy = lambda x: x.cpu()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))
_cuda_to_tensor = lambda x: x.cpu()
_cuda_to_numpy = lambda x: x.cpu().numpy()
