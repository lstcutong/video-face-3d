import torch
import numpy as np
from scipy.io import loadmat, savemat
from PIL import Image
from MorphableModelFitting.utils.Global import *
import neural_renderer as nr
import math


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = -euler_angle[:, 0].reshape(-1, 1, 1)
    phi = -euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones(batch_size, 1, 1).to(euler_angle.device)
    zero = torch.zeros(batch_size, 1, 1).to(euler_angle.device)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def texture_from_point2faces(triangles, texutures):
    batch = len(texutures)
    tex = nr.vertices_to_faces(texutures, triangles)
    tex = torch.Tensor.mean(tex, dim=2)
    return tex.reshape((batch, tex.shape[1], 1, 1, 1, 3))


def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2 * npts, 8])

    A[0:2 * npts - 1:2, 0:3] = x.transpose()
    A[0:2 * npts - 1:2, 3] = 1

    A[1:2 * npts:2, 4:7] = x.transpose()
    A[1:2 * npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2 * npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
    t = np.stack([sTx, sTy], axis=0)

    return t, s


def alignment_and_crop(image_path, align_kp):
    Lm3D = loadmat(SIMILARITY_LM3D_ALL_MODEL_PATH)
    Lm3D = Lm3D['lm']

    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
                     Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    Lm2D = np.stack(
        [align_kp[lm_idx[0], :], np.mean(align_kp[lm_idx[[1, 2]], :], 0), np.mean(align_kp[lm_idx[[3, 4]], :], 0),
         align_kp[lm_idx[5], :], align_kp[lm_idx[6], :]], axis=0)
    Lm2D = Lm2D[[1, 2, 0, 3, 4], :]

    img = Image.open(image_path)
    w0, h0 = img.size

    Lm2D = np.stack([Lm2D[:, 0], h0 - 1 - Lm2D[:, 1]], axis=1)
    t, s = POS(Lm2D.transpose(), Lm3D.transpose())

    img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0 / 2, 0, 1, h0 / 2 - t[1]))
    w = (w0 / s * 102).astype(np.int32)
    h = (h0 / s * 102).astype(np.int32)
    img = img.resize((w, h), resample=Image.BILINEAR)
    # lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) / s * 102
    lm68 = np.stack([align_kp[:, 0] - t[0] + w0 / 2, align_kp[:, 1] + t[1] - h0 / 2], axis=1) / s * 102
    # crop the image to 224*224 from image center
    left = (w / 2 - 112).astype(np.int32)
    right = left + 224
    up = (h / 2 - 112).astype(np.int32)
    below = up + 224

    img = img.crop((left, up, right, below))
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.expand_dims(img, 0)
    lm68 = lm68 - np.reshape(np.array([(w / 2 - 112), (h / 2 - 112)]), [1, 2])
    return img, np.expand_dims(lm68, 0)


def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T
    x = x.T
    assert (x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert (n >= 4)

    # --- 1. normalization
    # 2d points
    mean = np.mean(x, 1)  # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x ** 2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3, 3), dtype=np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean * scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1)  # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3, :] - X
    average_norm = np.mean(np.sqrt(np.sum(X ** 2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4, 4), dtype=np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean * scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n * 2, 8), dtype=np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])

    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype=np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine


def rad2degree(angles):
    '''

    :param angles: [batch, 3]
    :return:
    '''
    return angles * 180 / np.pi


def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert (isRotationMatrix)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return x, y, z


def P2sRt(P):
    ''' decompositing camera matrix P
    Args:
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation.
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def compute_face_norm(vertices, triangles):
    pt1_index, pt2_index, pt3_index = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    pts1, pts2, pts3 = vertices[:, pt1_index, :], vertices[:, pt2_index, :], vertices[:, pt3_index, :]

    vec1 = pts1 - pts2
    vec2 = pts1 - pts3

    face_norm = torch.Tensor.cross(vec1, vec2)

    return face_norm


def compute_point_norm(vertices, triangles, point_buf):
    batch = len(vertices)
    face_norm = compute_face_norm(vertices, triangles)

    face_norm = torch.cat([face_norm, torch.zeros((batch, 1, 3)).to(vertices.device)], dim=1)

    v_norm = torch.sum(face_norm[:, point_buf, :], dim=2)
    v_norm = v_norm / (torch.norm(v_norm, dim=2).unsqueeze(2))
    return v_norm


def texture_mapping(image, geo, s, R, t):
    pass


from matplotlib.path import Path


def inpolygon(xq, yq, xv, yv):
    """
    reimplement inpolygon in matlab
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # 合并xv和yv为顶点数组
    vertices = np.vstack((xv, yv)).T
    # 定义Path对象
    path = Path(vertices)
    # 把xq和yq合并为test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # 得到一个test_points是否严格在path内的mask，是bool值数组
    _in = path.contains_points(test_points)
    # 得到一个test_points是否在path内部或者在路径上的mask
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # 得到一个test_points是否在path路径上的mask
    _on = _in ^ _in_on

    return _in_on, _on


def create_mask_fiducial(fiducials, image):
    # fiducials: 2x68
    border_fid = fiducials[:, 0:17]
    face_fid = fiducials[:, 18:]

    c1 = np.array([[border_fid[0, 0]], [face_fid[1, 2]]])
    c2 = np.array([[border_fid[0, 16]], [face_fid[1, 7]]])

    eye = np.linalg.norm(face_fid[:, 22] - face_fid[:, 25])
    c3, c4 = face_fid[:, 2], face_fid[:, 7]
    c3[1] = c3[1] - 0.3 * eye
    c4[1] = c4[1] - 0.3 * eye

    border = np.column_stack([c1, border_fid, c2, c4, c3])

    h, w = image.shape[0:2]

    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    _in, _on = inpolygon(X.reshape(-1), Y.reshape(-1), border[0, :], border[1, :])

    mask = np.round(np.reshape(_in + _on, (h, w)))
    return (mask * 255).astype(np.uint8)


def save_obj(v,c,f,save_path):
    folder = os.path.split(save_path)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(save_path, 'w') as file:
        for i in range(len(c)):
            file.write('v %f %f %f %f %f %f\n' % (v[i, 0], v[i, 1], v[i, 2], c[i, 0], c[i, 1], c[i, 2]))

        file.write('\n')

        for i in range(len(f)):
            file.write('f %d %d %d\n' % (f[i, 0], f[i, 1], f[i, 2]))
    file.close()