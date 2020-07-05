# coding=utf8
import numpy as np
import sys
from matplotlib.path import Path


def create_shading_recon(n_out2, al_out2, light_out):
    """
    :type n_out2: np.ndarray
    :type al_out2: np.ndarray
    :type light_out: np.ndarray
    :return:
    """
    M = n_out2.shape[0]
    No1 = np.reshape(n_out2, (M * M, 3))
    tex1 = np.reshape(al_out2, (M * M, 3))

    la = lambertian_attenuation(3)
    HN1 = normal_harmonics(No1.T, la)

    HS1r = np.matmul(HN1, light_out[0:9])
    HS1g = np.matmul(HN1, light_out[9:18])
    HS1b = np.matmul(HN1, light_out[18:27])

    HS1 = np.zeros(shape=(M, M, 3), dtype=np.float32)
    HS1[:, :, 0] = np.reshape(HS1r, (M, M))
    HS1[:, :, 1] = np.reshape(HS1g, (M, M))
    HS1[:, :, 2] = np.reshape(HS1b, (M, M))
    Tex1 = np.reshape(tex1, (M, M, 3)) * HS1

    IRen0 = Tex1
    Shd = (200 / 255.0) * HS1  # 200 is added instead of 255 so that not to scale the shading to all white
    Ishd0 = Shd
    return [IRen0, Ishd0]


def lambertian_attenuation(n):
    # a = [.8862; 1.0233; .4954];
    a = [np.pi * i for i in [1.0, 2 / 3.0, .25]]
    if n > 3:
        sys.stderr.write('don\'t record more than 3 attenuation')
        exit(-1)
    o = a[0:n]
    return o


def normal_harmonics(N, att):
    """
    Return the harmonics evaluated at surface normals N, attenuated by att.
    :param N:
    :param att:
    :return:

    Normals can be scaled surface normals, in which case value of each
    harmonic at each point is scaled by albedo.
    Harmonics written as polynomials
    0,0    1/sqrt(4*pi)
    1,0    z*sqrt(3/(4*pi))
    1,1e    x*sqrt(3/(4*pi))
    1,1o    y*sqrt(3/(4*pi))
    2,0   (2*z.^2 - x.^2 - y.^2)/2 * sqrt(5/(4*pi))
    2,1e  x*z * 3*sqrt(5/(12*pi))
    2,1o  y*z * 3*sqrt(5/(12*pi))
    2,2e  (x.^2-y.^2) * 3*sqrt(5/(48*pi))
    2,2o  x*y * 3*sqrt(5/(12*pi))
    """
    xs = N[0, :].T
    ys = N[1, :].T
    zs = N[2, :].T
    a = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    denom = (a == 0) + a
    # %x = xs./a; y = ys./a; z = zs./a;
    x = xs / denom
    y = ys / denom
    z = zs / denom

    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z

    H1 = att[0] * (1 / np.sqrt(4 * np.pi)) * a
    H2 = att[1] * (np.sqrt(3 / (4 * np.pi))) * zs
    H3 = att[1] * (np.sqrt(3 / (4 * np.pi))) * xs
    H4 = att[1] * (np.sqrt(3 / (4 * np.pi))) * ys
    H5 = att[2] * (1 / 2.0) * (np.sqrt(5 / (4 * np.pi))) * ((2 * z2 - x2 - y2) * a)
    H6 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xz * a)
    H7 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (yz * a)
    H8 = att[2] * (3 * np.sqrt(5 / (48 * np.pi))) * ((x2 - y2) * a)
    H9 = att[2] * (3 * np.sqrt(5 / (12 * np.pi))) * (xy * a)
    H = [H1, H2, H3, H4, H5, H6, H7, H8, H9]

    # --------add by wang -----------
    H = [np.expand_dims(h, axis=1) for h in H]
    H = np.concatenate(H, -1)
    # -------------end---------------
    return H


def create_mask_fiducial(fiducials, Image):
    """
    create mask use fiducials of Image
    :param fiducials: the 68 landmarks detected using dlib
    :type fiducials np.ndarray
    :param Image: a 3-channel image
    :type Image np.ndarray
    :return:
    """
    # fiducals is 2x68
    fiducials = np.float32(fiducials)
    border_fid = fiducials[:, 0:17]
    face_fid = fiducials[:, 17:]

    c1 = np.array([border_fid[0, 0], face_fid[1, 2]])  # left
    c2 = np.array([border_fid[0, 16], face_fid[1, 7]])  # right
    eye = np.linalg.norm(face_fid[:, 22] - face_fid[:, 25])
    c3 = face_fid[:, 2]
    c3[1] = c3[1] - 0.3 * eye
    c4 = face_fid[:, 7]
    c4[1] = c4[1] - 0.3 * eye

    border = [c1, border_fid, c2, c4, c3]
    border = [item.reshape(2, -1) for item in border]
    border = np.hstack(border)

    M = Image.shape[0]  # row -> y
    N = Image.shape[1]  # col -> x

    y = np.arange(0, M, step=1, dtype=np.float32)
    x = np.arange(0, N, step=1, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    _in, _on = inpolygon(X, Y, border[0, :].T, border[1, :].T)

    mask = np.round(np.reshape(_in | _on, [M, N]))
    mask = 255 * np.uint8(mask)
    mask = np.repeat(np.expand_dims(mask, -1), 3, axis=-1)
    return mask


def inpolygon(xq, yq, xv, yv):
    """
    reimplement inpolygon in matlab
    :type xq: np.ndarray
    :type yq: np.ndarray
    :type xv: np.ndarray
    :type yv: np.ndarray
    """
    # http://blog.sina.com.cn/s/blog_70012f010102xnel.html
    # merge xy and yv into vertices
    vertices = np.vstack((xv, yv)).T
    # define a Path object
    path = Path(vertices)
    # merge X and Y into test_points
    test_points = np.hstack([xq.reshape(xq.size, -1), yq.reshape(yq.size, -1)])
    # get mask of test_points in path
    _in = path.contains_points(test_points)
    # get mask of test_points in path(include the points on path)
    _in_on = path.contains_points(test_points, radius=-1e-10)
    # get the points on path
    _on = _in ^ _in_on
    return _in_on, _on

