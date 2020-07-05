# coding=utf8
from __future__ import absolute_import, division, print_function
import dlib
import cv2
import numpy as np
import os
import sys
from matplotlib.path import Path


class MaskGenerator:
    def __init__(self, landmarks_path):
        """
        :param landmarks_path: the path of pretrained key points weight,
        it could be download from:
        http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        """
        if not os.path.exists(landmarks_path):
            raise RuntimeError('face landmark file is not exist. please download if from: \n'
                               'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 '
                               'and uncompress it.')
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(landmarks_path)

    def bounding_boxes(self, image):
        # convert to gray image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # get rect contains face
        face_rects = self._detector(gray_image, 0)
        return face_rects

    def align(self, image, size=(240, 240), scale=1.8, warp=True, crop=True, resize=True,
              crop_function_version=0, align_multi=False, draw_landmarks=False):
        """
        warp and crop image
        https://blog.csdn.net/qq_39438636/article/details/79304130

        :param image: a BGR format face image
        :type image: np.ndarray
        :param size: target size
        :param scale:
        :param warp: warp or not
        :param crop: crop or not
        :param resize: resize od not
        :param crop_function_version: crop function version
        :param align_multi: whther to detect multi face
        :param draw_landmarks: whether draw face landmarks
        :return: mask, image and whether successfully crop image
        """
        # check option
        if crop_function_version == 1 and align_multi:
            raise RuntimeError("When align_multi is true, crop_function_version must be 0")
        # if image is too big, resize to a smaller image
        if np.min(image.shape[0:2]) > 1000:
            ratio = 1000 / np.min(image.shape[0:2])
            image = cv2.resize(image, dsize=(0, 0), fx=ratio, fy=ratio)
        # make border for image
        border = int(np.min(image.shape[0:2]) * 0.3)
        image = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_CONSTANT)
        # backup image
        original_image = image.copy()
        # get rectangles which contains face
        face_rects = self.bounding_boxes(image)
        results = []
        if len(face_rects) > 0:
            for i in range(len(face_rects)):
                # get 68 landmarks of face
                landmarks = np.array([[p.x, p.y] for p in self._predictor(original_image, face_rects[i]).parts()])
                # draw landmarks
                if draw_landmarks:
                    landmark_image = self.draw_landmarks(original_image, landmarks)
                    # remove border
                    _row, _col, _ = landmark_image.shape
                    landmark_image = landmark_image[border:_row-border, border:_col-border, :]
                else:
                    landmark_image = None
                # create mask using landmarks
                mask = create_mask_by_landmarks(landmarks.T, original_image)
                if warp:
                    image, mask, r_mat = self._warp(original_image, mask, landmarks)
                    landmarks = self._get_rotated_points(landmarks, r_mat)
                if crop:
                    if crop_function_version == 0:
                        image = self._crop_v0(image, landmarks, scale)
                        mask = self._crop_v0(mask, landmarks, scale)
                    elif crop_function_version == 1:
                        image, mask, suc_ = self._crop_v1(image, mask, scale)
                        if not suc_:
                            sys.stderr.write('%s: Failed to crop image and mask\n' % __file__)
                    else:
                        raise RuntimeError("crop_function_version must be 0 or 1")

                if resize:
                    results.append((True, cv2.resize(mask, size), cv2.resize(image, size), landmark_image))
                else:
                    results.append((True, mask, image, landmark_image))

                if not align_multi:
                    return results
            return results
        else:
            sys.stderr.write("%s: Can't detect face in image\n" % __file__)
            image = cv2.resize(image, size)
            return [(False, np.ones(image.shape, dtype=image.dtype) * 255, image, None)]

    @staticmethod
    def _get_rotated_points(points, rotate_mat):
        # Blog； https://www.cnblogs.com/zhoug2020/p/7842808.html
        # add 1 to every point
        __padding = np.ones((points.shape[0], 1), dtype=points.dtype)
        points = np.concatenate([points, __padding], axis=1)
        # add [0, 0, 1] to rotate matrix
        __padding = np.array([0, 0, 1], dtype=points.dtype).reshape(1, 3)
        rotate_mat = np.concatenate([rotate_mat, __padding], axis=0)
        # compute rotated landmarks
        rotate_landmarks = np.matmul(rotate_mat, points.T)
        # remove the padding and transpose landmarks
        rotate_landmarks = rotate_landmarks[0:2, :].T
        # return landmark as integer numpy array
        return rotate_landmarks.astype(points.dtype)

    @staticmethod
    def _warp(image, mask, landmarks):
        """
        warp image and mask by landmarks
        :param image:
        :type image np.ndarray
        :param landmarks:
        :type landmarks np.ndarray
        :return: warped face and mask
        """
        # landmarks.shape = (68, 2)
        landmarks = np.array(landmarks)
        # compute rotate angle, r_angle=arctan((y1-y2)/(x1-x2))
        # landmarks[36]: corner of left eye
        # landmarks[42]: corner of right eye
        r_angle = np.arctan((landmarks[36][1] - landmarks[42][1]) /
                            (landmarks[36][0] - landmarks[42][0]))
        r_angle = 180 * r_angle / np.pi
        # get rotation matrix
        rot_mat = cv2.getRotationMatrix2D(tuple(landmarks[2]), r_angle, scale=1)

        # rotate image and mask
        rotated_image = cv2.warpAffine(image, rot_mat, dsize=image.shape[0:2])
        rotated_mask = cv2.warpAffine(mask, rot_mat, dsize=image.shape[0:2])

        return rotated_image, rotated_mask, rot_mat

    def _crop_v0(self, image, landmarks, scale):
        """
        crop image by face landmarks
        :param image:
        :param landmarks:
        :param scale:
        :return:
        """
        # left eye: landmarks[36]
        # left mouth: landmarks[48]
        # nose: landmarks[29]
        # find the most left point and most right point
        landmarks_x = landmarks[:, 0]
        most_left_x = np.min(landmarks_x)
        most_right_x = np.max(landmarks_x)
        mid_x = (most_left_x + most_right_x) // 2
        # print(most_left_x, most_right_x, mid_x)
        # define new center point use mid_x and y from nose point
        center_point = [mid_x, landmarks[29][1]]
        # compute the distance between left eye(landmarks[36])
        distance = most_right_x - mid_x
        size = distance * scale
        # print(center_point)
        # compute row_start, row_end, col_start, col_end
        row_start = int(center_point[1] - size)
        row_end = int(center_point[1] + size)
        col_start = int(center_point[0] - size)
        col_end = int(center_point[0] + size)
        # print('*' * 10)
        # print(row_start, row_end, col_start, col_end)
        # make range valid and compute padding
        if row_start < 0:
            padding_up = abs(row_start)
            row_start = 0
        else:
            padding_up = 0
        if col_start < 0:
            padding_left = abs(col_start)
            col_start = 0
        else:
            padding_left = 0
        if row_end > (image.shape[0] - 1):
            padding_down = row_end - (image.shape[0] - 1)
            row_end = image.shape[0] - 1
        else:
            padding_down = 0
        if col_end > (image.shape[1] - 1):
            padding_right = col_end - (image.shape[1] - 1)
            col_end = image.shape[1] - 1
        else:
            padding_right = 0
        # print(row_start, row_end, col_start, col_end)
        # print('*' * 10)
        # crop image
        cropped_image = self._crop_helper(image, row_start, row_end, col_start, col_end,
                                          padding_up, padding_down, padding_left, padding_right)
        return cropped_image

    def _crop_v1(self, image, mask, scale):
        face_rects = self.bounding_boxes(image)
        if len(face_rects) == 0:
            return image, mask, False
        # define crop size
        size = (face_rects[0].right() - face_rects[0].left()) / 2
        size *= scale
        # define new center point use mid_x and y from nose point
        _x = (face_rects[0].left() + face_rects[0].right()) // 2
        _y = (face_rects[0].top() + face_rects[0].bottom()) // 2
        center_point = [_x, _y]
        # compute the distance between left eye(landmarks[36])
        # print(center_point)
        # compute row_start, row_end, col_start, col_end
        row_start = int(center_point[1] - size)
        row_end = int(center_point[1] + size)
        col_start = int(center_point[0] - size)
        col_end = int(center_point[0] + size)
        # print('*' * 10)
        # print(row_start, row_end, col_start, col_end)
        # make range valid and compute padding
        if row_start < 0:
            padding_up = abs(row_start)
            row_start = 0
        else:
            padding_up = 0
        if col_start < 0:
            padding_left = abs(col_start)
            col_start = 0
        else:
            padding_left = 0
        if row_end > (image.shape[0] - 1):
            padding_down = row_end - (image.shape[0] - 1)
            row_end = image.shape[0] - 1
        else:
            padding_down = 0
        if col_end > (image.shape[1] - 1):
            padding_right = col_end - (image.shape[1] - 1)
            col_end = image.shape[1] - 1
        else:
            padding_right = 0
        # print(row_start, row_end, col_start, col_end)
        # print('*' * 10)
        # crop image
        image = self._crop_helper(image, row_start, row_end, col_start, col_end,
                                  padding_up, padding_down, padding_left, padding_right)
        mask = self._crop_helper(mask, row_start, row_end, col_start, col_end,
                                 padding_up, padding_down, padding_left, padding_right)
        return image, mask, True

    @staticmethod
    def _crop_helper(image, row_start, row_end, col_start, col_end,
                     padding_up, padding_down, padding_left, padding_right):
        cropped_image = image[row_start:row_end, col_start:col_end]

        # add padding to image
        rows, cols, _ = cropped_image.shape
        if padding_up > 0:
            padding = np.zeros(shape=(padding_up, cols, 3), dtype=cropped_image.dtype)
            cropped_image = np.vstack((padding, cropped_image))
        if padding_down > 0:
            padding = np.zeros(shape=(padding_down, cols, 3), dtype=cropped_image.dtype)
            cropped_image = np.vstack((cropped_image, padding))
        rows, cols, _ = cropped_image.shape
        if padding_left > 0:
            padding = np.zeros(shape=(rows, padding_left, 3), dtype=cropped_image.dtype)
            cropped_image = np.hstack((padding, cropped_image))
        if padding_right > 0:
            padding = np.zeros(shape=(rows, padding_right, 3), dtype=cropped_image.dtype)
            cropped_image = np.hstack((cropped_image, padding))
        return cropped_image

    @staticmethod
    def draw_landmarks(image, landmarks):
        landmark_im = image.copy()
        for i, landmark in enumerate(landmarks):
            cv2.circle(landmark_im, tuple(landmark), 3, (0, 0, 255))
            cv2.putText(landmark_im, str(i), tuple(landmark), cv2.FONT_HERSHEY_SIMPLEX,
                        0.3, (0, 255, 0))
        return landmark_im


def create_mask_by_landmarks(landmarks, Image):
    """
    create mask use fiducials of Image
    :param landmarks: the 68 landmarks detected using dlib
    :type landmarks np.ndarray
    :param Image: a 3-channel image
    :type Image np.ndarray
    :return:
    """
    # fiducals is 2x68
    landmarks = np.float32(landmarks)
    border_fid = landmarks[:, 0:17]
    face_fid = landmarks[:, 17:]

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
