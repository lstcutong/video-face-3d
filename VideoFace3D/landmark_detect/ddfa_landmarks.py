import torch
import torchvision.transforms as transforms

import numpy as np
import cv2
import dlib
from VideoFace3D.landmark_detect.ddfa_ddfa import ToTensorGjz, NormalizeGjz, str2bool

from VideoFace3D.landmark_detect.ddfa_inference import get_suffix, parse_roi_box_from_landmark, crop_img, \
    predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from VideoFace3D.landmark_detect.ddfa_estimate_pose import parse_pose

STD_SIZE = 120


def detect_landmark_ddfa_3D(image_path, model, face_regressor, device, bbox_init="one", rects=None):
    model.eval()

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    img_ori = cv2.imread(image_path)
    dlib_landmarks = True if rects is None else False
    if rects is None:
        face_detector = dlib.get_frontal_face_detector()
        gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
        rects = face_detector(gray, 1)

    if len(rects) == 0:
        return []
    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    vertices_lst = []  # store multiple face vertices
    ind = 0
    suffix = get_suffix(image_path)
    for rect in rects:


        if dlib_landmarks:
            pts = face_regressor(img_ori, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            roi_box = parse_roi_box_from_landmark(pts)
        else:
            roi_box = rect
        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        # two-step for more accurate bbox to crop face
        if bbox_init == 'two':
            roi_box = parse_roi_box_from_landmark(pts68)
            img_step2 = crop_img(img_ori, roi_box)
            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0).to(device)
            with torch.no_grad():
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68.transpose(1, 0)[:, 0:2])
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

    vertices = predict_dense(param, roi_box)
    # colors = get_colors(img_ori, vertices)
    return pts_res


'''
def detect_landmark_ddfa_3D(image_path, model, rects, face_regressor, device, bbox_init="one"):
    model.eval()

    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    img_ori = cv2.imread(image_path)
    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)

    if len(rects) == 0:
        return []
    pts_res = []
    Ps = []  # Camera matrix collection
    poses = []  # pose collection, [todo: validate it]
    vertices_lst = []  # store multiple face vertices
    ind = 0
    suffix = get_suffix(image_path)
    for rect in rects:

        pts = face_regressor(img_ori, rect).parts()
        pts = np.array([[pt.x, pt.y] for pt in pts]).T
        roi_box = parse_roi_box_from_landmark(pts)

        img = crop_img(img_ori, roi_box)

        # forward: one step
        img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
        input = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            param = model(input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
        # 68 pts
        pts68 = predict_68pts(param, roi_box)

        # two-step for more accurate bbox to crop face
        if bbox_init == 'two':
            roi_box = parse_roi_box_from_landmark(pts68)
            img_step2 = crop_img(img_ori, roi_box)
            img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
            input = transform(img_step2).unsqueeze(0).to(device)
            with torch.no_grad():
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            pts68 = predict_68pts(param, roi_box)

        pts_res.append(pts68.transpose(1, 0)[:, 0:2])
        P, pose = parse_pose(param)
        Ps.append(P)
        poses.append(pose)

    vertices = predict_dense(param, roi_box)
    # colors = get_colors(img_ori, vertices)
    return pts_res
'''