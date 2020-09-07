import VideoFace3D as vf3d
import cv2
import torch
import numpy as np
import os

segmenter = vf3d.FaceSegmentation()
land_2d = vf3d.FaceLandmarkDetector("2D")
land_3d = vf3d.FaceLandmarkDetector("3D")

comfortable_colors = vf3d.ComfortableColor()
colors = [
    comfortable_colors.sun_flower.to_bgr(),
    comfortable_colors.blue_jeans.to_bgr(),
    comfortable_colors.lavander.to_bgr(),
    comfortable_colors.bitter_sweet.to_bgr(),
    comfortable_colors.aqua.to_bgr()]


def get_video_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def do_results_tracking(video_path):
    global colors
    tracker = vf3d.FaceTracker(echo=True)
    print("start tracking ...")
    tracking_info = tracker.start_track(video_path)
    print("tracking done...")
    single_face_bbox = []
    all_frames = []
    for frame, people in tracking_info:
        single_face_bbox.append(people[0][0])
        all_frames.append(frame)

    single_face_bbox = np.array(single_face_bbox)  # [seq_len, 4]
    single_face_bbox = vf3d.video_temporal_smooth_constrains(single_face_bbox.T).T
    single_face_bbox = list(single_face_bbox.astype(np.int))

    all_frame_bboxs = []
    for i, bbox in enumerate(single_face_bbox):
        frame_bbox = vf3d.draw_bbox(all_frames[i], [bbox], colors=colors)
        all_frame_bboxs.append(frame_bbox)
        vf3d.progressbar(i + 1, len(single_face_bbox), prefix="plot tracking...")

    return all_frames, all_frame_bboxs, single_face_bbox


def do_results_landmark_detection(all_frames, bboxs, landmark_detector):
    # land_3d = vf3d.FaceLandmarkDetector("3D")
    cache_image_path = "./example_results/cache.png"

    ldmarks = []
    for i, frame in enumerate(all_frames):
        cv2.imwrite(cache_image_path, frame)
        lds = landmark_detector.detect_face_landmark(cache_image_path, [bboxs[i]])[0]
        ldmarks.append(lds)
        vf3d.progressbar(i + 1, len(all_frames), prefix="detect landmarks...")

    os.remove(cache_image_path)

    ldmarks = np.array(ldmarks)  # [seq, num_point, 2]
    seq, num_point, _ = ldmarks.shape
    ldmarks = ldmarks.reshape((seq, -1))
    ldmarks = vf3d.video_temporal_smooth_constrains(ldmarks.T).T
    ldmarks = ldmarks.reshape((seq, num_point, -1))

    all_frame_ldmarks = []
    for i in range(len(ldmarks)):
        frame_ldmarks = vf3d.draw_landmarks(all_frames[i], [ldmarks[i]], colors=colors)
        all_frame_ldmarks.append(frame_ldmarks)
        vf3d.progressbar(i + 1, len(all_frames), prefix="plot landmarks...")
    return all_frame_ldmarks, ldmarks


if __name__ == '__main__':
    video_path = r"E:\datasets\300VW_Dataset_2015_12_14\521\vid.avi"
    fps = get_video_fps(video_path)

    all_frames, all_frame_bboxs, all_bboxs = do_results_tracking(video_path)
    all_frame_ldmarks, all_ldmarks = do_results_landmark_detection(all_frames, all_bboxs, land_3d)
    vf3d.frames2video("./example_results/cache1.mp4", all_frame_ldmarks, fps=fps)
