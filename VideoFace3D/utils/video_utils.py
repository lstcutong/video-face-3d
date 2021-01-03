import cv2
import os
import sys
from VideoFace3D.utils.Global import *
from VideoFace3D.utils.temporal_smooth import *

def id_generator(number, id_len=4):
    number = str(number)
    assert len(number) < id_len

    return "0" * (id_len - len(number)) + number

def progressbar(current, total, num=40, prefix=""):
    sys.stdout.write("\r{} {}/{} |{}{}| {:.2f}%".format(prefix, current, total,
                                                        "*" * int(num * current / total),
                                                        " " * (num - int(num * current / total)),
                                                        100 * current / total))
    sys.stdout.flush()
    if current == total:
        print("")

def str2seconds(time):
    try:

        h, m, s = time.split(":")[0], time.split(":")[1], time.split(":")[2]
        h, m, s = int(h), int(m), int(s)
        assert h >= 0
        assert 0 <= m < 60
        assert 0 <= s < 60
        seconds = h * 3600 + m * 60 + s
        return int(seconds)
    except:
        assert False, "wrong time format"
        #sys.exit(0)

def extract_frame_from_video(video_path, save_path=None, ret_frame=True, time_start="default", time_end="default"):
    start_frame, end_frame = 0, 0
    
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(5)
    frame_nums = cap.get(7)
    total_seconds = int(frame_nums / fps)

    if time_start == "default":
        start_frame = 0
    else:
        start_frame = int(frame_nums * (str2seconds(time_start) / total_seconds))
    if time_end == "default":
        end_frame = frame_nums
    else:
        tmp = int(frame_nums * (str2seconds(time_end) / total_seconds))
        if tmp > frame_nums:
            end_frame = frame_nums
        else:
            end_frame = tmp

    assert start_frame <= end_frame

    iters = int(end_frame - start_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_frames = []
    for count in range(iters):
        ret, frame = cap.read()
        if frame is None:
            break
        if ret_frame:
            all_frames.append(frame)
        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, "{}.png".format(id_generator(count, 7))), frame)
            progressbar(count+1, iters, prefix="extract")
    if ret_frame:
        return all_frames
    else:
        return None

def frames2video(save_path, frames, fps=24):
    base_folder = os.path.split(save_path)[0]
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    H, W = frames[0].shape[0:2]
    img_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    num = 0
    for frame in frames:
        video_writer.write(frame)
        num += 1
        progressbar(num, len(frames), prefix="write video")

    video_writer.release()


def video_temporal_smooth_constrains(ref_param, method=SMOOTH_METHODS_GAUSSIAN):
    if method == SMOOTH_METHODS_DCNN:
        return smooth_DCNN(ref_param)

    if method == SMOOTH_METHODS_GAUSSIAN:
        return smooth_gaussian_filter(ref_param)

    if method == SMOOTH_METHODS_MEAN:
        return smooth_mean_filter(ref_param)

    if method == SMOOTH_METHODS_MEDUIM:
        return smooth_medium_filter(ref_param)

    if method == SMOOTH_METHODS_OPTIMIZE:
        return smooth_optimize(ref_param)
