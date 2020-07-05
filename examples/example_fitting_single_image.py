import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import VideoFace3D as vf3d

def render_single_batch_image(annot, image_size):
    device = torch.device("cuda")

    facemodel = vf3d.FaceModelBFM()

    idi, ex, tex, r, t, s, gamma = annot["id"], annot["exp"], annot["tex"], annot["r"], annot["t"], annot["s"], annot[
        "gamma"]

    s = torch.from_numpy(s).unsqueeze(2).to(device).float()
    t = torch.from_numpy(t).unsqueeze(1).to(device).float()
    r = torch.from_numpy(r).to(device).float()
    gamma = torch.from_numpy(gamma).to(device).float()

    batch = len(s)
    R = vf3d.euler2rot(r)
    K = torch.Tensor.repeat(torch.eye(3).unsqueeze(0), (batch, 1, 1)).to(device) * s

    shape = facemodel.shape_formation(torch.from_numpy(idi).to(device), torch.from_numpy(ex).to(device))
    texture = facemodel.texture_formation(torch.from_numpy(tex).to(device))

    triangles = torch.Tensor.repeat((torch.from_numpy(facemodel.tri) - 1).long().unsqueeze(0), (batch, 1, 1)).to(device)


    renderer = vf3d.Renderer(image_size=image_size, K=K, R=R, t=t, near=0.1, far=10000, light_mode="SH", SH_Coeff=gamma)
    rgb, depth, silh = renderer(shape, triangles, texture)
    rgb = rgb.detach().cpu().numpy().transpose((0, 2, 3, 1)) * 255
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb[:, :, :, ::-1]


def render_and_save_result(image, annot, save_path):
    render_im = render_single_batch_image(annot, 224)
    all_im = []
    for i in range(len(image)):
        im = image[i]
        re_im = render_im[i]
        all_im.append(np.column_stack([im, re_im]))

    all_im = np.row_stack(all_im)
    plt.clf()
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(all_im[:, :, ::-1])
    #plt.show()
    plt.savefig(save_path)


def ModelFitting():
    image_path = [
        "./example_pics/1.png",
        "./example_pics/2.png"
    ]

    #accurate_fitting = FaceFittingPipline(image_path, "accurate", show_mid=False)
    fast_fitting = vf3d.FaceFittingPipline("fast", show_mid=False, checking=False)
    landmark_detect = vf3d.FaceLandmarkDetector("3D")
    #t0 = time.time()
    #result_accu = accurate_fitting.start_fiiting()
    #t1 = time.time()

    t2 = time.time()
    result_fast = fast_fitting.start_fiiting(image_path,landmark_detect)
    t3 = time.time()
    for i in range(len(result_fast)):
        render_and_save_result(result_fast[i][0], result_fast[i][2], "./example_results/fitting_fast_{}.png".format(i))
    #for i in range(len(result_accu)):
    #    render_and_save_result(result_accu[i][0],result_accu[i][2],"./accu_{}.png".format(i))
    #print("accurate fitting using time:{}".format(t1-t0))
    print("fast fitting using time:{}".format(t3 - t2))

if __name__ == '__main__':
    ModelFitting()
