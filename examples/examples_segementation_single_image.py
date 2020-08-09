import VideoFace3D as vf3d
import cv2
import shutil
import os
import numpy as np

def example_segmentation():
    image_path = ["./example_pics/1.png",
                  "./example_pics/2.png"]

    segs = vf3d.FaceSegmentation()
    lmd = vf3d.FaceLandmarkDetector("3D")

    for id, imp in enumerate(image_path):
        lds = lmd.detect_face_landmark(imp)
        new_img, new_lds = vf3d.alignment_and_crop(imp, lds[0])
        new_img = new_img[0]

        cv2.imwrite("./cache.png", new_img)

        org_parsing, mask, mask_prob = segs.create_face_mask("./cache.png")
        vis_parsing = segs.visualize(org_parsing, "./cache.png")
        mask = (mask * 255).astype(np.uint8)
        mask_prob = (mask_prob * 255).astype(np.uint8)
        inputs = cv2.imread("./cache.png")


        cv2.imwrite("./example_results/{}_mask.png".format(id), mask)
        cv2.imwrite("./example_results/{}_mask_prob.png".format(id), mask_prob)
        cv2.imwrite("./example_results/{}_vis.png".format(id), vis_parsing)
        cv2.imwrite("./example_results/{}_input.png".format(id), inputs)

        os.remove("./cache.png")

if __name__ == '__main__':
    example_segmentation()