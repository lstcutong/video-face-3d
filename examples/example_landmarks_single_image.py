import VideoFace3D as vf3d
import cv2
import os
def example_landmarks():
    ld_2d = vf3d.FaceLandmarkDetector("2D")
    ld_3d = vf3d.FaceLandmarkDetector("3D")
    image_path = "./example_pics/1.png"

    l2d = ld_2d.detect_face_landmark(image_path)
    l3d = ld_3d.detect_face_landmark(image_path)

    im2d = vf3d.draw_landmarks(cv2.imread(image_path), l2d, colors=[vf3d.ComfortableColor().mint_d.to_bgr()])
    im3d = vf3d.draw_landmarks(cv2.imread(image_path), l3d, colors=[vf3d.ComfortableColor().mint_d.to_bgr()])

    cv2.imwrite("./example_results/lanmark_2D.png", im2d)
    cv2.imwrite("./example_results/lanmark_3D.png", im3d)

if __name__ == '__main__':
    example_landmarks()