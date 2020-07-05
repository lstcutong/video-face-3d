import VideoFace3D as vf3d
import cv2
import numpy as np

def SfSTest():
    fsp = vf3d.FaceSfSPipline()
    lmd = vf3d.FaceLandmarkDetector("3D")

    image_path = [
        "./example_pics/1.png",
        "./example_pics/2.png"
    ]

    ncount = 0
    for im_path in image_path:
        ncount += 1
        lds = lmd.detect_face_landmark(im_path)
        new_img, new_lds = vf3d.alignment_and_crop(im_path, lds[0])
        new_img = new_img[0]

        norm, albedo, light = fsp.disentangle(new_img)

        Irec, Ishd = vf3d.create_shading_recon(norm, albedo, light)

        from PIL import Image
        Irec = np.array(Image.fromarray((Irec.clip(0,1)*255).astype(np.uint8)).resize((224,224),Image.ANTIALIAS))
        norm = np.array(Image.fromarray((norm.clip(0,1)*255).astype(np.uint8)).resize((224,224),Image.ANTIALIAS))
        albedo = np.array(Image.fromarray((albedo.clip(0,1)*255).astype(np.uint8)).resize((224,224),Image.ANTIALIAS))
        Ishd = np.array(Image.fromarray((Ishd.clip(0,1)*255).astype(np.uint8)).resize((224,224),Image.ANTIALIAS))
        

        im = np.column_stack([new_img, albedo, norm, Ishd, Irec])
        cv2.imwrite("./example_results/sfs_{}.png".format(ncount), im)

if __name__ == '__main__':
    SfSTest()