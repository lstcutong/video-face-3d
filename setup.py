from setuptools import find_packages, setup

files =["data/*"]

setup(
    description='utils for video face data preprocess, include video face tracking, landmark detecting, morphable face model fitting',
    author='Luo Shoutong',
    author_email='MF1933071@smail.nju.edu.cn',
    license='Magic',
    version='0.1.0',
    name='VideoFace3D',
    include_package_data = True,
    packages=find_packages(),
)

'''
"./data/BFM_front_idx.mat",
"./data/BFM_model_front.mat",
"./data/FaceReconModel.pb",
"./data/phase1_wpdc_vdc.pth.tar",
"./data/shape_predictor_68_face_landmarks.dat",
"./data/similarity_Lm3D_all.mat",
"./data/tri.mat",
"./data/keypoints_sim.npy",
"./data/Model_PAF.pkl",
"./data/param_whitening.pkl",
"./data/pncc_code.npy",
"./data/u_exp.npy",
"./data/u_shp.npy",
"./data/w_exp_sim.npy",
"./data/w_shp_sim.npy"
'''