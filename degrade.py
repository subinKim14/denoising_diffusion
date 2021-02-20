'''
degrade.py: Apply degradations on clean images to acquire paired training samples.
Please modify the degradation type and source image directory before applying it.
'''
import cv2
import os
from tqdm import tqdm
import numpy as np
import imgaug.augmenters as ia
import matplotlib.pyplot as plt

def get_bicubic_blur(src, severity=2):
    blur = src
    for i in range(severity):
        blur = cv2.resize(blur, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    for i in range(severity):
        blur = cv2.resize(blur, dsize=(0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur = np.clip(blur, 0, 1)
    return blur

def get_gaussian_blur(src, severity=1):
    kernel = np.ones((5, 5), np.float32)/25
    blur = src
    for i in range(severity):
        blur = cv2.filter2D(blur, -1, kernel)
    blur = np.clip(blur, 0, 1)
    return blur

def get_motion_blur():
    return ia.OneOf([
        ia.MotionBlur(k=(10,20)),
        ia.GaussianBlur((3.0, 8.0)),
    ])

def create_mixed_dataset(input_dir, suffix='bicubic', severity=2):
    output_dir = input_dir + '_' + suffix + '_' + str(severity)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    
    if suffix == 'bicubic':
        for item in tqdm(os.listdir(input_dir)):
            hr = plt.imread(os.path.join(input_dir, item))[:, :, :3]
            lr = get_bicubic_blur(hr, severity=severity)
            plt.imsave(os.path.join(output_dir, item), lr)

    elif suffix == "gaussian":
        for item in tqdm(os.listdir(input_dir)):
            hr = plt.imread(os.path.join(input_dir, item))[:, :, :3]
            lr = get_gaussian_blur(hr, severity=severity)
            plt.imsave(os.path.join(output_dir, item), lr)

    elif suffix == "motion":
        trans = get_motion_blur() # or use other functions
        mix_degrade = lambda x: trans.augment_image(x)
        for item in tqdm(os.listdir(input_dir)):
            hr = cv2.imread(os.path.join(input_dir, item))
            lr = mix_degrade(hr)
            cv2.imwrite(os.path.join(output_dir, item), lr)

    else:
        print("[i]: suffix should be \'bicubic\', \'gaussian\' or \'motion\'.")

'''
if __name__ == '__main__':
    suffix = 'motion' # [bicubic/gaussian/motion]
    source_dir = './toy_originals'

    # severity option should be specified for bicubic and gaussian blur. (default value is 2)
    create_mixed_dataset(source_dir, suffix, severity=1)
    #create_mixed_dataset(source_dir, suffix, severity=2)
'''