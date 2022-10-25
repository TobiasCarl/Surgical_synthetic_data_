from ast import Tuple
from calendar import c
from cv2 import IMWRITE_PNG_STRATEGY_FILTERED
import cv2
import skimage.exposure
import os
import numpy as np
from os import listdir
import argparse


def extract_obj_from_greenscreen(file_path):
    img = cv2.imread(file_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab[:, :, 1]
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=5,
                            sigmaY=5, borderType=cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(blur, in_range=(
        127.5, 255), out_range=(0, 255)).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    x, y, w, h = cv2.boundingRect(result[..., 3])
    cropped_img = result[y:y+h, x:x+w, :]
    return cropped_img


def extract_object_from_images(input_folder, output_folder):
    if (not input_folder.endswith("/")):
        input_folder += "/"
    if (not output_folder.endswith("/")):
        output_folder += "/"
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            cropped_img = extract_obj_from_greenscreen(input_folder+filename)
            filename = filename.strip(".jpg")
            cv2.imwrite(output_folder+'trans'+filename+".png", cropped_img)
            cv2.destroyAllWindows()
            continue
        else:
            continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder',
                        help='Input folder path', required=True)
    parser.add_argument('-o', '--output_folder',
                        help='Output folder path', required=True)
    args = parser.parse_args()

    extract_object_from_images(args.input_folder, args.output_folder)
