import argparse
import glob
import os
import xml.etree.ElementTree as ET

from ast import Tuple
from calendar import c
from cv2 import IMWRITE_PNG_STRATEGY_FILTERED
import cv2
import skimage.exposure
import numpy as np
from os import listdir
import argparse
import albumentations as A
from PIL import Image
from scipy import ndimage
import random as rng



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj', '--object_folder', required=True, action='store', default='.', help="folder with object images with green screen background")
    parser.add_argument('-bg', '--background_folder', required=True, action='store', default='.', help="Folder with background images")
    parser.add_argument('-out', '--output_folder', required=True, action='store', default='.', help="Folder where generated images are stored")
    parser.add_argument('-generate', '--generate', required=True, action='store', default='.', help="How many image to generate")
    parser.add_argument('-max_obj', '--max_obj', required=True, action='store', default='.', help="Max number of objects per image")

    
    return parser.parse_args()

def image_augmentation(image):
    transform = A.Compose([
    #A.RandomCrop(width=450, height=450),
    #A.HorizontalFlip(p=0.5),
    #A.VerticalFlip(p=0.5),
    #A.RandomBrightnessContrast(p=1),
    A.augmentations.transforms.ImageCompression(quality_lower=10, quality_upper=100, always_apply=False, p=0.5),
    A.augmentations.transforms.ISONoise(),
    A.RandomBrightnessContrast(),
    A.augmentations.transforms.HueSaturationValue(),
    A.augmentations.transforms.ColorJitter(),
    A.augmentations.transforms.Downscale(scale_min=0.10, scale_max=0.75, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5)
    #A.augmentations.blur.transforms.MotionBlur(blur_limit=7, allow_shifted=True, always_apply=False, p=0.5),
    #A.augmentations.transforms.JpegCompression(quality_lower=15, quality_upper=15, always_apply=False, p=0.5)
    #A.augmentations.transforms.RandomShadow(shadow_roi=(0.3, 0.5, 0.4, 0.8), 
    #                                        num_shadows_lower=1, num_shadows_upper=2, 
    #                                        shadow_dimension=20, always_apply=False, p=0.9),
    #A.augmentations.transforms.RingingOvershoot()
    #A.augmentations.transforms.ColorJitter()
    
    #A.augmentations.transforms.ImageCompression(quality_lower=quality, quality_upper=quality, always_apply=False, p=1),
    #A.augmentations.transforms.Downscale(scale_min=0.25, scale_max=0.25, interpolation=None, always_apply=False, p=1)
    #A.augmentations.blur.transforms.MotionBlur()
    ], bbox_params=A.BboxParams(format='yolo'))


    #Notera, vi skickar för tillfället inte in några bounding boxes
    transformed = transform(image=image, bboxes=[])
    transformed_image = transformed['image']
    return transformed_image


def extract_obj_from_greenscreen_and_add_alpha_channel(cv2_img):
    #convert to LAB color space
    lab = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2LAB)
    #Coordinate that represents the position between red and green
    A = lab[:, :, 1]
    #Otsu's method of thresholding, threshold value is automatically calculated
    #then values below threshold are set to 0 and values above are set to the maxval argument
    A = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #smooth out the edges with a gaussian blur
    blur = cv2.GaussianBlur(A, (0, 0), sigmaX=5,
                            sigmaY=5, borderType=cv2.BORDER_DEFAULT)
    #What does resacle_intensity do?
    mask = skimage.exposure.rescale_intensity(blur, in_range=(
        127.5, 255), out_range=(0, 255)).astype(np.uint8)
    #Adds an alpha channel to the in input image
    result = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2BGRA)
    #Use the mask as the alpha channel
    result[:, :, 3] = mask
    #Calculate a bounding box for the mask, i.e. the alpha channel
    x, y, w, h = cv2.boundingRect(result[..., 3])
    cropped_img = result[y:y+h, x:x+w, :]
    return cropped_img

def initiate_annotation(final_image_path, img_width, img_height):
    tree = ET.ElementTree(ET.fromstring("<annotation> </annotation>"))
    root = tree.getroot()
    ET.SubElement(root, "folder").text = "image"
    ET.SubElement(root, "filename").text = os.path.basename(final_image_path)
    ET.SubElement(root, "path").text = "image"
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    return tree

def add_annotation(tree, xmin, ymin, xmax, ymax, class_name):
    root = tree.getroot()
    new_object = ET.SubElement(root, "object")
    ET.SubElement(new_object, "name").text = class_name
    ET.SubElement(new_object, "pose").text = "Unspecified"
    ET.SubElement(new_object, "truncated").text = "0"
    ET.SubElement(new_object, "difficult").text = "0"
    bnd_box = ET.SubElement(new_object, "bndbox")
    ET.SubElement(bnd_box, "xmin").text = str(xmin)
    ET.SubElement(bnd_box, "ymin").text = str(ymin)
    ET.SubElement(bnd_box, "xmax").text = str(xmax)
    ET.SubElement(bnd_box, "ymax").text = str(ymax)

def save_annotations(tree, final_image_path):
    xml_path = final_image_path[:-4] + ".xml"
    tree.write(xml_path)



'''
Calculates a mask for objects that are in front of a green screen. Then augmentations are applied to the original image
(the object plus in front of the green screen). The image is saved together with the mask, which is used as the alpha channel. Then 
it is converted into PIL-format and is rotated and scaled and finally pasted onto a background image. An xml file is generated
for every background image containing bounding box data for each object.
'''
def generate_images(object_folder, background_folder, output_folder, generate, max_obj):

    if not object_folder.endswith("/"):
        object_folder += "/"
    if not background_folder.endswith("/"):
        background_folder += "/"
    if not output_folder.endswith("/"):
        output_folder += "/"

    nbr_images_to_generate = int(generate)
    max_objects_per_image = int(max_obj)

    object_image_paths = glob.glob(object_folder + "*.jpg") #assumes object images are in jpg format
    background_image_paths = glob.glob(background_folder + "*.jpg") #assumes background images are in jpg format

    #List of indexes, each index represents one object image
    remaining_objects_idx = [x for x in range(len(object_image_paths))]
    for idx in range(nbr_images_to_generate):
        background_image_path = background_image_paths[idx % len(background_image_paths)]
        pil_background_image = Image.open(background_image_path)

        #Creates a new xml file with the same tags as generated by labelImg.
        tree = initiate_annotation(output_folder + os.path.basename(background_image_path)[:-4] + ".png", pil_background_image.width, pil_background_image.height)

        #Randomly selects a number of object images that will be placed in the background image
        nbr_images = rng.randint(0,max_objects_per_image)
        print("Generating image:", idx+1, "with", nbr_images, "images in it")
        if nbr_images <= len(remaining_objects_idx):
            indicies = rng.sample(remaining_objects_idx, k=nbr_images)
        else:
            remaining_objects_idx = [x for x in range(len(object_image_paths))]
            indicies = rng.sample(remaining_objects_idx, k=nbr_images)
        for i in sorted(indicies, reverse=True):
            print(remaining_objects_idx, i)
            remaining_objects_idx.remove(i)
        
        #Loop over each selected object image and perform object extraction and then augmentation and saves the resulting image
        #as a png. Then open the image as PIL, rotate and scale it and place it on the background image.
        for i in indicies:
            object_image_path = object_image_paths[i]

            cv2_img = cv2.imread(object_image_path)
            cv2_img_cropped = extract_obj_from_greenscreen_and_add_alpha_channel(cv2_img)
            cv2_img_cropped_augmented = image_augmentation(cv2_img_cropped[:,:,0:3]) #alpha channel removed
            cv2_img_cropped_augmented = cv2.cvtColor(cv2_img_cropped_augmented, cv2.COLOR_BGR2BGRA)
            cv2_img_cropped_augmented[:, :, 3] = cv2_img_cropped[:,:,3] #alpha channel restored

            #save as png
            temp_path = object_image_path.replace(os.path.basename(object_image_path), "temp_img_61045618034.png")
            cv2.imwrite(temp_path, cv2_img_cropped_augmented)


            
            background_width = pil_background_image.width
            background_height = pil_background_image.height

            #open the augmentated object image in PIL format, easier to perform rotations and add images onto each other using PIL
            #than using opencv.
            pil_img = Image.open(temp_path)
            pil_img = pil_img.rotate(rng.randint(0,360), expand=True)
            

            #scale the object image so that it fits inside the background image
            object_width = pil_img.width
            object_height = pil_img.height
            if object_width/background_width >= object_height/background_height:
                ratio = object_width/background_width
            else:
                ratio = object_height/background_height

            max_scaling = 10*ratio
            scaling = max(max(3, ratio), max_scaling*rng.random()) #make sure the object image is smaller than the background image
            pil_img = pil_img.resize((int(pil_img.width/scaling) ,int(pil_img.height/scaling)),  Image.Resampling.LANCZOS)


            #Randomly paste the object image onto the background image.
            #NOTE: Does not check if objects will overlap
            object_width = pil_img.width
            object_height = pil_img.height
            img_pos_x = rng.randint(0,background_width - object_width)
            img_pos_y = rng.randint(0, background_height - object_height)

            #third argument uses the alpha channel of the image as a mask
            pil_background_image.paste(pil_img, (img_pos_x, img_pos_y), pil_img)

            #Calculate the bounding box for the object based on the alpha channel
            #A for alpha channel
            alpha_channel = pil_img.getchannel("A")
            cv2_img = np.array(alpha_channel)
            bbox_xmin, bbox_ymin, bbox_width, bbox_height = cv2.boundingRect(cv2_img)

            xmin = img_pos_x + bbox_xmin+1
            ymin = img_pos_y + bbox_ymin+1
            xmax = xmin + bbox_width-1
            ymax = ymin + bbox_height-1
            add_annotation(tree, xmin, ymin, xmax, ymax, "forceps")
            
            
        pil_background_image.save(output_folder + os.path.basename(background_image_path).replace(".jpg", ".png"))
        save_annotations(tree, output_folder + os.path.basename(background_image_path)[:-4] + ".png")
            
                


def main():

    args = get_args()
    generate_images(args.object_folder, args.background_folder, args.output_folder, args.generate, args.max_obj)
    

if __name__ == "__main__":
    main()