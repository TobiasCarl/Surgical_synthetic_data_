import argparse
import glob
import os
from timeit import timeit
import xml.etree.ElementTree as ET
import random

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
    A.augmentations.transforms.ImageCompression(quality_lower=25, quality_upper=100, always_apply=False, p=0.5),
    A.augmentations.transforms.ISONoise(),
    A.RandomBrightnessContrast(),
    A.augmentations.transforms.HueSaturationValue(),
    A.augmentations.transforms.ColorJitter(),
    A.augmentations.transforms.Downscale(interpolation=cv2.INTER_LINEAR, p=0.3)
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

def is_overlap(l1, r1, l2, r2):
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    return True

def replace_green_background_with_white(cv2_img):
    alpha_channel = 255 - cv2_img[:, :, 3]
    for i in range(3):
        cv2_img[: ,: , i] = cv2.add(cv2_img[: ,: , i], alpha_channel)
    return cv2_img

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
        tree = initiate_annotation(output_folder + os.path.basename(background_image_path)[:-4] + ".jpg", pil_background_image.width, pil_background_image.height)

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
        list_of_objects_on_backg = []
        for i in indicies:
            object_image_path = object_image_paths[i]

            classes = ["diathermy", "needle_driver", "forceps"]
            class_name = " "
            for cls in classes:
                if cls in object_image_path:
                    class_name = cls

            cv2_img = cv2.imread(object_image_path, cv2.IMREAD_UNCHANGED)
            cv2_img = extract_obj_from_greenscreen_and_add_alpha_channel(cv2_img)
            cv2_img = replace_green_background_with_white(cv2_img)
            cv2_img_aug = image_augmentation(cv2_img[:,:,0:3]) #alpha channel removed
            cv2_img_aug = cv2.cvtColor(cv2_img_aug, cv2.COLOR_BGR2BGRA)
            cv2_img_aug[:, :, 3] = cv2_img[:,:,3] #alpha channel restored
            cv2_img = cv2_img_aug

            #save as png
            #temp_path = object_image_path.replace(os.path.basename(object_image_path), "temp_img_61045618034.png")
            #cv2.imwrite(temp_path, cv2_img_cropped_augmented)


            
            background_width = pil_background_image.width
            background_height = pil_background_image.height

            #open the augmentated object image in PIL format, easier to perform rotations and add images onto each other using PIL
            #than using opencv. 
            #Converting cv2 image to pil
            pil_img = Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGBA))
            pil_img = pil_img.rotate(rng.randint(0,360), expand=True)
            

            #scale the object image so that it fits inside the background image
            
            object_width = pil_img.width
            object_height = pil_img.height
            if object_width/background_width >= object_height/background_height:
                ratio = object_width/background_width
            else:
                ratio = object_height/background_height

            #Rescale to same size as background
            pil_img= pil_img.resize((int(pil_img.width/ratio) ,int(pil_img.height/ratio)),  Image.Resampling.LANCZOS)

            #Randomly paste the object image onto the background image.
            #NOTE: It does not check if objects will overlap
            

            while True:
                #What factor to use for downscale
                scale_fac =random.randint(3,7)
                pil_img_cp = pil_img.resize((int(pil_img.width/scale_fac) ,int(pil_img.height/scale_fac)),  Image.Resampling.LANCZOS)
                object_width = pil_img_cp.width
                object_height = pil_img_cp.height
                img_pos_x = rng.randint(0,background_width - object_width)
                img_pos_y = rng.randint(0, background_height - object_height)
                l2, r2 = (img_pos_x, img_pos_y), (img_pos_x+object_width, img_pos_y+object_height)

                if all(not is_overlap(l1, r1, l2, r2) for l1, r1 in list_of_objects_on_backg):
                    list_of_objects_on_backg.append((l2, r2))
                    break

                    
            #third argument uses the alpha channel of the image as a mask
            pil_background_image.paste(pil_img_cp, (img_pos_x, img_pos_y), pil_img_cp)

            #Calculate the bounding box for the object based on the alpha channel
            #A for alpha channel
            alpha_channel = pil_img_cp.getchannel("A")
            cv2_img = np.array(alpha_channel)
            bbox_xmin, bbox_ymin, bbox_width, bbox_height = cv2.boundingRect(cv2_img)

            xmin = img_pos_x + bbox_xmin+1
            ymin = img_pos_y + bbox_ymin+1
            xmax = xmin + bbox_width-1
            ymax = ymin + bbox_height-1
            add_annotation(tree, xmin, ymin, xmax, ymax, class_name)

            
            
        pil_background_image.save(output_folder + os.path.basename(background_image_path).replace(".jpg", ".jpg"))
        save_annotations(tree, output_folder + os.path.basename(background_image_path)[:-4] + ".jpg")
            
                


def main():

    args = get_args()
    generate_images(args.object_folder, args.background_folder, args.output_folder, args.generate, args.max_obj)
    

if __name__ == "__main__":
    main()