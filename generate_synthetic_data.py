import argparse
import cv2
import os
import numpy as np
import albumentations as Al
import skimage.exposure
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

from PIL import Image, ImageEnhance
import random as rand
import numpy
import time


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-config', '--config_path',  required=True, action='store', default='.',
                        help="Config file that describes augmentation parameters")
    return parser.parse_args()



def show_img(img):
    return
    img_h, img_w = img.shape[0], img.shape[1]
    print(f"image w: {img_w} h: {img_h}")
    win_name = "window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)

    max_size = 1080
    width_ratio, height_ratio = max_size/img_w, max_size/img_h
    scale_factor = min(1, width_ratio, height_ratio)
    window_w, window_h = int(scale_factor*img_w), int(scale_factor*img_h)
    print(f"window w: {window_w} h: {window_h}\n")

    cv2.resizeWindow(win_name, 3840, 2160) #resize must come after imshow for it to work properly
    cv2.waitKey(0)

def show_img_1(img):
    img_h, img_w = img.shape[0], img.shape[1]
    print(f"image w: {img_w} h: {img_h}")
    win_name = "window"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)

    max_size = 1080
    width_ratio, height_ratio = max_size/img_w, max_size/img_h
    scale_factor = min(1, width_ratio, height_ratio)
    window_w, window_h = int(scale_factor*img_w), int(scale_factor*img_h)
    print(f"window w: {window_w} h: {window_h}\n")

    cv2.resizeWindow(win_name, 1920, 1080) #resize must come after imshow for it to work properly
    cv2.waitKey(0)

#assumes input mask values are 0 and background values are 1
def show_mask(mask):
    mask = mask.copy()
    mask[mask == 0] = 255
    mask[mask == 1] = 0
    show_img(mask)



'''
Replace background of image with average color of cv2_img pixels that contain the mask.
'''
def replace_background_with_average_color(cv2_img, mask):
    for i in range(3):
        cv2_img[mask == 1, i] = np.mean(cv2_img[mask == 0, i])
    return cv2_img


'''
First the image is randomly flipped and cropped, then the object is extracted and a mask is generated
'''


def get_img_and_mask(img_path, color):

    cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


    lab = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2LAB)
    # Coordinate that represents the position between red and green
    # with red background, use: A = 255 - lab[:, :, 1]
    if (color == "red"):
        A = 255 - lab[:, :, 1]
    elif (color == "green"):
        A = lab[:, :, 1]

    # Otsu's method of thresholding, threshold value is automatically calculated
    # then values below threshold are set to 0 and values above are set to the maxval argument
    A = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # smooth out the edges with a gaussian blur
    blur = cv2.GaussianBlur(A, (0, 0), sigmaX=5,
                            sigmaY=5, borderType=cv2.BORDER_DEFAULT)
    # What does resacle_intensity do?
    mask = skimage.exposure.rescale_intensity(blur, in_range=(
        127.5, 255), out_range=(0, 255)).astype(np.uint8)

    cv2_img = replace_background_with_average_color(cv2_img, mask)

    show_img(cv2_img)
    show_img(mask)

    return cv2_img, mask


'''
Performs image augmentation and resizing
'''


def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False):

    transform = Al.Compose([
        Al.RandomBrightnessContrast(brightness_limit=(-0.5,-0.1), p=0.9),
        Al.RandomGamma(),
        Al.augmentations.transforms.Downscale(
            scale_min=0.30, scale_max=0.75, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5)

    ], bbox_params=Al.BboxParams(format='yolo'))
    flip = Al.Compose([
        Al.HorizontalFlip(p=0.5),
        Al.VerticalFlip(p=0.5),
    ], bbox_params=Al.BboxParams(format='yolo'))

    crop = Al.Compose([
        Al.RandomCropFromBorders(crop_left=random.uniform(0.2, 0.4), crop_right=random.uniform(
            0.2, 0.4), crop_top=random.uniform(0.2, 0.4), crop_bottom=random.uniform(0.2, 0.4), always_apply=False, p=1.0),
    ], bbox_params=Al.BboxParams(format='yolo'))

    flip_transform = flip(image=img, mask=mask, bboxes=[])
    crop_transform = crop(image=img, mask=mask, bboxes=[])
    img = flip_transform["image"]
    img = crop_transform["image"]
    mask = flip_transform["mask"]
    mask = crop_transform["mask"]
    #print("clip and flip mask")
    #show_img(mask)
    transformed = transform(image=img, bboxes=[])
    transformed_image = transformed['image']
    h, w = transformed_image.shape[0], transformed_image.shape[1]
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))

    '''if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
    transform_resize = Al.Resize(
        h_new, w_new, interpolation= cv2.INTER_LANCZOS4, always_apply=False, p=1)'''

    #transformed_resized = transform_resize(
    #    image=transformed_image.copy(), mask=mask.copy())
    #img_t = transformed_resized["image"]
    #mask_t = transformed_resized["mask"]





    pil_img = Image.fromarray(transformed_image.copy())
    pil_mask = Image.fromarray(mask)
    angle = rand.randint(0, 360)
    pil_img = pil_img.rotate(angle, expand=True)
    pil_mask = pil_mask.rotate(angle, expand=True)

    h, w = pil_img.height, pil_img.width
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new

    #show_img(np.array(pil_img))
    #show_img(mask_t)
    #print("PIL mask")
    #show_img(np.array(pil_mask))

    pil_img_cp = pil_img.resize((w_new ,h_new),  Image.Resampling.LANCZOS)
    pil_mask_cp = pil_mask.resize((w_new ,h_new),  Image.Resampling.LANCZOS)
    cv_img = np.array(pil_img_cp)
    cv_mask = np.array(pil_mask_cp)
    #show_img(img_t)
    #show_img(cv_img)
    #show_img(mask_t)
    #print("PIL mask")
    #show_img(cv_mask)

    #cv2.imshow("hej", m.astype(float))
    # cv2.waitKey(0)

    # Rotation by converting to PIL
    # -------------------------------------------------------------------------------------------------

    # changes the mask value from 0 because the rotated image will generate pixelvalues = 0
    #mask_t[mask_t == 0] = 100

    #pil_img = Image.fromarray(img_t)
    #pil_mask = Image.fromarray(mask_t)
    #angle = rand.randint(0, 360)
    #pil_img = pil_img.rotate(angle, expand=True)
    #pil_mask = pil_mask.rotate(angle, expand=True)

    img_t = cv_img
    mask_t = cv_mask
    # resets the original mask values to 0 and changes the newly added pixels to 1.
    #mask_t[mask_t == 0] = 1
    #mask_t[mask_t == 100] = 0
    # -------------------------------------------------------------------------------------------------

    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]

    #show_img(img_t)
    #print("resized and rotated mask")
    #show_img(mask_t)

    return img_t, mask_t


'''
Adds an object to the background image
'''



#With PIL
def add_obj(img_comp, mask_comp, img, mask, x, y, idx, is_noise=False):
    '''
    img_comp - composition of objects
    mask_comp - composition of objects` masks
    img - image of object
    mask - binary mask of object
    x, y - coordinates where center of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp
    '''
    h_comp, w_comp = img_comp.height, img_comp.width

    h, w = img.shape[0], img.shape[1]
    '''if (random.randint(0, 100) in range(0, int(100*0.5))):
        img = shade(img,random.uniform(0.1, 0.7))
    '''
    x = x - int(w/2)
    y = y - int(h/2)

    #creates a binary mask
    #temp_mask = cv2.cvtColor(mask.copy(), cv2.COLOR_RGB2BGR)
    #temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_BGR2RGB)
    mask_b = ~(mask == 0)
    #print(mask_b[0])
    #print(mask[100])
    #show_img_1(mask)
    mm = mask_b.astype(np.uint8)
    mm[mm == 1] = 255
    #show_img_1(mm)
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    mask_added = []

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    img[:,:,3] = mask
    img = Image.fromarray(img)

    if x >= 0 and y >= 0:

        # h_part - part of the image which gets into the frame of img_comp along y-axis
        
        h_part = h - max(0, y+h-h_comp)
        # w_part - part of the image which gets into the frame of img_comp along x-axis
        w_part = w - max(0, x+w-w_comp)

        #print(f"y:{y}, h:{h}, h_comp:{h_comp}, h_part:{h_part}")
        #print(f"img_w:{w}, img_h:{h}, paste_w:{w_part}, paste_h:{h_part}")
        #print()

        #img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * \
        #    ~mask_rgb_b[0:h_part, 0:w_part, :] + \
        #    (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        
        #img_comp.show()
        #import time
        #time.sleep(3)
        #img.show()
        #time.sleep(3)
        #print(f"img_comp {img_comp.width} {img_comp.height} img {img.width} {img.height}")
        #print(f"location: {(x, y, x+w_part, y+h_part)}")
        #print("\n x and y > 0 ")
        #print(f" paste into {(x, y, x+w_part, y+h_part)}\n")
        #print(f"background size: {img_comp.size} {img_comp.mode}, object size: {img.size} {img.mode}")
        cropped_obj = img.crop((0,0, w_part,h_part))
        img_comp.paste(cropped_obj, (x, y, x+w_part, y+h_part), cropped_obj)
        #cv2.imshow("hej", np.array(img_comp))
        #cv2.resizeWindow("hej", 3840, 2160)
        #cv2.waitKey(0)
        if (not is_noise):
            mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * \
                ~mask_b[0:h_part, 0:w_part] + \
                (idx * mask_b)[0:h_part, 0:w_part]
            mask_added = (~mask_b[0:h_part, 0:w_part]).astype(np.uint8)

    elif x < 0 and y < 0:

        h_part = h + y
        w_part = w + x

        #print("\n x and y < 0 ")
        #print(f" paste into {(0, 0, w_part, h_part)}\n")

        cropped_obj = img.crop((w-w_part,h-h_part, w,h))
        img_comp.paste(cropped_obj, (0, 0, w_part, h_part), cropped_obj)

        #img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * \
        #    ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + \
        #    (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
        if (not is_noise):

            mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * \
                ~mask_b[h-h_part:h, w-w_part:w] + \
                (idx * mask_b)[h-h_part:h, w-w_part:w]
            mask_added = (~mask_b[h-h_part:h, w-w_part:w]).astype(np.uint8)

    elif x < 0 and y >= 0:

        h_part = h - max(0, y+h-h_comp)
        w_part = w + x

        #print("\n x < 0 and y > 0 ")
        #print(f" paste into {(0, y, w_part, y+h_part)}\n")

        cropped_obj = img.crop((w-w_part,0, w,h_part))
        img_comp.paste(cropped_obj, (0, y, w_part, y+h_part), cropped_obj)

        #img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * \
        #    ~mask_rgb_b[0:h_part, w-w_part:w, :] + \
        #    (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
        if (not is_noise):

            mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * \
                ~mask_b[0:h_part, w-w_part:w] + \
                (idx * mask_b)[0:h_part, w-w_part:w]
            mask_added = (~mask_b[0:h_part, w-w_part:w]).astype(np.uint8)

    elif x >= 0 and y < 0:

        h_part = h + y
        w_part = w - max(0, x+w-w_comp)

        #print("\n x > 0 and y < 0 ")
        #print(f" paste into {(x, 0, x+w_part, h_part)}\n")

        cropped_obj = img.crop((0,h-h_part, w_part,h))
        img_comp.paste(cropped_obj, (x, 0, x+w_part, h_part), cropped_obj)

        #img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * \
        #    ~mask_rgb_b[h-h_part:h, 0:w_part, :] + \
        #    (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
        if (not is_noise):

            mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * \
                ~mask_b[h-h_part:h, 0:w_part] + \
                (idx * mask_b)[h-h_part:h, 0:w_part]
            mask_added = (~mask_b[h-h_part:h, 0:w_part]).astype(np.uint8)

    #show_img(np.array(img_comp))

    #show_img_1(mask_comp)
    #print(mask_added[0])

    return img_comp, mask_comp, mask_added


def add_obj_old(img_comp, mask_comp, img, mask,  x, y, idx, is_noise=False):
    '''
    img_comp - composition of objects
    mask_comp - composition of objects` masks
    img - image of object
    mask - binary mask of object
    x, y - coordinates where center of img is placed
    Function returns img_comp in CV2 RGB format + mask_comp
    '''
    h_comp, w_comp = img_comp.shape[0], img_comp.shape[1]

    h, w = img.shape[0], img.shape[1]
    '''if (random.randint(0, 100) in range(0, int(100*0.5))):
        img = shade(img,random.uniform(0.1, 0.7))
    '''
    x = x - int(w/2)
    y = y - int(h/2)
    mask_b = mask == 0
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)
    mask_added = []
    if x >= 0 and y >= 0:

        # h_part - part of the image which gets into the frame of img_comp along y-axis
        
        h_part = h - max(0, y+h-h_comp)
        # w_part - part of the image which gets into the frame of img_comp along x-axis
        w_part = w - max(0, x+w-w_comp)

        #print(f"y:{y}, h:{h}, h_comp:{h_comp}, h_part:{h_part}")
        #print(f"img_w:{w}, img_h:{h}, paste_w:{w_part}, paste_h:{h_part}")
        #print()

        img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * \
            ~mask_rgb_b[0:h_part, 0:w_part, :] + \
            (img * mask_rgb_b)[0:h_part, 0:w_part, :]
        if (not is_noise):
            mask_comp[y:y+h_part, x:x+w_part] = mask_comp[y:y+h_part, x:x+w_part] * \
                ~mask_b[0:h_part, 0:w_part] + \
                (idx * mask_b)[0:h_part, 0:w_part]
            mask_added = mask[0:h_part, 0:w_part]

    elif x < 0 and y < 0:

        h_part = h + y
        w_part = w + x

        img_comp[0:0+h_part, 0:0+w_part, :] = img_comp[0:0+h_part, 0:0+w_part, :] * \
            ~mask_rgb_b[h-h_part:h, w-w_part:w, :] + \
            (img * mask_rgb_b)[h-h_part:h, w-w_part:w, :]
        if (not is_noise):

            mask_comp[0:0+h_part, 0:0+w_part] = mask_comp[0:0+h_part, 0:0+w_part] * \
                ~mask_b[h-h_part:h, w-w_part:w] + \
                (idx * mask_b)[h-h_part:h, w-w_part:w]
            mask_added = mask[h-h_part:h, w-w_part:w]

    elif x < 0 and y >= 0:

        h_part = h - max(0, y+h-h_comp)
        w_part = w + x

        img_comp[y:y+h_part, 0:0+w_part, :] = img_comp[y:y+h_part, 0:0+w_part, :] * \
            ~mask_rgb_b[0:h_part, w-w_part:w, :] + \
            (img * mask_rgb_b)[0:h_part, w-w_part:w, :]
        if (not is_noise):

            mask_comp[y:y+h_part, 0:0+w_part] = mask_comp[y:y+h_part, 0:0+w_part] * \
                ~mask_b[0:h_part, w-w_part:w] + \
                (idx * mask_b)[0:h_part, w-w_part:w]
            mask_added = mask[0:h_part, w-w_part:w]

    elif x >= 0 and y < 0:

        h_part = h + y
        w_part = w - max(0, x+w-w_comp)

        img_comp[0:0+h_part, x:x+w_part, :] = img_comp[0:0+h_part, x:x+w_part, :] * \
            ~mask_rgb_b[h-h_part:h, 0:w_part, :] + \
            (img * mask_rgb_b)[h-h_part:h, 0:w_part, :]
        if (not is_noise):

            mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * \
                ~mask_b[h-h_part:h, 0:w_part] + \
                (idx * mask_b)[h-h_part:h, 0:w_part]
            mask_added = mask[h-h_part:h, 0:w_part]

    show_img(img_comp)

    return img_comp, mask_comp, mask_added


def check_areas(mask_comp, obj_areas, labels_comp=[], overlap_degree=0.3, i=10):
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]

    # Här hämtas listan av vad som syns i respektive mask. De hämtas genom att pixlarna skiljer lite på varje mask.
    # Varje mask skiljer en pixel och mask_comp innehåller all info, alltså hur alla masks ligger på varandra.
    #print("inside check area")
    #print(f"obj_ids:{obj_ids}")
    masks = mask_comp == obj_ids[:, None, None]
    #print("\n\n")
    #print(masks.shape)
    #print("\n\n")
    

    ok = True
    for idx, mask in enumerate(masks):
        #print(mask.shape)
        #show_img_1(mask)
        if (obj_areas[idx][0] != 0):
            if np.count_nonzero(mask) / obj_areas[idx][0] < 1 - overlap_degree:
                ok = False
                break

    return ok

def shade(imag, percent):
    """
    imag: the image which will be shaded
    percent: a value between 0 (image will remain unchanged
             and 1 (image will be blackened)
    """
    tinted_imag = imag * (1 - percent)
    return tinted_imag

'''
Positions objects on a background image, they may overlap to a certain degree.
'''

'''
longest_min and longesth_max are the minumum and maximum
'''
def create_composition(obj_dict, img_comp_bg,  longest_min, longest_max, noise_dict,
                       max_objs=5,
                       overlap_degree=0.2,
                       max_attempts_per_obj=10,
                       fac=0.5,saved_images_and_masks={}
                       ):

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h, w), dtype=np.uint8)

    obj_areas = []

    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2

    i = 1
    p = 1
    random_obj = 0

    '''
    1. We convert backround image to PIL format
    2. blurry mask is generated in 'get_img_and_mask'
    3. 'resize_transform_obj' transforms image and the blurry mask
    4. in add_obj, a binary mask is created from the blurry mask
    5. binary mask is used to check for overlap with other objects
    and the blurry mask is used to paste the object on the background image

    '''

    img_comp = Image.fromarray(cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB))


    for _ in range(1, num_objs):
        obj_idx = np.random.randint(len(obj_dict)) + 1
        noise_idx = np.random.randint(len(noise_dict)) + 1
        # Randomly places an object on the background and tries again if to much
        # overlap occured
        for _ in range(max_attempts_per_obj):

            imgs_number = len(obj_dict[obj_idx]['images'])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_idx]['images'][idx]
            ##If it is saved, take mask and image directly.
            if img_path in saved_images_and_masks:
                img,mask, = saved_images_and_masks[img_path]
            else:
                img, mask = get_img_and_mask(
                    img_path, obj_dict[obj_idx]["background_color"])
                saved_images_and_masks[img_path] = (img,mask)

            img, mask = resize_transform_obj(img,
                                             mask,
                                             longest_min,
                                             longest_max
                                             )

            x, y = np.random.randint(w), np.random.randint(h)

            if i == 1:

                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                obj_areas.append((np.sum(mask_added == 0), i))
                labels_comp.append((obj_idx, i))
                i += 1
                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()

                if (random.randint(0, 100) in range(0, int(100*fac))):
                    img_path = noise_dict[noise_idx]["images"][np.random.randint(
                        len(noise_dict[noise_idx]["images"]))]
                    if img_path in saved_images_and_masks:
                        img,mask = saved_images_and_masks[img_path]
                    else:
                        img, mask = get_img_and_mask(img_path, noise_dict[noise_idx]["background_color"])
                        saved_images_and_masks[img_path] = (img,mask)


                    img, mask = resize_transform_obj(img,
                                                     mask,
                                                     longest_min,
                                                     longest_max
                                                     )

                    img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                              mask_comp,
                                                              img,
                                                              mask,
                                                              x,
                                                              y,
                                                              15 + random_obj
                                                              )

                    noise_ok = check_areas(
                        mask_comp, obj_areas, labels_comp, overlap_degree, i)
                    if noise_ok:

                        obj_areas.append(
                            (np.sum(mask_added == 0), 15 + random_obj))
                        labels_comp.append((obj_idx, 15 + random_obj))
                        random_obj += 1
                        break
                    else:
                        img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()
                        break

                break
            else:

                img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                          mask_comp,
                                                          img,
                                                          mask,
                                                          x,
                                                          y,
                                                          i)
                ok = check_areas(mask_comp, obj_areas,
                                 labels_comp, overlap_degree, i)
                if ok:
                    obj_areas.append((np.sum(mask_added == 0), i))
                    labels_comp.append((obj_idx, i))

                    i += 1
                    if (random.randint(0, 100) in range(0, int(100*fac))):
                        

                        img_comp_prev, mask_comp_prev = img_comp.copy(), mask_comp.copy()
                        img_path = noise_dict[noise_idx]["images"][np.random.randint(
                            len(noise_dict[noise_idx]["images"]))]
                        if img_path in saved_images_and_masks:
                            img,mask = saved_images_and_masks[img_path]
                        else:
                            img, mask = get_img_and_mask(img_path, noise_dict[noise_idx]["background_color"])
                            saved_images_and_masks[img_path] = (img,mask)

                        img, mask = resize_transform_obj(img,
                                                         mask,
                                                         longest_min,
                                                         longest_max
                                                         )

                        img_comp, mask_comp, mask_added = add_obj(img_comp,
                                                                  mask_comp,
                                                                  img,
                                                                  mask,
                                                                  x,
                                                                  y,
                                                                  15 + random_obj
                                                                  )
                        noise_ok = check_areas(
                            mask_comp, obj_areas, labels_comp, overlap_degree, i)
                        if noise_ok:
                            obj_areas.append(
                                (np.sum(mask_added == 0), 15 + random_obj))
                            labels_comp.append((obj_idx, 15 + random_obj))
                            random_obj += 1
                            break
                        else:
                            img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()
                            break
                    break

                else:
                    p += 1
                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()

    return cv2.cvtColor(np.array(img_comp), cv2.COLOR_BGR2RGB), mask_comp, labels_comp,saved_images_and_masks


def initiate_annotation(final_image_path, img_width, img_height):
    tree = ET.ElementTree(
        ET.fromstring("<annotation> </annotation>"))
    root = tree.getroot()
    ET.SubElement(root, "folder").text = os.path.dirname(
        final_image_path).split("/")[-1]
    ET.SubElement(root, "filename").text = os.path.basename(
        final_image_path)
    ET.SubElement(root, "path").text = final_image_path
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

    #labelImg starts indexing at 1
    xmin = 1 if xmin == 0 else xmin
    ymin = 1 if ymin == 0 else ymin

    ET.SubElement(bnd_box, "xmin").text = str(xmin)
    ET.SubElement(bnd_box, "ymin").text = str(ymin)
    ET.SubElement(bnd_box, "xmax").text = str(xmax)
    ET.SubElement(bnd_box, "ymax").text = str(ymax)


def save_annotations(tree, final_image_path):
    xml_path = final_image_path[:-4] + ".xml"
    tree.write(xml_path)


def generate_images_and_xml(INPUT_LIST, BACKGROUND_LIST, NOISE_LIST, NUMBER_OF_IMAGES, MAX_OBJECTS_PER_IMG, OUTPUT_FOLDER, OVERLAP_DEGREE=0.3):

    files_bg_imgs = []
    for k in BACKGROUND_LIST:
        temp_files_bg_imgs = os.listdir(k[0])
        files_bg_imgs += [os.path.join(k[0], f) for f in temp_files_bg_imgs]

    noise_dict = {}
    ##If we load the image and retrive mask we put it in dictionary. To make it run faster
    saved_images_and_masks={}
    saved_backgrounds={}
    NOISE_NUMBER = 1
    for k in NOISE_LIST:
        files_noise = sorted(os.listdir(k[0]))
        files_imgs = [os.path.join(k[0], f) for f in files_noise]
        noise_dict[NOISE_NUMBER] = {'folder': k[0],  # folder path
                                    'images': files_imgs,
                                    'background_color': k[1]}  # image path
        NOISE_NUMBER += 1
    INPUT_NUMBER = 1
    obj_dict = {}
    for k in INPUT_LIST:
        files_imgs = sorted(os.listdir(k[0]))
        files_imgs = [os.path.join(k[0], f) for f in files_imgs]
        obj_dict[INPUT_NUMBER] = {'folder': k[0],  # folder path
                                  'class': k[1],  # class name
                                  'images': files_imgs,
                                  'background_color': k[2]}  # image path
        INPUT_NUMBER += 1
    colors = {1: (255, 0, 0), 2: (0, 255, 0),
              3: (0, 0, 255), 4: (0, 255, 255), 5: (255, 255, 0), 6: (255, 0, 255), 7: (255, 255, 255)}

    for x in tqdm(range(NUMBER_OF_IMAGES), desc="Loading..."):

        image_background_path = random.choice(files_bg_imgs)
        if image_background_path in saved_backgrounds:
            img_bg,h,w = saved_backgrounds[image_background_path]
        else:

            img_bg = cv2.imread(image_background_path)
            h, w = img_bg.shape[0], img_bg.shape[1]
            saved_backgrounds[image_background_path]=(img_bg,h,w)
        mask_comp = np.zeros((h, w), dtype=np.uint8)
        img_comp = img_bg.copy()

        img_comp, mask_comp, labels_comp,saved_images_and_masks = create_composition(obj_dict, img_bg, h*0.1, h*0.5, noise_dict=noise_dict,
                                                                         max_objs=MAX_OBJECTS_PER_IMG,
                                                                         overlap_degree=OVERLAP_DEGREE,
                                                                         max_attempts_per_obj=10,saved_images_and_masks=saved_images_and_masks)

        #show_img_1(mask_comp)
        img_comp_bboxes = img_comp.copy()
        new_output_name = os.path.join(OUTPUT_FOLDER, os.path.basename(
            image_background_path))[:-4]+"gen_"+str(x)+".jpg"
        cv2.imwrite(new_output_name, img_comp_bboxes)

        # save annotations
        obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
        masks = mask_comp == obj_ids[:, None, None]
        tree = initiate_annotation(new_output_name, w, h)

        result = np.where(obj_ids > 14)
        obj_ids = np.delete(obj_ids, result[0])

        deleteInd = []
        ind = 0
        for label in labels_comp:
            if label[1] > 14:
                deleteInd.append(ind)
            ind += 1

        for index in sorted(deleteInd, reverse=True):
            del labels_comp[index]

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            try:
                add_annotation(tree, xmin, ymin, xmax, ymax,
                               obj_dict[labels_comp[i][0]]["class"])
            except:
                print("An exception occurred: ", i)

            img_comp_bboxes = cv2.putText(img_comp_bboxes, obj_dict[labels_comp[i][0]]["class"],
                                          (xmin, ymin-20),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          1,
                                          colors[labels_comp[i][0]],
                                          2,
                                          2)
            img_comp_bboxes = cv2.rectangle(img_comp_bboxes,
                                            (xmin, ymin),
                                            (xmax, ymax),
                                            colors[labels_comp[i][0]],
                                            2)

        save_annotations(tree, new_output_name)


def main():

    args = get_args()

    config_desc = {"Object_desc": [], "Noise_desc": [], "Background_folder": [], "Output_folder": [
    ], "Generate_quantity": [], "Max_objects_per_image": [], "Max_overlap": []}
    current = ""
    with open(args.config_path) as config_txt:
        for line in config_txt:
            if line.strip() == "Object_desc" or line.strip() == "Noise_desc" or line.strip() == "Background_folder" or line.strip() == "Output_folder" or line.strip() == "Generate_quantity" or line.strip() == "Max_objects_per_image" or line.strip() == "Max_overlap":
                current = line.strip()
            elif current != "" and line.strip() != "":
                config_desc[current].append((line.split('\n')[0].split(":")))

    generate_images_and_xml(config_desc["Object_desc"], config_desc["Background_folder"], config_desc["Noise_desc"], int(config_desc["Generate_quantity"][0][0]), int(
        config_desc["Max_objects_per_image"][0][0]), config_desc["Output_folder"][0][0], float(config_desc["Max_overlap"][0][0]))


if __name__ == "__main__":
    main()


# forstsätt här: https://medium.com/@alexppppp/how-to-create-synthetic-dataset-for-computer-vision-object-detection-fd8ab2fa5249