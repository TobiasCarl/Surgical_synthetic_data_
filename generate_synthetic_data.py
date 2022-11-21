import argparse
import cv2
import os
import numpy as np
import albumentations as Al
import skimage.exposure
import random
import xml.etree.ElementTree as ET
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-obj_f', '--object_folder',nargs='+', required=True, action='store', default='.', 
        help="a list of folders, one for each class, with object images with green screen background")
    parser.add_argument('-obj_c', '--object_classes',nargs='+', required=True, action='store', default='.', 
        help="a list of class names corresponding to the list of object folders")
    parser.add_argument('-bg', '--background_folder', required=True, action='store', default='.', 
        help="Folder with background images")
    parser.add_argument('-out', '--output_folder', required=True, action='store', default='.', 
        help="Folder where generated images are stored")
    parser.add_argument('-generate', '--generate', required=True,type=int, action='store', default='.', 
        help="How many images to generate")
    parser.add_argument('-max_obj', '--max_obj', required=True,type=int, action='store', default='.', 
        help="Max number of objects per image")
    parser.add_argument('-max_overlap', '--max_ovp', required=True,type=float, action='store', default='.', 
        help="Max overlap of objects per image")

    return parser.parse_args()

'''
First the image is randomly flipped and cropped, then the object is extracted and a mask is generated
'''
def get_img_and_mask(img_path):
    cv2_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    flip = Al.Compose([
        Al.HorizontalFlip(p=0.5),
        Al.VerticalFlip(p=0.5)
    ], bbox_params=Al.BboxParams(format='yolo'))

    crop = Al.Compose([
        Al.RandomCropFromBorders(crop_left=random.uniform(0.2, 0.4), crop_right=random.uniform(
            0.2, 0.4), crop_top=random.uniform(0.2, 0.4), crop_bottom=random.uniform(0.2, 0.4), always_apply=False, p=1.0),
    ], bbox_params=Al.BboxParams(format='yolo'))

    cv2_img = flip(image=cv2_img, bboxes=[])["image"]
    cv2_img = crop(image=cv2_img, bboxes=[])["image"]
    lab = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2LAB)
    # Coordinate that represents the position between red and green
    #with red background, use: A = 255 - lab[:, :, 1]
    A = 255 - lab[:, :, 1]
    # Otsu's method of thresholding, threshold value is automatically calculated
    # then values below threshold are set to 0 and values above are set to the maxval argument
    A = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    # smooth out the edges with a gaussian blur
    blur = cv2.GaussianBlur(A, (0, 0), sigmaX=5,
                            sigmaY=5, borderType=cv2.BORDER_DEFAULT)
    # What does resacle_intensity do?
    mask = skimage.exposure.rescale_intensity(blur, in_range=(
        127.5, 255), out_range=(0, 255)).astype(np.uint8)
    
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_b = mask[:, :, 0] == 0  
    mask = mask_b.astype(np.uint8)  

    return cv2_img, mask



'''
Performs image augmentation and resizing
'''
def resize_transform_obj(img, mask, longest_min, longest_max, transforms=False):

    transform = Al.Compose([
        Al.RandomBrightnessContrast(),
        Al.RandomGamma(),
        Al.augmentations.transforms.Downscale(
            scale_min=0.30, scale_max=0.75, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5)

    ], bbox_params=Al.BboxParams(format='yolo'))
    transformed = transform(image=img, bboxes=[])
    transformed_image = transformed['image']
    h, w = transformed_image.shape[0], transformed_image.shape[1]
    longest, shortest = max(h, w), min(h, w)
    longest_new = np.random.randint(longest_min, longest_max)
    shortest_new = int(shortest * (longest_new / longest))

    if h > w:
        h_new, w_new = longest_new, shortest_new
    else:
        h_new, w_new = shortest_new, longest_new
    transform_resize = Al.Resize(
        h_new, w_new, interpolation=1, always_apply=False, p=1)

    transformed_resized = transform_resize(
        image=transformed_image, mask=mask)
    img_t = transformed_resized["image"]
    mask_t = transformed_resized["mask"]

    if transforms:
        transformed = transforms(image=img_t, mask=mask_t)
        img_t = transformed["image"]
        mask_t = transformed["mask"]

    return img_t, mask_t

'''
Adds an object to the background image 
'''
def add_obj(img_comp, mask_comp, img, mask, x, y, idx):
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

    x = x - int(w/2)
    y = y - int(h/2)

    mask_b = mask == 0
    mask_rgb_b = np.stack([mask_b, mask_b, mask_b], axis=2)

    if x >= 0 and y >= 0:

        # h_part - part of the image which gets into the frame of img_comp along y-axis
        h_part = h - max(0, y+h-h_comp)
        # w_part - part of the image which gets into the frame of img_comp along x-axis
        w_part = w - max(0, x+w-w_comp)

        img_comp[y:y+h_part, x:x+w_part, :] = img_comp[y:y+h_part, x:x+w_part, :] * \
            ~mask_rgb_b[0:h_part, 0:w_part, :] + \
            (img * mask_rgb_b)[0:h_part, 0:w_part, :]
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
        mask_comp[0:0+h_part, x:x+w_part] = mask_comp[0:0+h_part, x:x+w_part] * \
            ~mask_b[h-h_part:h, 0:w_part] + \
            (idx * mask_b)[h-h_part:h, 0:w_part]
        mask_added = mask[h-h_part:h, 0:w_part]

    return img_comp, mask_comp, mask_added


def check_areas(mask_comp, obj_areas, overlap_degree=0.3):
    obj_ids = np.unique(mask_comp).astype(np.uint8)[1:-1]
    #Här hämtas listan av vad som syns i respektive mask. De hämtas genom att pixlarna skiljer lite på varje mask.
    #Varje mask skiljer en pixel och mask_comp innehåller all info, alltså hur alla masks ligger på varandra.
    masks = mask_comp == obj_ids[:, None, None]

    ok = True

    if len(np.unique(mask_comp)) != np.max(mask_comp) + 1:
        ok = False
        return ok
    ##Varje masks synliga area finns i obj_areas. 
    for idx, mask in enumerate(masks):

        if np.count_nonzero(mask) / obj_areas[idx] < 1 - overlap_degree:
            ok = False
            break

    return ok

'''
Positions objects on a background image, they may overlap to a certain degree.
'''
def create_composition(obj_dict, img_comp_bg,  longest_min, longest_max,
                       max_objs=5,
                       overlap_degree=0.2,
                       max_attempts_per_obj=10
                       ):

    img_comp = img_comp_bg.copy()
    h, w = img_comp.shape[0], img_comp.shape[1]
    mask_comp = np.zeros((h, w), dtype=np.uint8)

    obj_areas = []

    labels_comp = []
    num_objs = np.random.randint(max_objs) + 2

    i = 1
    p = 1
    for _ in range(1, num_objs):

        obj_idx = np.random.randint(len(obj_dict)) + 1

        #Randomly places an object on the background and tries again if to much
        #overlap occured
        for _ in range(max_attempts_per_obj):

            imgs_number = len(obj_dict[obj_idx]['images'])
            idx = np.random.randint(imgs_number)
            img_path = obj_dict[obj_idx]['images'][idx]
            img, mask = get_img_and_mask(img_path)
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
                obj_areas.append(np.sum(mask_added == 0))
                labels_comp.append(obj_idx)
                i += 1
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
                #If new object is not vailidly placed, revert back to the old state
                ok = check_areas(mask_comp, obj_areas, overlap_degree)
                if ok:
                    obj_areas.append(np.sum(mask_added == 0))
                    labels_comp.append(obj_idx)
                    i += 1
                    break
                else:
                    p += 1

                    img_comp, mask_comp = img_comp_prev.copy(), mask_comp_prev.copy()

    return img_comp, mask_comp, labels_comp, obj_areas


def initiate_annotation(final_image_path, img_width, img_height):
    tree = ET.ElementTree(ET.fromstring("<annotation> </annotation>"))
    root = tree.getroot()
    ET.SubElement(root, "folder").text = os.path.dirname(
        final_image_path).split("/")[-1]
    ET.SubElement(root, "filename").text = os.path.basename(final_image_path)
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
    ET.SubElement(bnd_box, "xmin").text = str(xmin)
    ET.SubElement(bnd_box, "ymin").text = str(ymin)
    ET.SubElement(bnd_box, "xmax").text = str(xmax)
    ET.SubElement(bnd_box, "ymax").text = str(ymax)


def save_annotations(tree, final_image_path):
    xml_path = final_image_path[:-4] + ".xml"
    tree.write(xml_path)


def generate_images_and_xml(INPUT_LIST, BACKGROUND_PATH, NUMBER_OF_IMAGES, MAX_OBJECTS_PER_IMG, OUTPUT_FOLDER,OVERLAP_DEGREE=0.3):

    files_bg_imgs = os.listdir(BACKGROUND_PATH)
    files_bg_imgs = [os.path.join(BACKGROUND_PATH, f) for f in files_bg_imgs]

    INPUT_NUMBER = 1
    obj_dict = {}
    for k in INPUT_LIST:
        files_imgs = sorted(os.listdir(k[1]))
        files_imgs = [os.path.join(k[1], f) for f in files_imgs]
        obj_dict[INPUT_NUMBER] = {'folder': k[1], #folder path
                                  'class': k[0],  #class name
                                  'images': files_imgs} #image path
        INPUT_NUMBER += 1
    ##colors = {1: (255, 0, 0), 2: (0, 255, 0),
    ##          3: (0, 0, 255), 4: (0, 255, 255), 5: (255, 255, 0), 6: (255, 0, 255), 5: (255, 255, 255)}

    for x in range(NUMBER_OF_IMAGES):
        image_background_path = random.choice(files_bg_imgs)
        img_bg = cv2.imread(image_background_path)
        #img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB)
        h, w = img_bg.shape[0], img_bg.shape[1]
        mask_comp = np.zeros((h, w), dtype=np.uint8)
        img_comp = img_bg.copy()

        #Place objects on one background image
        img_comp, mask_comp, labels_comp, obj_areas = create_composition(obj_dict, img_bg, h*0.1, h*0.5,
                                                                         max_objs=MAX_OBJECTS_PER_IMG,
                                                                         overlap_degree=OVERLAP_DEGREE,
                                                                         max_attempts_per_obj=10)
        #save image
        img_comp_bboxes = img_comp.copy()
        new_output_name = os.path.join(OUTPUT_FOLDER, os.path.basename(
            image_background_path))[:-4]+"gen_"+str(x)+".png"
        cv2.imwrite(new_output_name, img_comp_bboxes)

        #save annotations
        obj_ids = np.unique(mask_comp).astype(np.uint8)[1:]
        masks = mask_comp == obj_ids[:, None, None]
        tree = initiate_annotation(new_output_name, w, h)

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            add_annotation(tree, xmin, ymin, xmax, ymax,
                obj_dict[labels_comp[i]]["class"])
            '''
            img_comp_bboxes = cv2.putText(img_comp_bboxes, obj_dict[labels_comp[i]]["class"],
                                          (xmin, ymin-20),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          1,
                                          colors[labels_comp[i]],
                                          5,
                                          2)
            img_comp_bboxes = cv2.rectangle(img_comp_bboxes,
                                            (xmin, ymin),
                                            (xmax, ymax),
                                            colors[labels_comp[i]],
                                            6)
            '''

        save_annotations(tree, new_output_name)
        #cv2.imwrite("result_img_"+str(x)+".png", img_comp_bboxes)

def main():

    args = get_args()
    count = 0
    test_input_list=[]
    #the object folders are matched with their class names
    for obj_folder in args.object_folder:
        test_input_list+= [(args.object_classes[count],obj_folder)]
        count+=1

    generate_images_and_xml(
        test_input_list, args.background_folder, args.generate, args.max_obj, args.output_folder,args.max_ovp
    )    

if __name__ == "__main__":
    main()





# forstsätt här: https://medium.com/@alexppppp/how-to-create-synthetic-dataset-for-computer-vision-object-detection-fd8ab2fa5249
