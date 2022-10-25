import argparse
import glob
import os
from PIL import Image
import random

"""
[ref](https://www.geeksforgeeks.org/find-two-rectangles-overlap/)
"""
def is_overlap(l1, r1, l2, r2):
    if l1[0] > r2[0] or l2[0] > r1[0]:
        return False

    if l1[1] > r2[1] or l2[1] > r1[1]:
        return False

    return True
def get_refactored_img(background,image,fac_from,fac_to):
    width_back,height_back=background.size

    width,height = image.size
    facHeight= height/height_back
    facWidth= width/width_back

    new_fac =random.randint(fac_from, fac_to)
    image=image.resize((int((int(width/new_fac)/max(facWidth,facHeight))),int((int(height/new_fac))/max(facWidth,facHeight))), Image.Resampling.LANCZOS)
    rotate = random.randint(0, 360)
    image = image.rotate(rotate, expand=True)
    return image

def put_objects_in_background(background_folder,input_image_folder,output_folder,iterations,iteration_per_background,max_objects):


    if(not output_folder.endswith("/")):
        output_folder=output_folder+"/"
    if(not background_folder.endswith("/")):
        background_folder=background_folder+"/"
    if(not input_image_folder.endswith("/")):
        input_image_folder=input_image_folder+"/"
 
    background_filename_list = random.sample(glob.glob(background_folder+'*.jpg'),iterations)
    for background_filename in background_filename_list:

        for i in range(iteration_per_background):
            background = Image.open(background_filename)

            place_object_list =random.sample(glob.glob(input_image_folder+'*.png'),random.randrange(1,max_objects))

    

            paste_image_list =[]
            for place_object in place_object_list:
                paste_image_list.append(get_refactored_img(background,Image.open(place_object),2,3))


            alread_paste_point_list = []

            for img in paste_image_list:
                # if all not overlap, find the none-overlap start point
                while True:
                    # left-top point
                    # x, y = random.randint(0, background.size[0]), random.randint(0, background.size[1])

                    # if image need in the bg area, use this
                    x, y = random.randint(0, max(0, background.size[0]-img.size[0])), random.randint(0, max(0, background.size[1]-img.size[1]))

                    # right-bottom point
                    l2, r2 = (x, y), (x+img.size[0], y+img.size[1])

                    if all(not is_overlap(l1, r1, l2, r2) for l1, r1 in alread_paste_point_list):
                        # save alreay pasted points for checking overlap
                        alread_paste_point_list.append((l2, r2))
                        background.paste(img, (x, y), img)
                        break

            background.save(output_folder+"gen_"+str(i)+"_"+os.path.basename(background_filename).strip(".jpg")+".png")

            # check like this, all three rectangles all not overlapping each other
            from itertools import combinations
            assert(all(not is_overlap(l1, r1, l2, r2) for (l1, r1), (l2, r2) in combinations(alread_paste_point_list, 2)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Adding optional argument
    parser.add_argument('-b', '--background_folder',
                        help='Output image folder path', required=True)
    parser.add_argument('-i', '--input_image_folder',
                        help='Input image folder path', required=True)
    parser.add_argument('-o', '--output_folder',
                        help='Output folder path', required=True)
    parser.add_argument('-iter', '--iterations',
                        help='Number of random images', required=True,type=int)
    parser.add_argument('-iterpb', '--iteration_per_background',
                        help='Number of iterations of random images', required=True,type=int)
    parser.add_argument('-m', '--max_objects',
                        help='Number of max synthetic objects per images', required=True,type=int)
    args = parser.parse_args()
    put_objects_in_background(args.background_folder, args.input_image_folder,args.output_folder,args.iterations,args.iteration_per_background,args.max_objects)