import cv2
import numpy as np
import skimage.exposure
from PIL import Image
from PIL import Image, ImageOps
from os import listdir
import argparse




def extract_object_from_video(input_file, background_image,output_name):
    video = cv2.VideoCapture(input_file)
    image =Image.open(background_image)
    img_array = []
    while True:
        image_cp =image.copy()

        ret, frame = video.read()
        if frame is None:
            break
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        A = lab[:, :, 1]
        thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        blur = cv2.GaussianBlur(thresh, (0, 0), sigmaX=5,
                                sigmaY=5, borderType=cv2.BORDER_DEFAULT)
        mask = skimage.exposure.rescale_intensity(blur, in_range=(
            127.5, 255), out_range=(0, 255)).astype(np.uint8)
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        result[:, :, 3] = mask
        x, y, w, h = cv2.boundingRect(result[..., 3])
        cropped_img = result[y:y+h, x:x+w, :]

        new_img = ImageOps.expand(Image.fromarray(cropped_img), border=(20, 10, 20, 10), fill="green")
        w,h = new_img.size
        tot_w,tot_h = image_cp.size
        image_cp.paste(new_img,(int((tot_w-w)/2),int((tot_h-h)/2)),new_img)
        im_np = np.asarray(image_cp)
        #cv2.imshow("video", frame)
        #cv2.imshow("mask", im_np)
        img_array.append(im_np)
        height, width, layers = im_np.shape
        size = (width,height)
        if cv2.waitKey(25) == 27:
            break
    
    video.release()
    out = cv2.VideoWriter(output_name+'.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file',
                        help='Input mp4 file path', required=True)
    parser.add_argument('-bg', '--background_image',
                        help='Background image file path', required=True)
    parser.add_argument('-o', '--output_name',
                        help='Output file name', required=True)
    args = parser.parse_args()

    extract_object_from_video(args.input_file, args.background_image,args.output_name)
