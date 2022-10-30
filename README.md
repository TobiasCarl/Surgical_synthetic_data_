# Surgical_synthetic_data  


Simple overview of use

## Description 2D synthetic data  

Extracting interesting objects from image, replacing background with transparent background and cropping area around the object. The extracted objects are then placed randomly in the background.  

## Getting Started

### Executing example program
To extract scissors from greenscreen  
python retrieve_objects_from_images.py -i demo_scissors -o demo_transparent_scissors  

To place the extracted scissors from greenscreen randomly on background  
python put_objects_in_background.py -b demo_backgrounds -i demo_transparent_scissors -o demo_results -iter 5 -m 5 -iterpb 5  
 
Take a look in demo_results!    

To extract object from video and add a new background, run the following command below.  
python put_new_background_video.py -i demo_video_background/demo_greenscreen_vid.mp4 -bg demo_video_background/demo_background.jpg -o demo_video_background/demo_output_with_background.mp4  

Take a look at the resulting video in demo_video_background

