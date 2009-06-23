import roslib
roslib.update_path('opencv_util')
from image_msgs.msg import Image as RosImage
from std_msgs.msg import UInt8MultiArray
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import MultiArrayDimension

import opencv.adaptors as ad
import Image as pil
import numpy as np

##
# Convert from ROS to OpenCV image datatype
# @param image to convert
def ros2cv(image):
    #ros to pil then pil to opencv
    if image.encoding != 'bgr' or image.encoding != 'rgb':
        raise RuntimeError('Unsupported format "%s"' % image.encoding)
    if image.depth != 'uint8':
        raise RuntimeError('Unsupported depth "%s"' % image.depth)

    height    = image.uint8_data.layout.dim[0].size
    width     = image.uint8_data.layout.dim[1].size
    channels  = image.uint8_data.layout.dim[2].size
    np_image  = np.reshape(np.fromstring(image.uint8_data.data, dtype='uint8', count=height*width*channels), [height, width, channels])
    return ad.NumPy2Ipl(np_image)

##
# Simple drawing of text on an image with a drop shadow
def text(image, x, y, a_string):
    font = cvInitFont(CV_FONT_HERSHEY_SIMPLEX, .3, .3)
    cvPutText(image, a_string, cvPoint(x, y),     font, cvScalar(0,0,0))
    cvPutText(image, a_string, cvPoint(x+1, y+1), font, cvScalar(255,255,255))

##
# Tries to clean up segmentation 
# @param cv_image
# @param n_times to run erode & dilate
# @return image in original type with hopefully cleaner segmentations
def clean_segmentation(cv_image, n_times=1):
    dst  = cv.cvCloneImage(cv_image)
    dst2 = cv.cvCloneImage(cv_image)
    cv.cvErode(cv_image, dst, None, n_times)
    cv_image = cv.cvDilate(dst, dst2, None, n_times)
    return dst2

##
# Mask a color image with a given black and white mask
# @param img
# @param img_mask one channeled image
# @return color image with masked part filled with black
def mask(img, img_mask):
    dim      = img.width, img.height
    depth    = img.depth
    channels = img.nChannels

    r_chan = cvCreateImage(cvSize(*dim), depth, 1)
    g_chan = cvCreateImage(cvSize(*dim), depth, 1)
    b_chan = cvCreateImage(cvSize(*dim), depth, 1)
    combined = cvCreateImage(cvSize(*dim), depth, 3)
    cvSplit(img, r_chan, g_chan, b_chan, None)

    cvAnd(r_chan, img_mask, r_chan)
    cvAnd(g_chan, img_mask, g_chan)
    cvAnd(b_chan, img_mask, b_chan)
    cvMerge(r_chan, g_chan, b_chan, None, combined)
    return combined

##
# Split a color image into its component parts
def split(image):
    img1 = cvCloneImage(image)
    img2 = cvCloneImage(image)
    img3 = cvCloneImage(image)
    cvSplit(image, img1, img2, img3, None)
    return (img1, img2, img3)
###
## Fill holes in a binary image using scipy
##
#def fill_holes(cv_img):
#    img_np     = ut.cv2np(cv_img)
#    img_binary = img_np[:,:,0]
#    results    = ni.binary_fill_holes(img_binary)
#    img_cv     = ut.np2cv(np_mask2np_image(results))
#    return img_cv

##
# Easy scaling of an image by factor s
# @param image
# @param s
def scale(image, s):
    scaled = cv.cvCreateImage(cv.cvSize(int(image.width * s), int(image.height * s)), image.depth, image.nChannels)
    cv.cvResize(image, scaled, cv.CV_INTER_AREA)
    return scaled

if __name__ == '__main__':
    import opencv.highgui as hg
    import rospy
    from photo.srv import *
    #a = create_ros_image()
    #print a

    rospy.wait_for_service('/photo/capture')
    say_cheese = rospy.ServiceProxy('/photo/capture', Capture)
    ros_img = say_cheese().image
    print dir(ros_img)
    cv_img = ros2cv(ros_img)
    hg.cvSaveImage('test.png', cv_img)

#def create_ros_image(width=1, height=1, channels=2, data='12'):
#    d1 = MultiArrayDimension(label='height',   size=height,   stride=width*height*channels)
#    d2 = MultiArrayDimension(label='width',    size=width,    stride=width*channels)
#    d3 = MultiArrayDimension(label='channels', size=channels, stride=channels)
#
#    layout     = MultiArrayLayout(dim = [d1,d2,d3])
#    multiarray = UInt8MultiArray(layout=layout, data=data)
#    return RosImage(label='image', encoding='bgr', depth='uint8', uint8_data=multiarray)
