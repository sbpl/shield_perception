#! /usr/bin/python
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import tf
import pickle

# Instantiate CvBridge
bridge = CvBridge()

frame_counter = 0
listener = None
list_tf_1 = []
list_tf_2 = []

def image_callback(msg):
    global frame_counter, listener, list_tf_1, list_tf_2
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("Image", cv2_img)
        k = cv2.waitKey(33)
        if k==32:
            print("Recording TF")
            try:
                (trans1,rot1) = listener.lookupTransform('base_footprint','ATag',rospy.Time(0))
                print("translation 1:", trans1)
                print("rotation 1:", rot1)
                found_1 = True
            except:
                print("tf 1 not found")
                found_1 = False
            try:
                (trans2,rot2) = listener.lookupTransform('sc_color_frame','ATag_sc',rospy.Time(0))
                print("translation 2:", trans2)
                print("rotation 2:", rot2)
                found_2 = True
            except:
                print("tf 2 not found")
                found_2 = False
            if found_1 and found_2:
                list_tf_1.append((trans1,rot1))
                list_tf_2.append((trans2,rot2))
        elif k==27:
            f = open("tf_data.pkl",'w')
            pickle.dump([list_tf_1,list_tf_2],f)
            f.close()    
                        
            
    except CvBridgeError, e:
        print(e)

def main():
    global listener
    rospy.init_node('image_listener')
    listener = tf.TransformListener()
    # Define your image topic
    image_topic = "/camera/rgb/image_rect_mono"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()