#!/usr/bin/env python

import sys
import rospy
import message_filters
from detector import Detector
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from skeleton.msg import DetectionMsg, DetectionArrayMsg, KeypointsMsg
import cv2
#from utils import Utils
import numpy as np
import imageio
import os

class DetectorNode:

    def __init__(self):
        rospy.init_node('skeleton', anonymous=True)
        self.node_name = rospy.get_name()
        s_ = self.node_name+ '/obj_names'
        print(rospy.get_param(s_))
        
        '''
        yolo_cfg = rospy.get_param(f'/{self.node_name}/yolo_cfg')
        yolo_weights = rospy.get_param(f'/{self.node_name}/yolo_weights')
        obj_names = rospy.get_param(f'/{self.node_name}/obj_names')

        conf_threshold = float(rospy.get_param(f'/{self.node_name}/conf_threshold'))
        nms_threshold = float(rospy.get_param(f'/{self.node_name}/nms_threshold'))
        '''
        yolo_cfg = rospy.get_param(self.node_name + '/yolo_cfg')
        yolo_weights = rospy.get_param(self.node_name + '/yolo_weights')
        obj_names = rospy.get_param(self.node_name + '/obj_names')

        conf_threshold = float(rospy.get_param(self.node_name + '/conf_threshold'))
        nms_threshold = float(rospy.get_param(self.node_name + '/nms_threshold'))

        self.name_obj = rospy.get_param(self.node_name + '/name_obj')
        self.name_obj_pt = rospy.get_param(self.node_name + '/name_obj_pt')

        self.detector = Detector(yolo_cfg=yolo_cfg, \
                                 yolo_weights=yolo_weights,\
                                 obj_names=obj_names,\
                                 conf_threshold=conf_threshold,\
                                 nms_threshold=nms_threshold,\
                                 obj=self.name_obj,\
                                 obj_pt=self.name_obj_pt)
        
        self.__bridge = CvBridge()
        self.image_save_path = rospy.get_param(self.node_name + '/image_save_path')
        self.number_of_keypoints = int(rospy.get_param(self.node_name + '/number_of_keypoints'))

        #bbox_topic = rospy.get_param(self.node_name + '/bbox_topic')

        #SUBSCRIBERS
        trigger_camera_topic = rospy.get_param(self.node_name + '/trigger_camera_topic')
        image_source_topic = rospy.get_param(self.node_name + '/image_source_topic')
        depth_source_topic = rospy.get_param(self.node_name + '/depth_source_topic')

        #buffer size = 2**24 | 480*640*3
        #rospy.Subscriber(image_source_topic, Image, self.detect_image, queue_size=1, buff_size=2**24)   
        rgb_sub = message_filters.Subscriber(image_source_topic, Image, queue_size=10)
        depth_sub = message_filters.Subscriber(depth_source_topic, Image, queue_size=10)
        
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], 20, 0.1, allow_headerless=True) 
        ts.registerCallback(self.take_images)

        rospy.Subscriber(trigger_camera_topic, String, self.trigger_camera, queue_size=1, buff_size=2**24)   

        # PUBLISHERS
        detections_image_topic = rospy.get_param(self.node_name + '/detections_image_topic')
        detections_bbox_topic = rospy.get_param(self.node_name + '/detections_bbox_topic')
        keypoints_topic = rospy.get_param(self.node_name + '/keypoints_topic')
        self.__publisher = rospy.Publisher(detections_image_topic, Image, queue_size=1)
        #Questo nodo deve pubblicare le bbox sul topic per farle leggere all'altro nodo che creeremo
        #self.__publisherBB = rospy.Publisher(bbox_topic, DetectionMsg, queue_size=1)
        #Pubblichiamo anche l'array dell bboxes che dovra' leggere l'altro nodo
        self.__publisher_DETECTION_BBS = rospy.Publisher(detections_bbox_topic, DetectionArrayMsg, queue_size=1)
        self.__publisher_keypoints = rospy.Publisher(keypoints_topic, KeypointsMsg, queue_size=1)

        self.take_image = False


    def trigger_camera(self, data):
        try:
            if data.data=='take_image':
                self.take_image = True
        except CvBridgeError as e:
 	        print(e)
        

    def convert_depth(self, data):
        try:
            depth_image = self.__bridge.imgmsg_to_cv2(data, "passthrough")
        except CvBridgeError as e:
 	        print(e)

        #Convert the depth image to a Numpy array
        self.depth_array = np.array(depth_image, dtype=np.float32)


    def take_images(self, data, depth_data): #data is for the rgb image
        if self.take_image == True:
            try:
                image_i = self.__bridge.imgmsg_to_cv2(data, 'rgb8')
                self.convert_depth(depth_data)

                # SALVA LE IMMAGINI - TODO Vedere come salvare la depth 
                cv2.imwrite(self.image_save_path + '/rgb_01.png', image_i)
                img16 = self.depth_array.astype(np.uint16)
                imageio.imwrite(self.image_save_path + '/depth_01.tiff', img16)
                imageio.imwrite(self.image_save_path + '/depth_011.png', img16)

                self.take_image = False

                # DETECTION BB - TODO
                self.detect_image()

                # DETECTION KEYPOINTS - TODO
                self.detect_keypoints()

            except CvBridgeError as cve:
                rospy.logerr(str(cve))
                return

    #Il nodo controller si sottoscrive alle BB e ai keypoints, carica i keypoints e vede se ce ne sono tre. 
    # carica la depth e se tutto e' ok passa da  2D a 3D. e chiama la funzione di ragionamento
    # se non e' ok dice al robot di muoversi

    def detect_keypoints(self):
        # RUN DNN
        os.system('bash /home/monica/ros_catkin_ws_mine/src/skeleton/runDNN.sh')
        keypoints = np.loadtxt(self.image_save_path + 'keypoints.txt', comments="#", delimiter=" ", usecols=range(2))
        #esito = True
        #while(esito):
        #    try:
        #        keypoints = np.loadtxt(self.image_save_path + 'keypoints.txt', comments="#", delimiter=" ", usecols=range(2))
        #        esito = False
        #    except Exception as e:
        #        rospy.logerr(str(e))

        keys_msg = KeypointsMsg()
        for i in range(self.number_of_keypoints):
            for j in range(2):
                keys_msg.keypoints.append(int(keypoints[i,j]))
                #print(int(keypoints[i,j]))

        self.__publisher_keypoints.publish(keys_msg)
        #NEL CONTROLLER USA RESHAPE
        #arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        #newarr = arr.reshape(5, 2)

    def detect_image(self):
        # CARICA LE IMMAGINI - TODO
        image_o = cv2.imread(self.image_save_path + '/rgb_01.png')
        image = image_o.copy()

        outputs = self.detector.detect_objects(image)
        [idxs, boxes, classIDs] = self.detector.process_image(outputs, image)

        try:
            detection_message = self.__bridge.cv2_to_imgmsg(image, "bgr8")
            self.__publisher.publish(detection_message)

            detections_msg = DetectionArrayMsg()
            if len(idxs)>0: # the system finds any objects
                for i in idxs.flatten():
                    det_temp = DetectionMsg()
                    det_temp.type = "bb"
                    det_temp.classe = self.detector.labels[classIDs[i]] #sUtils.get_detection_class(classIDs[i])# str(classIDs[i])
                    det_temp.x = boxes[i][0]
                    det_temp.y = boxes[i][1]
                    det_temp.w = boxes[i][2]
                    det_temp.h = boxes[i][3]
                    detections_msg.detections.append(det_temp)
            else:
                print("NO OBJECT FOUND")

            self.__publisher_DETECTION_BBS.publish(detections_msg)

        except CvBridgeError as cve:
            rospy.logerr(str(cve))
            return

def main(args):

    detector_node = DetectorNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)
