#!/usr/bin/env python

import sys
import rospy
import os
import message_filters
from detector import Detector
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from geometry_utility import GeomUtility
#from cv_bridge import CvBridge, CvBridgeError

from skeleton.msg import BoxesMsg, DetectionArrayMsg, KeypointsMsg, Command

import cv2 as cv
import numpy as np

class Controller:

    def __init__(self):
        rospy.init_node('skeleton', anonymous=True)
        self.node_name = rospy.get_name()

        #self.__bridge = CvBridge()
        self.image_save_path = rospy.get_param(self.node_name + '/image_save_path')
        self.number_of_keypoints = int(rospy.get_param(self.node_name + '/number_of_keypoints'))
        self.grasp_width = float(rospy.get_param(self.node_name + '/grasp_width'))

        # SUBSCRIBERS
        detections_bbox_topic = rospy.get_param(self.node_name + '/detections_bbox_topic') # dove sono pubblica le BB (senza le parti piccole TODO)
        keypoints_topic = rospy.get_param(self.node_name + '/keypoints_topic')

        #rospy.Subscriber(detections_bbox_topic, DetectionArrayMsg, self.get_detection_BB, queue_size=1, buff_size=2**24)
        #rospy.Subscriber(keypoints_topic, KeypointsMsg, self.get_keypoints, queue_size=1, buff_size=2**24)
        bboxes_sub = message_filters.Subscriber(detections_bbox_topic, DetectionArrayMsg, queue_size=1)
        keypoints_sub = message_filters.Subscriber(keypoints_topic, KeypointsMsg, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([bboxes_sub, keypoints_sub], 20, 4, allow_headerless=True) 
        ts.registerCallback(self.get_BB_and_keypoints)
        # Nel topic seguente gestiamo gli errori di movimento del robot e errori di grasping (prevedi 2 o 3 stringhe diverse)
        rospy.Subscriber(self.node_name + '/robot_command_done', String, self.robot_done_callback, queue_size=1, buff_size=2**24) 
        rospy.Subscriber(self.node_name + '/robot_command_error', String, self.robot_error_callback, queue_size=1, buff_size=2**24) 

        # PUBLISHERS
        trigger_camera_topic = rospy.get_param(self.node_name + '/trigger_camera_topic')
        self.__publisherTriggerCamera = rospy.Publisher(trigger_camera_topic, String, queue_size=1)
        self.__publisherCommandToRobot = rospy.Publisher(self.node_name + '/robot_command', Command, queue_size=1)

        # si iscrive a DETECTED BB

        # si iscrive a keypoints (sincronizzato??)

        #pubblico comandi per il robot

        # controllo se ho i 3 punti, se non li ho dico al robot di muoversi 

        # quando il robot ha finito dico di prendere un'altr immagine --> pubblico su rtrigger camera

        # se ho i 3 punti: uso la BB per capire se è girato o no
        # mi calcolo il piano 
        # se è capovolto: calcolo altezza, normale e mando al robot
        # se non è capovolto : uso il 4 keypoint per andare e il piano per la normale e mando al robot

        self.detectedBBs_list = []
        self.keypoints = []
        self.currentCommand = ""

        self.p_home = np.array([0.2, 0.3, 0.2])
        self.T_home = np.array([[-1, 0, 0,  0],
                                [0,  1,  0,  0],
                                [0,  0, -1, 0],
                                [0., 0.,  0., 1.]])
        
        self.p_release = np.array([0.2, 0.3, 0.2])
        self.T_release = np.array([[-1, 0, 0,  0],
                                [0,  1,  0,  0],
                                [0,  0, -1, 0],
                                [0., 0.,  0., 1.]])

    def get_BB_and_keypoints(self, detectedBB, keypoints):
        print('------------------------')
        print(detectedBB)
        print('------------------------')
        print(keypoints)
        for i in range(len(detectedBB.detections)):
            self.detectedBBs_list.append(detectedBB.detections[i])
        keypoints_list = []
        for j in range(len(keypoints.keypoints)):
            keypoints_list.append(keypoints.keypoints[j])
        keypoints_list = np.array(keypoints_list)

        self.keypoints = keypoints_list.reshape(self.number_of_keypoints, 2)

        print('*********')
        print(self.detectedBBs_list)
        print(self.keypoints)

        esito = self.checkKeypoints()

        if esito==True: #i punti ci sono
            print("esito TRUE")
            #CALCOLA PIANO E PUNTO DI PRESA CONVERTI IN 3D  ECC !++/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/+/!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
            #self.T e self.p ci arrivano dal ragionatore
            self.T = np.array([[-1,  0,  0],
                                [0,  1,  0],
                                [0,  0, -1]])
            self.p = np.array([0.2, 0.3, 0.2])

            self.quat = GeomUtility.r2quat(self.T)
            p_approach = self.p + self.T.dot([0,0,-0.10])
            print(p_approach)
            self.currentCommand = 'approachToObject'
            cmd_to_robot = Command()
            cmd_to_robot.command = '' 
            cmd_to_robot.typePlan = 'ee'
            cmd_to_robot.goalJM = ''
            cmd_to_robot.goal.position.x = p_approach[0]
            cmd_to_robot.goal.position.y = p_approach[1]
            cmd_to_robot.goal.position.z = p_approach[2]
            cmd_to_robot.goal.orientation.x = self.quat[1]
            cmd_to_robot.goal.orientation.y = self.quat[2]
            cmd_to_robot.goal.orientation.z = self.quat[3]
            cmd_to_robot.goal.orientation.w = self.quat[0]
            cmd_to_robot.grasp_width = 0.0
            self.__publisherCommandToRobot.publish(cmd_to_robot)

        else:
            print("esito FALSE")
            self.goToOtherPos()

    def checkKeypoints(self): #funzione per vedere se ci sono i 3 keypoints di interesse
        i = 0 #indice che indica il primo dei tre punti 
        e1 = self.keypoints[i, 0] != -1 and self.keypoints[i, 1] != -1
        e2 = self.keypoints[i+1, 0] != -1 and self.keypoints[i+1, 1] != -1
        e3 = self.keypoints[i+2, 0] != -1 and self.keypoints[i+2, 1] != -1
        return (e1 and e2 and e3)


    def robot_done_callback(self, data):
        if self.currentCommand == 'approachToObject':
            self.goToObject()
        elif self.currentCommand == 'goToObject':
            self.grasp()
        elif self.currentCommand == 'grasp':
            self.goToRelease()
        elif self.currentCommand == 'goToRelease':
            self.open_gripper()
        elif self.currentCommand == 'release':
            self.goToHome('home')
        elif self.currentCommand == 'error':
            self.__publisherTriggerCamera.publish('i')
        elif self.currentCommand == 'home':
            rospy.loginfo('END.')

    def goToRelease(self):
        self.currentCommand = 'goToRelease'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = self.p_release[0]
        cmd_to_robot.goal.position.y = self.p_release[1]
        cmd_to_robot.goal.position.z = self.p_release[2]
        quat = GeomUtility.r2quat(self.T_release)
        cmd_to_robot.goal.orientation.x = quat[1]
        cmd_to_robot.goal.orientation.y = quat[2]
        cmd_to_robot.goal.orientation.z = quat[3]
        cmd_to_robot.goal.orientation.w = quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def open_gripper(self):
        self.currentCommand = 'release'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'open'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = 0
        cmd_to_robot.goal.position.y = 0
        cmd_to_robot.goal.position.z = 0
        cmd_to_robot.goal.orientation.x = 0
        cmd_to_robot.goal.orientation.y = 0
        cmd_to_robot.goal.orientation.z = 0
        cmd_to_robot.goal.orientation.w = 1
        cmd_to_robot.grasp_width = self.grasp_width
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def grasp(self):
        self.currentCommand = 'grasp'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'close'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = 0
        cmd_to_robot.goal.position.y = 0
        cmd_to_robot.goal.position.z = 0
        cmd_to_robot.goal.orientation.x = 0
        cmd_to_robot.goal.orientation.y = 0
        cmd_to_robot.goal.orientation.z = 0
        cmd_to_robot.goal.orientation.w = 1
        cmd_to_robot.grasp_width = self.grasp_width
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def goToObject(self):
        self.currentCommand = 'goToObject'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = self.p[0]
        cmd_to_robot.goal.position.y = self.p[1]
        cmd_to_robot.goal.position.z = self.p[2]
        cmd_to_robot.goal.orientation.x = self.quat[1]
        cmd_to_robot.goal.orientation.y = self.quat[2]
        cmd_to_robot.goal.orientation.z = self.quat[3]
        cmd_to_robot.goal.orientation.w = self.quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def goToHome(self, currentCmd):
        self.currentCommand = currentCmd
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = self.p_home[0]
        cmd_to_robot.goal.position.y = self.p_home[1]
        cmd_to_robot.goal.position.z = self.p_home[2]
        quat = GeomUtility.r2quat(self.T_home)
        cmd_to_robot.goal.orientation.x = quat[1]
        cmd_to_robot.goal.orientation.y = quat[2]
        cmd_to_robot.goal.orientation.z = quat[3]
        cmd_to_robot.goal.orientation.w = quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def goToOtherPos(self):
        self.currentCommand = 'error'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'ee_other'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = 0
        cmd_to_robot.goal.position.y = 0
        cmd_to_robot.goal.position.z = 0
        cmd_to_robot.goal.orientation.x = 0
        cmd_to_robot.goal.orientation.y = 0
        cmd_to_robot.goal.orientation.z = 0
        cmd_to_robot.goal.orientation.w = 1
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def robot_error_callback(self, data):
        if data.data == 'errorGrasp':
            self.goToHome('error')
        elif self.currentCommand == 'errorRob':
            self.goToOtherPos()


    
def main(args):

    controller = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)


