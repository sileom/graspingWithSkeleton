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

from skeleton.msg import BoxesMsg, DetectionArrayMsg, KeypointsMsg, Command, CommandResult, DetectionMsg

import cv2 as cv
import numpy as np
import imageio
import timeit
import time
import matplotlib.pyplot as plt

class Controller:

    def __init__(self):
        rospy.init_node('skeleton', anonymous=True)
        self.node_name = rospy.get_name()

        #self.__bridge = CvBridge()
        self.image_save_path = rospy.get_param(self.node_name + '/image_save_path')
        self.number_of_keypoints = int(rospy.get_param(self.node_name + '/number_of_keypoints'))
        self.grasp_width = float(rospy.get_param(self.node_name + '/grasp_width'))
        trigger_camera_topic = rospy.get_param(self.node_name + '/trigger_camera_topic')

        # SUBSCRIBERS
        detections_bbox_topic = rospy.get_param(self.node_name + '/detections_bbox_topic') # dove sono pubblica le BB (senza le parti piccole TODO)
        keypoints_topic = rospy.get_param(self.node_name + '/keypoints_topic')

        bboxes_sub = message_filters.Subscriber(detections_bbox_topic, DetectionArrayMsg, queue_size=1)
        keypoints_sub = message_filters.Subscriber(keypoints_topic, KeypointsMsg, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([bboxes_sub, keypoints_sub], 1, 4, allow_headerless=True) 
        ts.registerCallback(self.get_BB_and_keypoints)

        rospy.Subscriber('/robot_command_done', CommandResult, self.robot_done_callback, queue_size=1, buff_size=2**24) 
        rospy.Subscriber('/robot_command_error', String, self.robot_error_callback, queue_size=1, buff_size=2**24) 

        rospy.Subscriber(trigger_camera_topic, String, self.trigger_camera, queue_size=1, buff_size=2**24)

        # PUBLISHERS
        self.__publisherTriggerCamera = rospy.Publisher(trigger_camera_topic, String, queue_size=1)
        self.__publisherCommandToRobot = rospy.Publisher('/robot_command', Command, queue_size=1)


        self.detectedBBs_list = []
        self.currentCommand = ""
        self.depth_array = []

        self.Ae_curr = np.array([[-0.106389, 0.994181, 0.0162987, -0.163438],
                            [0.994245, 0.106173, 0.0135973, 0.240477],
                            [0.0117877, 0.0176515, -0.999775, 0.201297],
                            [0, 0, 0, 1]])

        self.Ace = np.array([[0.02219194357, -0.9997472712, 0.003593246995, 0.06013792774],
                            [0.9997499728,  0.02220156848,  0.00266124757,  -0.03995585374],
                            [-0.002740350715,  0.00353329033,  0.9999900031,  -0.0641041473],
                            [0, 0, 0, 1]])

        self.T_start = np.array([[0.099664, 0.99501, 0.001314, -0.10378],
                            [0.995011, 0.099661, 0.000322758, 0.370664], 
                            [0.000190106, 0.00134041, -0.999999, 0.277478], 
                            [0., 0., 0., 1.0000]])

        self.T_home = np.array([[-0.3256,   0.945416, 0.0123956, -0.00809442],
                                 [0.945362,  0.325303, 0.0212066, 0.480168],
                                 [0.0160167, 0.0186232, -0.999698, 0.25131],
                                 [0, 0, 0, 1]])

        self.T_release = np.array([[0.099664, 0.99501, 0.001314, -0.10378],
                            [0.995011, 0.099661, 0.000322758, 0.370664], #38
                            [0.000190106, 0.00134041, -0.999999, 0.327478], #15
                            [0., 0., 0., 1.0000]])

        self.K = np.array([[613.29049956446272, 0.0, 323.86558088925034], 
                           [0.0, 612.53173150788348, 238.85284745225653], 
                           [0.0, 0.0, 1.0]])

        self.angleForLook = np.pi/2 # questo angolo indica la successiva posa di vista

    def trigger_camera(self, data):
        try:
            if data.data=='start':
                self.T_home = self.T_start
                self.goToHome('homeStart')
        except CvBridgeError as e:
 	        print(e)

    def get_BB_and_keypoints(self, detectedBB, keypoints):
        # NB - IF THERE ARE MULTIPLE OBJECTS WE TAKE THE FIRST
        keypoints_list = []
        for j in range(len(keypoints.keypoints)):
            keypoints_list.append(keypoints.keypoints[j])
        keypoints_list = np.array(keypoints_list)

        num = 2*self.number_of_keypoints
        keypointsFirstObj = keypoints_list[:num]
        key2D = keypoints_list.reshape(self.number_of_keypoints, 2)  
        
        bbFirstObject = []
        for i in range(len(detectedBB.detections)):
            self.detectedBBs_list.append(detectedBB.detections[i])
            
            if GeomUtility.isIn(self.detectedBBs_list[i], key2D[0,:]) == True:
                bbFirstObject = detectedBB.detections[i]

        bbFirstObject = detectedBB.detections[0] 

        esito = self.checkKeypoints(key2D)

        # Load IMG DEPTH
        start = timeit.default_timer()
        depth_image = imageio.imread(self.image_save_path + '/depth_01.tiff')
        self.depth_array = np.array(depth_image)

        if esito==True: 
            self.key3D_cam = np.zeros((self.number_of_keypoints,3))
            # 2D TO 3D
            for j in range(len(key2D)):
                if np.all(key2D[j,:] > 0):
                    keyCam_ = GeomUtility.deproject_pixel_to_point(key2D[j,:], self.depth_array, self.K)
                    keyCam = np.array([keyCam_[0], keyCam_[1], keyCam_[2], 1])

                    # Robot FRAMe
                    self.key3D_cam[j,:] = keyCam.T[:3]
                else:
            
            
            Tg_cam = GeomUtility.computeGraspingPointWithTriple(bbFirstObject, self.key3D_cam, self.Ae_curr[:3,:3])
            T_a = self.Ae_curr.dot(self.Ace.dot(Tg_cam))

            stop = timeit.default_timer()
            print('TIME COMPUTATION GRASPING_POINT: ', stop - start)  
            
            self.R = T_a[:3,:3]
            self.p = T_a[:3, 3]            

            self.quat = GeomUtility.r2quat(self.R)
            p_approach = self.p + self.R.dot([0,0,-0.10])
            self.currentCommand = 'approachToObject'
            cmd_to_robot = Command()
            cmd_to_robot.command = 'approachToObject' 
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
            self.goToOtherPos()

    def checkKeypoints(self, key2D): 
        count = 0
        for i in range(len(key2D)):
            if key2D[i, 0] != -1 and key2D[i, 1] != -1:
                count = count + 1
        
        return (count >= 3)


    def robot_done_callback(self, data):
        if self.currentCommand == 'approachToObject':
            self.goToObject()
        elif self.currentCommand == 'goToObject':
            self.grasp()
        elif self.currentCommand == 'grasp':
            self.goToReleaseApproach()
        elif self.currentCommand == 'goToReleaseApproach':
            self.goToRelease()
        elif self.currentCommand == 'goToRelease':
            self.open_gripper()
        elif self.currentCommand == 'release':
            self.goToHome('home')
        elif self.currentCommand == 'error':
            self.__publisherTriggerCamera.publish('take_image')
        elif self.currentCommand == 'homeStart': 
            quat = np.array([data.rPose.orientation.w, data.rPose.orientation.x, data.rPose.orientation.y, data.rPose.orientation.z])
            T = GeomUtility.quat2r(quat)
            self.Ae_curr[:3,:3] = np.array(T)
            self.Ae_curr[0,3] = data.rPose.position.x
            self.Ae_curr[1,3] = data.rPose.position.y
            self.Ae_curr[2,3] = data.rPose.position.z
            print('Ae')
            print(self.Ae_curr)
            self.__publisherTriggerCamera.publish('take_image')
        elif self.currentCommand == 'home':
            rospy.loginfo('END.')

    def goToReleaseApproach(self):
        self.currentCommand = 'goToReleaseApproach'
        cmd_to_robot = Command()
        cmd_to_robot.command = 'goToReleaseApproach' 
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        p_approach = self.p + self.R.dot([0,0,-0.20])
        cmd_to_robot.goal.position.x = p_approach[0]
        cmd_to_robot.goal.position.y = p_approach[1]
        cmd_to_robot.goal.position.z = p_approach[2]
        quat = GeomUtility.r2quat(self.T_release[:3,:3])
        cmd_to_robot.goal.orientation.x = quat[1]
        cmd_to_robot.goal.orientation.y = quat[2]
        cmd_to_robot.goal.orientation.z = quat[3]
        cmd_to_robot.goal.orientation.w = quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)


    def goToRelease(self):
        self.currentCommand = 'goToRelease'
        cmd_to_robot = Command()
        cmd_to_robot.command = 'goToRelease' 
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = self.T_release[0,3]
        cmd_to_robot.goal.position.y = self.T_release[1,3]
        cmd_to_robot.goal.position.z = self.T_release[2,3]
        quat = GeomUtility.r2quat(self.T_release[:3,:3])
        cmd_to_robot.goal.orientation.x = quat[1]
        cmd_to_robot.goal.orientation.y = quat[2]
        cmd_to_robot.goal.orientation.z = quat[3]
        cmd_to_robot.goal.orientation.w = quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def open_gripper(self):
        self.currentCommand = 'release'
        cmd_to_robot = Command()
        cmd_to_robot.command = 'release' 
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
        cmd_to_robot.command = 'grasp' 
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
        cmd_to_robot.command = 'goToObject' 
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
        cmd_to_robot.command = currentCmd
        cmd_to_robot.typePlan = 'ee'
        cmd_to_robot.goalJM = ''
        cmd_to_robot.goal.position.x = self.T_home[0,3]
        cmd_to_robot.goal.position.y = self.T_home[1,3]
        cmd_to_robot.goal.position.z = self.T_home[2,3]
        quat = GeomUtility.r2quat(self.T_home[:3,:3])
        cmd_to_robot.goal.orientation.x = quat[1]
        cmd_to_robot.goal.orientation.y = quat[2]
        cmd_to_robot.goal.orientation.z = quat[3]
        cmd_to_robot.goal.orientation.w = quat[0]
        cmd_to_robot.grasp_width = 0.0
        self.__publisherCommandToRobot.publish(cmd_to_robot)

    def goToOtherPos(self):
        if self.angleForLook == np.pi*2:
            self.goToHome('home')
            return
        poI = GeomUtility.getCenterBB(self.detectedBBs_list[0]) # p oggetto in terna immagine
        poI_cam = GeomUtility.deproject_pixel_to_point(poI, self.depth_array, self.K)
        poI_cam_ = np.array([poI_cam[0], poI_cam[1], poI_cam[2], 1])
        poO = self.Ae_curr.dot(self.Ace.dot(poI_cam_.T))
        po = poO[:3]
        newT = GeomUtility.computeNewLookPose(po, self.Ae_curr, self.angleForLook)
        self.angleForLook = self.angleForLook + (np.pi/2)
        if self.angleForLook == (2*np.pi):
            self.angleForLook = 0.
        self.T_home = newT
        self.goToHome('homeStart')

    def robot_error_callback(self, data):
        if data.data == 'errorGrasp':
            print("ERROR ON GRASPING")
            self.goToHome('error')
        elif data.data == 'errorRob':
            print("ERROR ON ROBOT")
            self.goToOtherPos()

def main(args):
    controller = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)
