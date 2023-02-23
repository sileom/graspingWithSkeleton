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

from skeleton.msg import BoxesMsg, DetectionArrayMsg, KeypointsMsg, Command, CommandResult

import cv2 as cv
import numpy as np
import imageio
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

        #rospy.Subscriber(detections_bbox_topic, DetectionArrayMsg, self.get_detection_BB, queue_size=1, buff_size=2**24)
        #rospy.Subscriber(keypoints_topic, KeypointsMsg, self.get_keypoints, queue_size=1, buff_size=2**24)
        bboxes_sub = message_filters.Subscriber(detections_bbox_topic, DetectionArrayMsg, queue_size=1)
        keypoints_sub = message_filters.Subscriber(keypoints_topic, KeypointsMsg, queue_size=1)

        ts = message_filters.ApproximateTimeSynchronizer([bboxes_sub, keypoints_sub], 1, 4, allow_headerless=True) 
        ts.registerCallback(self.get_BB_and_keypoints)
        # Nel topic seguente gestiamo gli errori di movimento del robot e errori di grasping (prevedi 2 o 3 stringhe diverse)
        rospy.Subscriber('/robot_command_done', CommandResult, self.robot_done_callback, queue_size=1, buff_size=2**24) 
        rospy.Subscriber('/robot_command_error', String, self.robot_error_callback, queue_size=1, buff_size=2**24) 

        rospy.Subscriber(trigger_camera_topic, String, self.trigger_camera, queue_size=1, buff_size=2**24)

        #rostopic pub /controller/robot_command_done std_msgs/String "ok"

        # PUBLISHERS
        self.__publisherTriggerCamera = rospy.Publisher(trigger_camera_topic, String, queue_size=1)
        self.__publisherCommandToRobot = rospy.Publisher('/robot_command', Command, queue_size=1)

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
        self.keypoints3D = []
        self.currentCommand = ""

        self.Ae_curr = np.array([[-0.470616, -0.865341, -0.172351, -0.287212],
                                [-0.854349, 0.495718, -0.156046, -0.416609],
                                [0.220471, 0.0738101, -0.972597, 0.00588167],
                                [0, 0, 0, 1]])


        self.Ace = np.array([[-0.005812655098, -0.9991180429, 0.04158544615, 0.06709041116],
                            [0.9999425447, -0.006181950697, -0.008757327031, -0.03431457374],
                            [0.009006682622, 0.04153215353, 0.9990965719, -0.06649198096],
                            [0.,  0.,  0.,  1.]])     # matrice camera-end-effector

                             

        # TODO
        self.T_home = np.array([[-0.1382, 0.9903, 0.0132,-0.0949],
                            [0.9904, 0.1382, 0.0043, 0.3899],
                            [0.0024, 0.0137, -0.9999,  0.3916],
                            [0., 0., 0., 1.0000]])
        
        self.T_release = np.array([[-0.1382, 0.9903, 0.0132,-0.0949],
                            [0.9904, 0.1382, 0.0043, 0.3899],
                            [0.0024, 0.0137, -0.9999,  0.3016],
                            [0., 0., 0., 1.0000]])

        self.K = np.array([[605.81918473810867, 0.0, 324.66592260274075], 
                           [0.0, 603.77141649252928, 236.93792129936679], 
                           [0.0, 0.0, 1.0]])

    def trigger_camera(self, data):
        try:
            if data.data=='start':
                self.goToHome('homeStart')
        except CvBridgeError as e:
 	        print(e)

    def get_BB_and_keypoints(self, detectedBB, keypoints):
        # NB - IF THERE ARE MULTIPLE OBJECTS WE TAKE THE FIRST
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

        num = 2*self.number_of_keypoints
        keypointsFirstObj = keypoints_list[:num]
        key2D = keypoints_list.reshape(self.number_of_keypoints, 2)  
        #key2D = key2D[:5,:]

        print('*********')
        print(self.detectedBBs_list)
        print(key2D)

        esito = self.checkKeypoints(key2D)

        if esito==True: #i punti ci sono
            print("esito TRUE")
            # CARICA IMMAGINE DEPTH
            depth_image = imageio.imread(self.image_save_path + '/depth_01.tiff')
            print(type(depth_image))
            depth_array = np.array(depth_image)
            #plt.imshow(depth_array)
            #plt.show()
            
            self.keypoints3D = np.zeros((self.number_of_keypoints,3))
            # CONVERTI 2D IN 3D
            for j in range(len(key2D)):
                if np.all(key2D[j,:] > 0):
                    keyCam_ = GeomUtility.deproject_pixel_to_point(key2D[j,:], depth_array, self.K)
                    keyCam = np.array([keyCam_[0], keyCam_[1], keyCam_[2], 1])
                    #keyCam = np.array([keyCam_[0], keyCam_[1], keyCam_[2], 1])

                    print("Punto " + str(j))
                    print("terna camera 3D")
                    print(keyCam.T)
                    print()
                    print("terna end-effector")
                    print(self.Ace.dot(keyCam.T))
                    print()
                    print("terna mondo")
                    print(self.Ae_curr.dot(self.Ace.dot(keyCam.T)))
                    print()
                    #print('-_-_-_-_-_')
                    # TRAFORMA I PUNTI IN COORD ROBOT
                    key3D = self.Ae_curr.dot(self.Ace.dot(keyCam.T))
                    #print('-_********** 3d')
                    #print(np.array(key3D))
                    self.keypoints3D[j,:] = key3D[:3]
                else:
                    self.keypoints3D[j,:] = np.array([-10, -10, -10])
            
            print('Punti in terna base')
            print(self.keypoints3D)
            print()
            
            # CALCOLA PIANO
            #normal_vec = GeomUtility.getNormalVec(self.keypoints3D[0,:], self.keypoints3D[1,:], self.keypoints3D[2,:]) #planeWith3Points(P):
            #keyPin = self.keypoints[5,:]
            
            # CALCOLA PUNTO DI PRESA
            T_a = GeomUtility.computeGraspingPoint(self.detectedBBs_list[0], self.keypoints3D, self.Ae_curr[:3,:3])
            self.R = T_a[:3,:3]
            self.p = T_a[:3, 3]            
            #self.R e self.p ci arrivano dal ragionatore
            #self.R = np.array([[-1,  0,  0],
            #                    [0,  1,  0],
            #                    [0,  0, -1]])
            #self.p = np.array([0.2, 0.3, 0.2])
            print()
            print('Grasping point')
            print(self.R)
            print(self.p)

            self.quat = GeomUtility.r2quat(self.R)
            print('{position: {x: ' + str(self.p[0]) + ', y: ' + str(self.p[1]) + ', z: ' + str(self.p[2]) + '}, orientation: {x: ' + str(self.quat[1]) + ', y: ' + str(self.quat[2]) + ', z: ' + str(self.quat[3]) + ', w: ' + str(self.quat[0]) + '}}')
            p_approach = self.p + self.R.dot([0,0,-0.10])
            print()
            print('Approccio')
            print(p_approach)
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
            print("HO MANDATO L'APPROCCIO AL ROBOT. Attendo la sua risposta")

        else:
            print("esito FALSE")
            self.goToOtherPos()
            print("HO MANDATO IN OTHER IL ROBOT")

    def checkKeypoints(self, key2D): #funzione per vedere se ci sono i 3 keypoints di interesse
        i = 1 #indice che indica il primo dei tre punti 
        e1 = key2D[i, 0] != -1 and key2D[i, 1] != -1
        e2 = key2D[i+1, 0] != -1 and key2D[i+1, 1] != -1
        e3 = key2D[i+2, 0] != -1 and key2D[i+2, 1] != -1
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
            self.__publisherTriggerCamera.publish('take_image')
        elif self.currentCommand == 'homeStart': ########################
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
        print("HO MANDATO IL RELEASE AL ROBOT. Attendo la sua risposta")

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
        print("HO MANDATO APRI AL ROBOT. Attendo la sua risposta")

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
        print("HO MANDATO CHIUDI AL ROBOT. Attendo la sua risposta")

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
        print("HO MANDATO VAI ALL'OGGETTO AL ROBOT. Attendo la sua risposta")

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
        print("HO MANDATO IL ROBOT IN HOME")

    def goToOtherPos(self):
        # INDIVIDUARE CENTRO bb --> p in terna base (mettere la z a zero)

        # SCELTA ASSE PARALLELO A Z BASE PASSANTE PER CENTRO BB
        
        self.currentCommand = 'error'
        cmd_to_robot = Command()
        cmd_to_robot.command = '' 
        cmd_to_robot.typePlan = 'eeOther'
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
            print("HO RICEVUTO ERRORE SU GRASPING")
            self.goToHome('error')
        elif data.data == 'errorRob':
            print("HO RICEVUTO ERRORE SU ROBOT")
            self.goToOtherPos()


    
def main(args):

    controller = Controller()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.logerr('Shutting down')

if __name__ == '__main__':
    main(sys.argv)


