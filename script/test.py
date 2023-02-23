#!/usr/bin/env python
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/monica/ros_catkin_ws_mine/src/skeleton/src/')
import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
from geometry_utility import GeomUtility
import collections  

# VARIABLES
K = np.array([[605.81918473810867, 0.0, 324.66592260274075], 
                           [0.0, 603.77141649252928, 236.93792129936679], 
                           [0.0, 0.0, 1.0]])

Ae_curr = np.array([[-0.13821154, 0.99031435, 0.01323344, -0.09489997],
 [ 0.99039982, 0.13816612, 0.00428674, 0.38990006],
 [ 0.00241681, 0.01369888, -0.99990331, 0.3915997 ],
 [ 0.,          0. ,         0.  ,        1.        ]]
)


#Ace = np.array([[-0.005812655098, -0.9991180429, 0.04158544615, 0.06709041116],
#                            [0.9999425447, -0.006181950697, -0.008757327031, -0.03431457374],
#                            [0.009006682622, 0.04153215353, 0.9990965719, -0.06649198096],
#                            [0.,  0.,  0.,  1.]])                                  #matrice calibrata 21 febbraio

Ace = np.array([[-0.005812655098, -0.9991180429, 0.04158544615, 0.06709041116],
                            [0.9999425447, -0.006181950697, -0.008757327031, -0.03431457374],
                            [0.009006682622, 0.04153215353, 0.9990965719, -0.05649198096],
                            [0.,  0.,  0.,  1.]])       #matrice aggiustata a mano solo la z

DetectedBB = collections.namedtuple('DetectedBB', 'type, classe,x,y,w,h')  
detectedBB = DetectedBB(type='type', classe='back_oil_separator_crankcase_castiron', x='0', y='0', w='0', h='0')  


depth_image = imageio.imread('/home/monica/ros_catkin_ws_mine/src/skeleton/data/depth_01.tiff')
depth = np.array(depth_image)

rgb_image = imageio.imread('/home/monica/ros_catkin_ws_mine/src/skeleton/data/rgb_01.png')
rgb = np.array(rgb_image)

key2D = np.array([[325, 153],
 [487, 158],
 [397, 295],
 [401, 206],
 [286, 213]]
)

keypoints3D = np.zeros((5,3))
# CONVERTI 2D IN 3D
for j in range(len(key2D)):
    keyCam_ = GeomUtility.deproject_pixel_to_point(key2D[j,:], depth, K)
    keyCam = np.array([keyCam_[0], keyCam_[1], keyCam_[2], 1])
    #print(key2D[j,:])
    #print('-_-_-_-_-_')
    #print(depth_array[320,240])
    print("Punto " + str(j) + " in terna camera")
    print(keyCam.T)
    print()
    #print('-_-_-_-_-_')
    # TRAFORMA I PUNTI IN COORD ROBOT
    key3D = Ae_curr.dot(Ace.dot(keyCam.T))
    #print('-_********** 3d')
    #print(np.array(key3D))
    keypoints3D[j,:] = key3D[:3]

print('Punti in terna base')
print(keypoints3D)
print()

[[-0.14494919  0.50905481  0.13303025]
 [-0.07046391  0.5093137   0.17299162]
 [-0.10500097  0.44218865  0.18607495]
 [-0.10698842  0.48185334  0.1758228 ]
 [-0.16107144  0.47523382  0.13055026]]

# FINO A QUI E' TUTTO GIUSTO, PERCHE' HO MISURATO CON IL ROBOT IL 4 KEYPOINT CHE E' IL PERNO ED E' CORRETTO IN TERNA BASE

# CALCOLA PUNTO DI PRESA
T_a = GeomUtility.computeGraspingPoint(detectedBB, keypoints3D)
R = T_a[:3,:3]
p = T_a[:3, 3]
#R e p ci arrivano dal ragionatore

print(R)
print(p)

quat = GeomUtility.r2quat(R)
print('{position: {x: ' + str(p[0]) + ', y: ' + str(p[1]) + ', z: ' + str(p[2]) + '}, orientation: {x: ' + str(quat[1]) + ', y: ' + str(quat[2]) + ', z: ' + str(quat[3]) + ', w: ' + str(quat[0]) + '}}')