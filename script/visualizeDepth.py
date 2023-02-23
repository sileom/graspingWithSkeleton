#!/usr/bin/env python
import sys
# the mock-0.3.1 dir contains testcase.py, testutils.py & mock.py
sys.path.append('/home/monica/ros_catkin_ws_mine/src/skeleton/src/')
import imageio
import matplotlib.pyplot as plt
import cv2
import numpy as np


i = imageio.imread('/home/monica/ros_catkin_ws_mine/src/skeleton/data/depth_01.tiff')
print(type(i))
nimg = np.array(i)


#img8 = cv2.imread('/home/monica/ros_catkin_ws_mine/src/skeleton/data/depth.tiff', )

#cv2.imshow('o',a)
#cv2.waitKey(0)

k = np.array([[249, 307],
 [305, 236],
 [159, 281],
 [271, 386],
 [361, 270],
                ])



plt.imshow(i)
plt.scatter(k[0,0], k[0,1], color="red") # plotting single point
plt.scatter(k[1,0], k[1,1], color="green") # plotting single point
plt.scatter(k[2,0], k[2,1], color="blue") # plotting single point
plt.scatter(k[3,0], k[3,1], color="black") # plotting single point
plt.scatter(k[4,0], k[4,1], color="white") # plotting single point
#for i in range(5):
#    plt.scatter(k[i,0], k[i,1], color="red") # plotting single point
plt.show()
