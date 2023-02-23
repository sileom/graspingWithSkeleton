#!/usr/bin/env python

import rospy
from std_msgs.msg import String


if __name__ == '__main__':
    try:
        pub = rospy.Publisher('trigger_camera', String, queue_size=10)
        rospy.init_node('camera_input', anonymous=True)
        rate = rospy.Rate(10) # 10hz
        while not rospy.is_shutdown():
            c = input("Press i to take image or s to start or e to exit!\n")
            if (c == 'i'):
                msg = 'take_image'
                pub.publish(msg)
            if (c == 's'):
                msg = 'start'
                pub.publish(msg)
            elif (c == 'e'):
                exit()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass