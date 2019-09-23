#!/usr/bin/env python

import rospy
import crazyflie
import time
import uav_trajectory

import swarmlib
import message_filters
from geometry_msgs.msg import PoseStamped, TransformStamped
import os
import sys
from multiprocessing import Process
import random
import numpy as np
from numpy.linalg import norm




def start_recording():
    print "Data recording started"
    os.system("mkdir -p "+PATH+Subject_name)
    os.system("rosbag record -o "+PATH+Subject_name+"/vicon_data /vicon/cf1/cf1 /vicon/cf2/cf2 /vicon/cf3/cf3 /vicon/cf4/cf4")
    # os.system("rosbag record -o "+PATH+"poses /vicon/cf1/cf1 /vicon/cf2/cf2 /vicon/cf3/cf3 /vicon/cf4/cf4 /vicon/lp1/lp1 /vicon/lp2/lp2 /vicon/lp3/lp3 /vicon/lp4/lp4")

def land_detector():
    land_time = - np.ones( min(len(drone_list), len(lp_list)) )
    switched_off = np.zeros(len(drone_list))
    while not rospy.is_shutdown():
        for drone in drone_list: drone.position()
        for lp in lp_list: lp.position()
        landed_drones_number = 0
        for i in range(min(len(drone_list), len(lp_list))):
            print(abs(drone_list[i].pose[0] - lp_list[i].pose[0]), abs(drone_list[i].pose[1] - lp_list[i].pose[1]), abs(drone_list[i].pose[2] - lp_list[i].pose[2]))
            if abs(drone_list[i].pose[0] - lp_list[i].pose[0])<0.07 and abs(drone_list[i].pose[1] - lp_list[i].pose[1])<0.07 and abs(drone_list[i].pose[2] - lp_list[i].pose[2])<0.1:
                landed_drones_number += 1
                # if land_time[i]==-1:
                #     land_time[i] = time.time()-start_time
                #     print("Drone %d is landed after %s seconds" % (i+1, land_time[i]))
                if toFly:
                    print "Switch off the motors, %d drone" %i
                    if switched_off[i]==0:
                        for t in range(3): cf_list[i].stop()                  # stop motors
                    switched_off[i] = 1
                    # cf_list[i].setParam("tf/state", 0) # switch off LEDs
                if landed_drones_number==len(drone_list): rospy.signal_shutdown("landed")


def flight(cf_list, TakeoffHeight, goal_poses_XY):
    num_commands = 5
    for t in range(num_commands):
        print("Takeoff")
        for cf in cf_list:
            cf.takeoff(targetHeight=TakeoffHeight, duration=8.0)
    time.sleep(8.0)

    print("Moving to landing positions")
    for t in range(num_commands):
        for i in range(len(cf_list)):
            print goal_poses_XY[i]
            cf_list[i].goTo(goal=[goal_poses_XY[i][0], goal_poses_XY[i][1], TakeoffHeight], yaw=0.0, duration=6.0, relative=False)
    time.sleep(8.0)

    print("Switch on LEDs")
    for cf in cf_list:
        cf.setParam("tf/state", 4) # LED is ON
    global start_time
    start_time = time.time()
    time.sleep(0.1)

    for t in range(num_commands):
        for cf in cf_list:
            print 'Landing...'
            cf.land(targetHeight=-0.1, duration=landing_velocity)





if __name__ == '__main__':
    rospy.init_node('swarmskin')

    """ initialization """
    # Names and variables
    TakeoffHeight  = 2.0
    data_recording = 0

    toFly          = 1

    # lp_names = []

    lp_names = ['lp4', 'lp1', 'lp2']
    cf_names = ['cf1', 'cf2', 'cf3']
    # lp_names = ['lp4']
    # cf_names = ['cf1']
    # cf_names = ['cf1', 'cf2', 'cf3', 'cf4']
    PATH = "~/Desktop/Swarm/Swarmskin/data/"       

    drone_list = []
    for name in cf_names:
        drone = swarmlib.Drone(name)
        drone_list.append(drone)

    # landing pads init
    lp_list = []
    for lp_name in lp_names:
        lp_list.append( swarmlib.Mocap_object(lp_name) )

    # landing_velocity = random.choice([13,22]) #13,22
    landing_velocity = 20 # 22 , 30
    print "landing_velocity", landing_velocity

    if data_recording:
        try:
            Subject_name = sys.argv[1]
        except:
            print "\nEnter subject's name. For example:"
            print "rosrun crazyflie_demo swarmskin.py NAME"
            os.system("rosnode kill /swarmskin")
            sys.exit(1)
        pose_recording = Process(target=start_recording)
        pose_recording.start()


    # flight to landing positions
    l=0.25
    global_goal_poses = [ [ -0.4, 1*l], [ -0.4, 2*l], [ -0.4, 3*l] ]
    if toFly:
        cf_list = []
        for cf_name in cf_names:
            cf = crazyflie.Crazyflie(cf_name, '/vicon/'+cf_name+'/'+cf_name)
            cf.setParam("commander/enHighLevel", 1)
            cf.setParam("stabilizer/estimator", 2)  # Use EKF
            cf.setParam("stabilizer/controller", 2) # Use Mellinger controller
            time.sleep(0.1)
            cf_list.append(cf)

        flight(cf_list, TakeoffHeight, global_goal_poses)


    #detect if drones touch landing pads
    print "start land detector"
    land_detector()



