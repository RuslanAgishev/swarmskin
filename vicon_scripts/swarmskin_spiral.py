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
from threading import Thread
import random
import numpy as np
from numpy.linalg import norm
from math import *




def start_recording():
    print "Data recording started"
    os.system("mkdir -p "+PATH+Subject_name)
    os.system("rosbag record -o "+PATH+Subject_name+"/vicon_data /vicon/cf1/cf1 /vicon/cf2/cf2 /vicon/cf3/cf3 /vicon/cf4/cf4")
    # os.system("rosbag record -o "+PATH+"poses /vicon/cf1/cf1 /vicon/cf2/cf2 /vicon/cf3/cf3 /vicon/cf4/cf4 /vicon/lp1/lp1 /vicon/lp2/lp2 /vicon/lp3/lp3 /vicon/lp4/lp4")

def land_detector():
    for lp in lp_list: lp.pose = lp.position()
    print abs(drone_list[0].pose[0] - lp_list[0].pose[0]), abs(drone_list[0].pose[1] - lp_list[0].pose[1]), abs(drone_list[0].pose[2] - lp_list[0].pose[2])
    landed_drones_number = 0
    for i in range(min(len(drone_list), len(lp_list))):
        if abs(drone_list[i].pose[0] - lp_list[i].pose[0])<0.07 and abs(drone_list[i].pose[1] - lp_list[i].pose[1])<0.07 and abs(drone_list[i].pose[2] - lp_list[i].pose[2])<0.05:
        # if abs(drones[i].sp[0] - lp_list[i].pose[0])<0.07 and abs(drones[i].sp[1] - lp_list[i].pose[1])<0.07 and abs(drones[i].sp[2] - lp_list[i].pose[2])<0.07:
            landed_drones_number += 1
            if toFly:
                print "Stop motors %d drone" %i
                for t in range(5): cf_list[i].stop()
            if landed_drones_number==len(drone_list): rospy.signal_shutdown("landed")


def flight(cf_list, TakeoffHeight, goal_poses_XY):
    num_commands = 5
    for t in range(num_commands):
        print("Takeoff")
        for cf in cf_list:
            cf.takeoff(targetHeight=TakeoffHeight, duration=6.0)
    time.sleep(6.0)



def hover():
    print "hovering..."
    if toFly:
        print "Switch on LEDs"
        for cf in cf_list: cf.setParam("tf/state", 4) # LED is ON
    while not rospy.is_shutdown():
        for i in range(200):
            for drone in drone_list:
                if toFly: drone.fly()
                drone.publish_sp()
                drone.publish_path()
                rate.sleep()
        break



if __name__ == '__main__':
    rospy.init_node('swarmskin')

    """ initialization """
    TakeoffHeight  = 2.0
    data_recording = 0
    toFly          = 1
    TimeFlightSec    = 3 # [s]
    ViconRate        = 200 # [Hz]
    N_samples        = ViconRate * TimeFlightSec
    lp_names = []
    lp_names = ['lp2', 'lp4']
    cf_names = ['cf1', 'cf2']

    PATH = "~/Desktop/Swarm/Swarmskin/data/"
    l=0.25; global_goal_poses = [ [ -0.4, 2*l], [ -0.4+l, 2*l] ]

    # landing pads init
    lp_list = []
    for lp_name in lp_names:
        lp_list.append( swarmlib.Mocap_object(lp_name) )

    # landing_velocity = random.choice([13,22]) #13,22
    landing_velocity = 10 # 22 , 30
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


    if toFly:
        cf_list = []
        for cf_name in cf_names:
            cf = crazyflie.Crazyflie(cf_name, '/vicon/'+cf_name+'/'+cf_name)
            cf.setParam("commander/enHighLevel", 1)
            cf.setParam("stabilizer/estimator",  2) # Use EKF
            cf.setParam("stabilizer/controller", 2) # Use Mellinger controller
            time.sleep(0.1)
            cf_list.append(cf)

        flight(cf_list, TakeoffHeight, global_goal_poses)
        # thread_list = []
        # for i in range(len(cf_list)):
        #     thread_list.append( Thread(target=flight, args=(cf_list[i], drone_list[i], TakeoffHeight, global_goal_poses[i],)) )
        #     thread_list[-1].start()

    drone_list = []
    for name in cf_names:
        drone = swarmlib.Drone(name)
        drone_list.append(drone)
        drone_list[-1].sp = drone_list[-1].position()

    # """ trajectory generation """
    start_poses = []
    for drone in drone_list: start_poses.append( drone.sp )
    for i in range(len(drone_list)):
        drone_list[i].traj = np.array([np.linspace(start_poses[i][0], global_goal_poses[i][0], N_samples),
                                       np.linspace(start_poses[i][1], global_goal_poses[i][1], N_samples),
                                       np.linspace(start_poses[i][2], TakeoffHeight,    N_samples)]).T

    sp_ind = 0
    print "going to landing location..."
    rate = rospy.Rate(200)
    while not rospy.is_shutdown():
        if sp_ind >= N_samples-1: sp_ind = N_samples-1 # holding the last pose
        for drone in drone_list: drone.sp = drone.traj[sp_ind,:]
        sp_ind += 1

        # TO FLY
        if toFly:
            for drone in drone_list: drone.fly()

        # TO VISUALIZE
        for drone in drone_list:
            drone.publish_sp()
            drone.publish_path(limit=N_samples)

        """ Landing """
        if sp_ind == N_samples-1:
            # hover and LEDs switch
            hover()

            print 'Landing!!!'
            rate_land = rospy.Rate(10)
            N = 50; T = np.linspace(0,2*pi,N); t = 0
            r = 0.1
            while not rospy.is_shutdown():
                # print "start land detector"
                land_detector() # detect if drones touch landing pads
                for i in range(len(drone_list)):
                    drone_list[i].position()
                    drone_list[i].sp[0] = global_goal_poses[i][0] + r * (cos(T[t])-1)
                    drone_list[i].sp[1] = global_goal_poses[i][1] + r * sin(T[t])
                    drone_list[i].sp[2] -= 0.007
                if toFly:
                    for drone in drone_list: drone.fly()
                for drone in drone_list:
                    drone.publish_sp()
                    drone.publish_path()

                if drone_list[0].sp[2]<-0.3: #and drones[1].sp[2]<-1.0 and drones[2].sp[2]<-1.0 and drones[3].sp[2]<-1.0:
                    time.sleep(1)
                    if toFly:
                        for cf in cf_list: cf.stop()
                    print 'reached the floor, shutdown'
                    rospy.signal_shutdown('landed')
                t+=1; t = t%N
                rate_land.sleep()

        rate.sleep()

    
    





