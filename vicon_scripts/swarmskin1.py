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
from math import *


def start_recording(cf_names, lp_names, folder_name='test'):
    topics = ''
    for name in cf_names:
        topics += '/vicon/'+name+'/'+name + ' '
    for name in lp_names:
        topics += '/vicon/'+name+'/'+name + ' '
    os.system("mkdir -p "+PATH+folder_name)
    os.system("rosbag record -o "+PATH+folder_name+"/" + lp + ' ' +topics + ' __name:=recorder')


def land_detector():
    print "Land Detector..."
    land_time = - np.ones( min(len(drone_list), len(lp_list)) )
    switched_off = np.zeros(len(drone_list))
    landed_drones_number = 0
    while not rospy.is_shutdown():
        for drone in drone_list: drone.pose = drone.position()
        for lp in lp_list: lp.pose = lp.position()
        for d in range(len(drone_list)):
            for l in range(len(lp_list)):
                print abs(drone_list[d].pose[0] - lp_list[l].pose[0]), abs(drone_list[d].pose[1] - lp_list[l].pose[1]), abs(drone_list[d].pose[2] - lp_list[l].pose[2])
                if abs(drone_list[d].pose[0] - lp_list[l].pose[0])<0.10 and abs(drone_list[d].pose[1] - lp_list[l].pose[1])<0.10 and abs(drone_list[d].pose[2] - lp_list[l].pose[2])<0.07:
                    if switched_off[d]==0:
                        print "Switch off the motors, %d drone" %d
                        landed_drones_number += 1
                        # print "rosnode kill /node_"+cf_names[0]+"_"+lp_names[0]
                        if data_recording:
                            time.sleep(0.1)
                            os.system("rosnode kill /recorder")
                        if toFly:
                            for t in range(3): cf_list[d].stop() # stop motors
                    switched_off[d] = 1
                    # print "Drones landed: ", landed_drones_number
                    if landed_drones_number==len(drone_list): rospy.signal_shutdown("landed")


rospy.init_node('test_high_level')

""" initialization """
# Names and variables
TAKEOFFHEIGHT  = 2.0
data_recording = 1
toFly          = 1
PATH = "~/Desktop/Swarm/Swarmskin/data/"

cf_name = 'cf1'
cf_names = [cf_name]

lp_names = ['lp1', 'lp2', 'lp3', 'lp4']
# lp_names = ['lp3', 'lp1', 'lp4']
# lp_names = []


drone = swarmlib.Drone(cf_name)
drone_list = [drone]

lp_list = []
for lp_name in lp_names:
    lp_list.append( swarmlib.Mocap_object(lp_name) )



# landing_time = random.choice([13,22]) #13,22
landing_time = 17 # [s] 17, 22, 30, 40

print "landing_time", landing_time

lp = sys.argv[1]
experiment_type = sys.argv[2]
print "go to landing pad location... ", lp

if toFly:
    """ takeoff """
    print "adding.. ", cf_name
    cf = crazyflie.Crazyflie(cf_name, '/vicon/'+cf_name+'/'+cf_name)
    cf.setParam("commander/enHighLevel", 1)
    cf.setParam("stabilizer/estimator",  2) # Use EKF
    cf.setParam("stabilizer/controller", 2) # Use mellinger controller
    cf_list = [cf]
    print "takeoff.. ", cf.prefix
    for t in range(3):
        cf.takeoff(targetHeight = TAKEOFFHEIGHT, duration = 5.0)
    time.sleep(5.0)

    """ going to landing poses """
    # r = 0.3; theta1 = pi/4; theta2 = pi/4
    # l = 0.230 # distance between drones (arm length)
    # width = 0.45 # between person's shoulders
    # human_pose = np.array([-1.0,0.0]); hx = human_pose[0]; hy = human_pose[1]
    # goto_arr = [ [hx+ (r+l)*cos(theta1), hy+ (r+l)*sin(theta1) +width/2],
    #              [hx+ r*cos(theta1),     hy+ r*sin(theta1) +width/2],
    #              [hx+ r*cos(theta2),     hy- r*sin(theta2) -width/2],
    #              [hx+ (r+l)*cos(theta2), hy- (r+l)*sin(theta2) -width/2] ]
    goto_arr = [[-0.83, 0.70], [-0.97, 0.52], [-0.98, -0.50], [-0.85, -0.70]]

    for t in range(3): cf.goTo(goal = goto_arr[lp_names.index(lp)]+[TAKEOFFHEIGHT], yaw=0.0, duration = 3.0, relative = False)
    time.sleep(3.0)


    """ landing """
    print 'Landing...'
    for t in range(3): cf.land(targetHeight = -0.05, duration = landing_time)


if data_recording:
    print "Data recording started"
    pose_recording = Process(target=start_recording, args=(cf_names, lp_names, experiment_type, ))
    pose_recording.start()


land_detector()







    




























    # cf.goTo(goal = [0.0, -0.4, 0.0], yaw=0.0, duration = 2.0, relative = True)
    # time.sleep(2.0)

    # cf.goTo(goal = [0.0, 0.8, 0.0], yaw=0.0, duration = 4.0, relative = True)
    # time.sleep(4.0)

    # cf.goTo(goal = [0.0, -0.4, 0.0], yaw=0.0, duration = 2.0, relative = True)
    # time.sleep(2.0)






    # time_to_sleep = 1.5
    # cf.goTo(goal = [.5, .5, 0.0], yaw=-.75*3.14, duration = 2.0, relative = True)
    # time.sleep(3.0)
    # for i in range(2):
    #     cf.goTo(goal = [0.0, -1.0, 0.0], yaw=-1.57, duration = 2.0, relative = True)
    #     time.sleep(time_to_sleep)
    #     cf.goTo(goal = [-1.0, 0.0, 0.0], yaw=-1.57, duration = 2.0, relative = True)
    #     time.sleep(time_to_sleep)
    #     cf.goTo(goal = [0.0, 1.0, 0.0], yaw=-1.57, duration = 2.0, relative = True)
    #     time.sleep(time_to_sleep)
    #     cf.goTo(goal = [1.0, 0.0, 0.0], yaw=-1.57, duration = 2.0, relative = True)
    #     time.sleep(time_to_sleep)
    # cf.goTo(goal = [0, 0.0, 0.0], yaw=0, duration = 2.0, relative = False)
    # time.sleep(3.0)



    # traj1 = uav_trajectory.Trajectory()
    # traj1.loadcsv("takeoff.csv")

    # traj2 = uav_trajectory.Trajectory()
    # traj2.loadcsv("figure8.csv")

    # print(traj1.duration)

    # cf.uploadTrajectory(0, 0, traj1)
    # cf.uploadTrajectory(1, len(traj1.polynomials), traj2)

    # cf.startTrajectory(0, timescale=1.0)
    # time.sleep(traj1.duration * 2.0)

    # cf.startTrajectory(1, timescale=2.0)
    # time.sleep(traj2.duration * 2.0)

    # cf.startTrajectory(0, timescale=1.0, reverse=True)
    # time.sleep(traj1.duration * 1.0)

    # cf.stop()