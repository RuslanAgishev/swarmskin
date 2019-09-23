#!/usr/bin/env python

from __future__ import division
import rospy
import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped, TransformStamped
from crazyflie_driver.msg import FullState
from std_srvs.srv import Empty
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import Path

from math import *
import time
import numpy as np

import crazyflie
import swarmlib

import sys
import os
from multiprocessing import Process


def start_recording():
    print "Data recording started"
    PATH = "~/Desktop/Swarm/Swarmskin/data/" 
    os.system("rosbag record -o "+PATH+"/vicon_data /vicon/cf1/cf1 /vicon/cf2/cf2 /vicon/cf3/cf3 /vicon/cf4/cf4")


def takeoff():
	print "take off"
	while not rospy.is_shutdown():
		for i in range(400):
			for drone in drones:
				drone.sp[2] += 0.001
				if toFly: drone.fly()
				drone.publish_sp()
				drone.publish_path()
				rate.sleep()
		break

def hover():
	print "hovering..."
	if toFly:
		print "Switch on LEDs"
		# for cf in cf_list: cf.setParam("tf/state", 4) # LED is ON
	while not rospy.is_shutdown():
		for i in range(200):
			for drone in drones:
				if toFly: drone.fly()
				drone.publish_sp()
				drone.publish_path()
				rate.sleep()
		break

def land_detector():
	for i in range(min(len(drones), len(lp_list))):
	    if abs(drones[i].pose[0] - lp_list[i].pose[0])<0.07 and abs(drones[i].pose[1] - lp_list[i].pose[1])<0.07 and abs(drones[i].pose[2] - lp_list[i].pose[2])<0.05:
	    # if abs(drones[i].sp[0] - lp_list[i].pose[0])<0.07 and abs(drones[i].sp[1] - lp_list[i].pose[1])<0.07 and abs(drones[i].sp[2] - lp_list[i].pose[2])<0.07:
	        if toFly:
	        	print "Stop motors"
	        	for cf in cf_list: cf.stop()
	        time.sleep(1)
	        print "Shutdown"
	        rospy.signal_shutdown("landed")


# PARAMETERs #############
toFly            = 1
data_recording   = 0
const_height	 = 1
TakeoffHeight    = 1.8 # meters
TakeoffTime      = 4.0   # seconds
TimeFlightSec    = 5 # [s]
ViconRate        = 200 # [Hz]
N_samples        = ViconRate * TimeFlightSec


# cf_names         = ['cf1',
# 					'cf2',
# 					'cf3',
# 					'cf4'
# 				   ]
cf_names = ['cf2']

lp_names         = ['lp2', 'lp4']
# lp_names         = ['lp1', 'lp2', 'lp3', 'lp4']


if __name__ == '__main__':
	rospy.init_node('swarmskin', anonymous=True)
	rate = rospy.Rate(ViconRate)

	# drones init
	drones = []
	for name in cf_names:
		drones.append( swarmlib.Drone(name) )
		drones[-1].sp = drones[-1].position()

	# landing pads init
	lp_list = []
	for lp_name in lp_names: lp_list.append( swarmlib.Mocap_object(lp_name) )

	if toFly:
		cf_list = []
		for name in cf_names:
		    cf = crazyflie.Crazyflie(name, '/vicon/'+name+'/'+name)
		    cf.setParam("commander/enHighLevel", 1)
		    # cf.setParam("stabilizer/estimator", 2)  # Use EKF
		    # cf.setParam("stabilizer/controller", 2) # Use mellinger controller
		    # cf.takeoff(targetHeight = TakeoffHeight, duration = TakeoffTime)
		    cf_list.append(cf)
		# time.sleep(TakeoffTime+1)
		# for cf in cf_list: cf.land(targetHeight = -0.1, duration = 3.0)
		# time.sleep(3.0)
		# for cf in cf_list: cf.stop()

	if data_recording:
		pose_recording = Process(target=start_recording)
		pose_recording.start()


	takeoff()

	# """ trajectory generation """
	start_poses = []
	for drone in drones: start_poses.append( drone.sp )

	l = 0.3
	# goal_poses = [ lp_list[0].position()[:2], lp_list[1].position()[:2], [ l,  -l], [ 0.0,-l] ]
	goal_poses = [ [ 0, l], [ l, l], [ l, -l], [ 0, -l] ]
	for i in range(len(drones)):
		drones[i].traj = np.array([np.linspace(start_poses[i][0], goal_poses[i][0], N_samples),
							       np.linspace(start_poses[i][1], goal_poses[i][1], N_samples),
							       np.linspace(start_poses[i][2], TakeoffHeight,    N_samples)]).T

	sp_ind = 0
	print "going to landing location..."
	while not rospy.is_shutdown():
		if sp_ind >= N_samples-1: sp_ind = N_samples-1 # holding the last pose
		for drone in drones: drone.sp = drone.traj[sp_ind,:]
		sp_ind += 1

		# TO FLY
		if toFly:
			for drone in drones: drone.fly()

		# TO VISUALIZE
		for drone in drones:
			drone.publish_sp()
			drone.publish_path(limit=N_samples)

		""" Landing """
		if sp_ind == N_samples-1:
			# hover and LEDs switch
			hover()

			print 'Landing!!!'
			while not rospy.is_shutdown():
				land_detector()
				for drone in drones: drone.sp[2] -= 0.0005
				if toFly:
					for drone in drones: drone.fly()
				for drone in drones:
					drone.publish_sp()
					drone.publish_path()

				if drones[0].sp[2]<-1.0: #and drones[1].sp[2]<-1.0 and drones[2].sp[2]<-1.0 and drones[3].sp[2]<-1.0:
					time.sleep(1)
					if toFly:
						for cf in cf_list: cf.stop()
					print 'reached the floor, shutdown'
					rospy.signal_shutdown('landed')
				rate.sleep()

		rate.sleep()
