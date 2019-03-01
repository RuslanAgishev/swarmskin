#!/usr/bin/env python


import csv

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

subject_name_list = ['Taha', 'Luisa', 'Miguel', 'DimaM', 'Professor', 'Akerke', 'Ruslan']
# subject_name_list = ['Miguel']



# DISPLACEMENT #############################################################################3

xy_error_list_v = np.array([])
xy_error_list_h = np.array([])
xy_error_list_vh = np.array([])
displacement_list_v_x = np.array([])
displacement_list_v_y = np.array([])
displacement_list_h_x = np.array([])
displacement_list_h_y = np.array([])
displacement_list_vh_x = np.array([])
displacement_list_vh_y = np.array([])
y_human_list_v = np.array([])
y_human_list_h = np.array([])
y_human_list_vh = np.array([])


for name in subject_name_list:
	print "\nName:", name

	directory_v = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/V/experiment/"
	directory_h = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/H/experiment/"
	directory_vh = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/VH/experiment/"

	list_of_experiments = [directory_v, directory_h, directory_vh]

	for exper_type in list_of_experiments:
		experiment_list = sorted([x[0] for x in os.walk(exper_type)])
		experiment_list.pop(0)
		xy_error_list = np.array([])
		displacement_list_x = np.array([])
		displacement_list_y = np.array([])
		
		# for drone in ['100','101']:
		for drone in ['101']:

			for experiment_location in experiment_list:
				# print "experiment_location", experiment_location

				x_cf = np.array([]); y_cf = np.array([]); z_cf = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_crazyflie'+drone+'_slash_crazyflie'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_cf = np.append(x_cf, float(row[10]))
							y_cf = np.append(y_cf, float(row[11]))
							z_cf = np.append(z_cf, float(row[12]))

				x_lp = np.array([]); y_lp = np.array([]); z_lp = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_landing_pad'+drone+'_slash_landing_pad'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_lp = np.append(x_lp, float(row[10]))
							y_lp = np.append(y_lp, float(row[11]))
							z_lp = np.append(z_lp, float(row[12]))

				# x_human = np.array([]); y_human = np.array([]); z_human = np.array([])
				# human_time = np.array([])
				# csv_file_name = experiment_location + '/_slash_vicon_slash_human_head_slash_human_head.csv'
				# with open(csv_file_name) as csvfile:
				# 	reader = csv.reader(csvfile)
				# 	for row in reader:
				# 		if row[10] != "x": # skip the first line
				# 			# x_human = np.append(x_lp, float(row[10]))
				# 			y_human_list = np.append(y_lp, float(row[10]))
				# 			human_time = np.append(human_time, float(row[0]))
				# 			# z_lp = np.append(z_lp, float(row[12]))
				
				# y_human_list = y_human_list[:-30]
				# y_human_time = human_time[:len(y_human_list)] / 1000000000  # to seconds
				# y_human_vel = abs(np.diff(y_human_list) / np.diff(y_human_time))
				# print 'y_human_vel.mean()', y_human_vel.mean()

				# if len(x_lp)<400:

				cf_last_pose = np.array([x_cf[-1], y_cf[-1]])
				lp_last_pose = np.array([x_lp[-1], y_lp[-1]])
				# print "cf_last_pose", cf_last_pose
				# print "lp_last_pose", lp_last_pose
				xy_error = np.linalg.norm(cf_last_pose-lp_last_pose)
				# print "xy_error", xy_error
				xy_error_list = np.append(xy_error_list, xy_error)

				displacement = cf_last_pose - lp_last_pose
				# print "displacement", displacement
				displacement_x = displacement[0]
				displacement_y = displacement[1]

				displacement_list_x = np.append(displacement_list_x, displacement_x)
				displacement_list_y = np.append(displacement_list_y, displacement_y)
			
			if "/V/" in exper_type[-16:]:
				# print "mean xy_error V:", xy_error_list.mean()
				xy_error_list_v = np.append(xy_error_list_v, xy_error_list)
				displacement_list_v_x = np.append(displacement_list_v_x, displacement_list_x)
				displacement_list_v_y = np.append(displacement_list_v_y, displacement_list_y)
				# y_human_list_v = np.append(y_human_list_v, y_human_vel.mean())
			if "/H/" in exper_type[-16:]:
				# print "mean xy_error H:", xy_error_list.mean()
				xy_error_list_h = np.append(xy_error_list_h, xy_error_list)
				displacement_list_h_x = np.append(displacement_list_h_x, displacement_list_x)
				displacement_list_h_y = np.append(displacement_list_h_y, displacement_list_y)
				# y_human_list_h = np.append(y_human_list_h, y_human_vel.mean())
			if "/VH/" in exper_type[-16:]:
				# print "mean xy_error VH:", xy_error_list.mean()
				xy_error_list_vh = np.append(xy_error_list_vh, xy_error_list)
				displacement_list_vh_x = np.append(displacement_list_vh_x, displacement_list_x)
				displacement_list_vh_y = np.append(displacement_list_vh_y, displacement_list_y)
				# y_human_list_vh = np.append(y_human_list_vh, y_human_vel.mean())

print "\nmean xy_error V total:", xy_error_list_v.mean()
print "mean xy_error H total:", xy_error_list_h.mean()
print "mean xy_error VH total:", xy_error_list_vh.mean()

print "\nstandard deviation xy_error V total:", np.std(xy_error_list_v)
print "standard deviation xy_error H total:", np.std(xy_error_list_h)
print "standard deviation xy_error VH total:", np.std(xy_error_list_vh)

print "\nmaximum xy_error V total:", np.amax(xy_error_list_v)
print "maximum xy_error H total:", np.amax(xy_error_list_h)
print "maximum xy_error VH total:", np.amax(xy_error_list_vh)

# print "\nvariance xy_error V total:", np.var(xy_error_list_v)
# print "variance xy_error H total:", np.var(xy_error_list_h)
# print "variance xy_error VH total:", np.var(xy_error_list_vh)

# print '\ny_human_list_v', y_human_list_v.mean()
# print 'y_human_list_h', y_human_list_h.mean()
# print 'y_human_list_vh', y_human_list_vh.mean()


# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
f, ((ax1, ax4), (ax2, ax5), (ax3, ax6)) = plt.subplots(3, 2)
# x = displacement_list_h_x
# y = displacement_list_h_y
ax1.plot(displacement_list_v_x, displacement_list_v_y, 'g^')
# ax1.plot(displacement_list_v_x.mean(), displacement_list_v_y.mean(), 'o', color='c', markersize=20)
ax1.set_title('Vision feedback')
ax2.plot(displacement_list_h_x, displacement_list_h_y, 'bs')
# ax2.plot(displacement_list_h_x.mean(), displacement_list_h_y.mean(), 'o', color='c', markersize=20)
ax2.set_title('Haptics feedback')
ax3.plot(displacement_list_vh_x, displacement_list_vh_y, 'ro')
# ax3.plot(displacement_list_vh_x.mean(), displacement_list_vh_y.mean(), 'o', color='c', markersize=20)
ax3.set_title('Vision and Haptics feedback')
# ax4.plot(displacement_list_vh_x, displacement_list_vh_y, 'ro', label="Vision+Haptics")
# ax4.plot(displacement_list_v_x, displacement_list_v_y, 'g^', label="Vision")
# ax4.plot(displacement_list_h_x, displacement_list_h_y, 'bs', label="Haptics")

# ax4.legend()
# ax4.set_title('Linear regression')
# ax4.plot(displacement_list_v_x_test, displacement_list_v_x_pred, color = 'g')
# ax4.plot(displacement_list_h_x_test, displacement_list_h_x_pred, color = 'b')
# ax4.plot(displacement_list_vh_x_test, displacement_list_vh_x_pred, color = 'r')

ax1.axis([-0.06, 0.06, -0.06, 0.06])
ax2.axis([-0.06, 0.06, -0.06, 0.06])
ax3.axis([-0.06, 0.06, -0.06, 0.06])
ax4.axis([-0.06, 0.06, -0.06, 0.06])
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
ax3.axhline(y=0, color='k')
ax3.axvline(x=0, color='k')
# ax4.axhline(y=0, color='k')
# ax4.axvline(x=0, color='k')

# ax1.set_ylabel('Meters, m')
# ax3.set_ylabel('Meters, m')
ax3.set_xlabel('Meters, m')
ax6.set_xlabel('Meters, m')
# ax4.set_xlabel('Meters, m')

ax1.set_ylabel('Meters, m')
ax2.set_ylabel('Meters, m')
ax3.set_ylabel('Meters, m')

ax4.set_ylabel('Number of landings')
ax5.set_ylabel('Number of landings')
ax6.set_ylabel('Number of landings')

# plt.suptitle('Displacement of the drone in respect to the landing pad, after landing, in XY plane', fontsize=18)




# ax1.text(-0.058, 0.055, r'mean xy_error V total:   ' + str(xy_error_list_v.mean())[:6]+" m", fontsize=12)
# ax1.text(-0.058, 0.05, r'variance x_error V total: ' + str(np.var(displacement_list_v_x))[:8]+" m", fontsize=12)
# ax1.text(-0.058, 0.045, r'variance y_error V total:  ' + str(np.var(displacement_list_v_y))[:8]+" m", fontsize=12)

# ax2.text(-0.058, 0.055, r'mean xy_error H total:   ' + str(xy_error_list_h.mean())[:6]+" m", fontsize=12)
# ax2.text(-0.058, 0.05, r'variance x_error H total: ' + str(np.var(displacement_list_h_x))[:8]+" m", fontsize=12)
# ax2.text(-0.058, 0.045, r'variance y_error H total:  ' + str(np.var(displacement_list_h_y))[:8]+" m", fontsize=12)

# ax3.text(-0.058, 0.055, r'mean xy_error VH total:   ' + str(xy_error_list_vh.mean())[:6]+" m", fontsize=12)
# ax3.text(-0.058, 0.05, r'variance x_error VH total: ' + str(np.var(displacement_list_vh_x))[:8]+" m", fontsize=12)
# ax3.text(-0.058, 0.045, r'variance y_error VH total:  ' + str(np.var(displacement_list_vh_y))[:8]+" m", fontsize=12)




# Circles
print
threshhold = 0
for i in range(300):
	threshhold = threshhold + 0.0005
	xy_error_list_v_new = xy_error_list_v[xy_error_list_v<threshhold]
	# xy_error_list_v_new = xy_error_list_v[xy_error_list_v<0.018]
	rate = 100*(len(xy_error_list_v_new))/(len(xy_error_list_v))
	if rate>90:
		print "threshhold V", threshhold
		print "rate V", rate
		break
disk = plt.Circle((0, 0), threshhold, color='k', fill=False)
ax1.add_artist(disk)

threshhold = 0
for i in range(300):
	threshhold = threshhold + 0.0005
	xy_error_list_h_new = xy_error_list_h[xy_error_list_h<threshhold]
	# xy_error_list_v_new = xy_error_list_v[xy_error_list_v<0.018]
	rate = 100*(len(xy_error_list_h_new))/(len(xy_error_list_h))
	if rate>90:
		print "threshhold H", threshhold
		print "rate H", rate
		break
disk = plt.Circle((0, 0), threshhold, color='k', fill=False)
ax2.add_artist(disk)

threshhold = 0
for i in range(300):
	threshhold = threshhold + 0.0005
	xy_error_list_vh_new = xy_error_list_vh[xy_error_list_vh<threshhold]
	# xy_error_list_v_new = xy_error_list_v[xy_error_list_v<0.018]
	rate = 100*(len(xy_error_list_vh_new))/(len(xy_error_list_vh))
	if rate>90:
		print "threshhold VH", threshhold
		print "rate VH", rate
		break
disk = plt.Circle((0, 0), threshhold, color='k', fill=False)
ax3.add_artist(disk)



# HISTOGRAMS
ax4.hist(xy_error_list_v, bins=8, color='g')  # arguments are passed to np.histogram
# plt.title("Hist xy_error V total")
ax4.axis([0, 0.08, 0,30])


# print 'xy_error_list_h', xy_error_list_h
xy_error_list_h = xy_error_list_h[xy_error_list_h<0.1]
ax5.hist(xy_error_list_h, bins=8, color='b')  # arguments are passed to np.histogram
# plt.title("Hist xy_error H total")
ax5.axis([0, 0.08, 0,30])

ax6.hist(xy_error_list_vh, bins=8, color='r')  # arguments are passed to np.histogram
# plt.title("Hist xsy_error VH total")
ax6.axis([0, 0.08, 0,30])









# Liner regression
displacement_list_v_x = displacement_list_v_x.reshape(-1, 1)
displacement_list_v_y = displacement_list_v_y.reshape(-1, 1)
# Create linear regression object
regr_v = linear_model.LinearRegression()
# Train the model using the training sets
regr_v.fit(displacement_list_v_x, displacement_list_v_y)
# Make predictions using the testing set
displacement_list_v_x_test = np.linspace(-0.05, 0.05, num=50)
displacement_list_v_x_test = displacement_list_v_x_test.reshape(-1, 1)
displacement_list_v_x_pred = regr_v.predict(displacement_list_v_x_test)

displacement_list_h_x = displacement_list_h_x.reshape(-1, 1)
displacement_list_h_y = displacement_list_h_y.reshape(-1, 1)
# Create linear regression object
regr_h = linear_model.LinearRegression()
# Train the model using the training sets
regr_h.fit(displacement_list_h_x, displacement_list_h_y)
# Make predictions using the testing set
displacement_list_h_x_test = np.linspace(-0.05, 0.05, num=50)
displacement_list_h_x_test = displacement_list_h_x_test.reshape(-1, 1)
displacement_list_h_x_pred = regr_h.predict(displacement_list_h_x_test)

displacement_list_vh_x = displacement_list_vh_x.reshape(-1, 1)
displacement_list_vh_y = displacement_list_vh_y.reshape(-1, 1)
# Create linear regression object
regr_vh = linear_model.LinearRegression()
# Train the model using the training sets
regr_vh.fit(displacement_list_vh_x, displacement_list_vh_y)
# Make predictions using the testing set
displacement_list_vh_x_test = np.linspace(-0.05, 0.05, num=50)
displacement_list_vh_x_test = displacement_list_vh_x_test.reshape(-1, 1)
displacement_list_vh_x_pred = regr_vh.predict(displacement_list_vh_x_test)

# plt.plot(displacement_list_vh_x, displacement_list_vh_y, 'ro', label="Vision+Haptics")
# plt.plot(displacement_list_v_x, displacement_list_v_y, 'g^', label="Vision")
# plt.plot(displacement_list_h_x, displacement_list_h_y, 'bs', label="Haptics")

# plt.legend()
# plt.title('Linear regression')
ax1.plot(displacement_list_v_x_test, displacement_list_v_x_pred, color = 'g')
ax2.plot(displacement_list_h_x_test, displacement_list_h_x_pred, color = 'b')
ax3.plot(displacement_list_vh_x_test, displacement_list_vh_x_pred, color = 'r')












# plt.show()

























# TRAJECTORIES #######################################################################


# x_cf_before_landing = np.array([])
# y_cf_before_landing = np.array([])

x_lp_total_v_hs = np.array([])
y_lp_total_v_hs = np.array([])
lp_total_v_hs_time = np.array([])
accel_lp_v_hs = np.array([])

x_lp_total_v_ls = np.array([])
y_lp_total_v_ls = np.array([])
lp_total_v_ls_time = np.array([])
accel_lp_v_ls = np.array([])

x_lp_total_h_hs = np.array([])
y_lp_total_h_hs = np.array([])
lp_total_h_hs_time = np.array([])
accel_lp_h_hs = np.array([])

x_lp_total_h_ls = np.array([])
y_lp_total_h_ls = np.array([])
lp_total_h_ls_time = np.array([])
accel_lp_h_ls = np.array([])

x_lp_total_vh_hs = np.array([])
y_lp_total_vh_hs = np.array([])
lp_total_vh_hs_time = np.array([])
accel_lp_vh_hs = np.array([])

x_lp_total_vh_ls = np.array([])
y_lp_total_vh_ls = np.array([])
lp_total_vh_ls_time = np.array([])
accel_lp_vh_ls = np.array([])


dynamics_v_hs = np.array([0,0,0,0])
dynamics_h_hs = np.array([0,0,0,0])
dynamics_vh_hs = np.array([0,0,0,0])

dynamics_v_ls = np.array([0,0,0,0])
dynamics_h_ls = np.array([0,0,0,0])
dynamics_vh_ls = np.array([0,0,0,0])


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
plt.suptitle('Human head trajectories', fontsize=18)

for name in subject_name_list:
	print "\nName:", name

	directory_v = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/V/experiment/"
	directory_h = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/H/experiment/"
	directory_vh = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/VH/experiment/"

	list_of_experiments = [directory_v, directory_h, directory_vh]

	for exper_type in list_of_experiments:
		# print "Experiment_type", exper_type
		experiment_list = sorted([x[0] for x in os.walk(exper_type)])
		experiment_list.pop(0)
		# print "experiment_list", experiment_list
		xy_error_list = np.array([])
		displacement_list_x = np.array([])
		displacement_list_y = np.array([])
		for experiment_location in experiment_list:
			# print "experiment_location", experiment_location


			# x_cf = np.array([]); y_cf = np.array([])
			# csv_file_name = experiment_location + '/_slash_vicon_slash_crazyflie100_slash_crazyflie100.csv'
			# with open(csv_file_name) as csvfile:
			# 	reader = csv.reader(csvfile)
			# 	for row in reader:
			# 		if row[10] != "x": # skip the first line
			# 			x_cf = np.append(x_cf, float(row[10]))
			# 			y_cf = np.append(y_cf, float(row[11]))

			x_lp = np.array([]); y_lp = np.array([])
			lp_time = np.array([])
			csv_file_name = experiment_location + '/_slash_vicon_slash_human_head_slash_human_head.csv'
			with open(csv_file_name) as csvfile:
				reader = csv.reader(csvfile)
				for row in reader:
					if row[10] != "x": # skip the first line
						x_lp = np.append(x_lp, float(row[10]))
						y_lp = np.append(y_lp, float(row[11]))
						lp_time = np.append(lp_time, float(row[0]))








			# x_lp_forvelcalc = x_lp[-200:-30]
			# lp_time = lp_time[-200:-30]/1000000000 # to seconds
			#
			# vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
			# lp_time = np.delete(lp_time, 0)
			# acc = np.diff(vel) / np.diff(lp_time)
			# lp_time = np.delete(lp_time, 0)
			# jerk = np.diff(acc) / np.diff(lp_time)
			# lp_time = np.delete(lp_time, 0)
			# snap = np.diff(jerk) / np.diff(lp_time)
			# dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
			# # print "dynamics", dynamics


			# print "len x_lp", len(x_lp)
			# print "x_cf before landing", x_cf[-50]
			# x_cf_before_landing = np.append(x_cf_before_landing, x_cf[-50])
			# y_cf_before_landing = np.append(y_cf_before_landing, y_cf[-50])

			if len(x_lp)<400: # High speed
				frames_to_consider = 200
				# x_lp=x_lp[-frames_to_consider:]
				# y_lp=y_lp[-frames_to_consider:]

				y_lp_forvelcalc = y_lp[-frames_to_consider:-30]
				lp_time = lp_time[-frames_to_consider:-30] / 1000000000  # to seconds

				vel = np.diff(y_lp_forvelcalc) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				acc = np.diff(vel) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				jerk = np.diff(acc) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				snap = np.diff(jerk) / np.diff(lp_time)
				dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
				# print "dynamics", dynamics

				if "/V/" in exper_type[-16:]:
					# print "V hs"
					ax1.plot(x_lp, y_lp, color = 'g')
					x_lp_total_v_hs = np.append(x_lp_total_v_hs, x_lp[-frames_to_consider:-40])
					y_lp_total_v_hs = np.append(y_lp_total_v_hs, y_lp[-frames_to_consider:-40])
					# accel_lp_v_hs = np.append(accel_lp_v_hs, accel_lp.mean())
					# print "accel_lp_v_hs",accel_lp_v_hs

					dynamics_v_hs = np.vstack([dynamics_v_hs, dynamics])

				if "/H/" in exper_type[-16:]:
					# print "H hs"
					x_lp_total_h_hs = np.append(x_lp_total_h_hs, x_lp[-frames_to_consider:-40])
					y_lp_total_h_hs = np.append(y_lp_total_h_hs, y_lp[-frames_to_consider:-40])
					# accel_lp_h_hs = np.append(accel_lp_h_hs, accel_lp.mean())
					# print "accel_lp_h_hs", accel_lp_h_hs
					ax3.plot(x_lp, y_lp, color = 'b')

					dynamics_h_hs = np.vstack([dynamics_h_hs, dynamics])

				if "/VH/" in exper_type[-16:]:
					# print "VH hs"
					x_lp_total_vh_hs = np.append(x_lp_total_vh_hs, x_lp[-frames_to_consider:-40])
					y_lp_total_vh_hs = np.append(y_lp_total_vh_hs, y_lp[-frames_to_consider:-40])
					# accel_lp_vh_hs = np.append(accel_lp_vh_hs, accel_lp.mean())
					# print "accel_lp_vh_hs", accel_lp_vh_hs
					ax5.plot(x_lp, y_lp, color = 'r')

					dynamics_vh_hs = np.vstack([dynamics_vh_hs, dynamics])


			if len(x_lp)>400: # Low speed
				frames_to_consider = 300
				# x_lp=x_lp[-frames_to_consider:]
				# y_lp=y_lp[-frames_to_consider:]

				y_lp_forvelcalc = y_lp[-frames_to_consider:-40]
				lp_time = lp_time[-frames_to_consider:-40] / 1000000000  # to seconds

				vel = np.diff(y_lp_forvelcalc) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				acc = np.diff(vel) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				jerk = np.diff(acc) / np.diff(lp_time)
				lp_time = np.delete(lp_time, 0)
				snap = np.diff(jerk) / np.diff(lp_time)
				dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
				# print "dynamics", dynamics

				if "/V/" in exper_type[-16:]:
					# print "V ls"
					x_lp_total_v_ls = np.append(x_lp_total_v_ls, x_lp[-frames_to_consider:-40])
					y_lp_total_v_ls = np.append(y_lp_total_v_ls, y_lp[-frames_to_consider:-40])
					# accel_lp_v_ls = np.append(accel_lp_v_ls, accel_lp.mean())
					# print "accel_lp_v_ls", accel_lp_v_ls
					ax2.plot(x_lp, y_lp, color = 'g')
					dynamics_v_ls = np.vstack([dynamics_v_ls, dynamics])
				if "/H/" in exper_type[-16:]:
					# print "H ls"
					x_lp_total_h_ls = np.append(x_lp_total_h_ls, x_lp[-frames_to_consider:-40])
					y_lp_total_h_ls = np.append(y_lp_total_h_ls, y_lp[-frames_to_consider:-40])
					# accel_lp_h_ls = np.append(accel_lp_h_ls, accel_lp.mean())
					# print "accel_lp_h_ls", accel_lp_h_ls
					ax4.plot(x_lp, y_lp, color = 'b')
					dynamics_h_ls = np.vstack([dynamics_h_ls, dynamics])
				if "/VH/" in exper_type[-16:]:
					# print "VH ls"
					x_lp_total_vh_ls = np.append(x_lp_total_vh_ls, x_lp[-frames_to_consider:-40])
					y_lp_total_vh_ls = np.append(y_lp_total_vh_ls, y_lp[-frames_to_consider:-40])
					# accel_lp_vh_ls = np.append(accel_lp_vh_ls, accel_lp.mean())
					# print "accel_lp_vh_ls", accel_lp_vh_ls			
					ax6.plot(x_lp, y_lp, color = 'r')
					dynamics_vh_ls = np.vstack([dynamics_vh_ls, dynamics])

print
print 'HIGH speed'
# print 'dynamics_v_hs', dynamics_v_hs
print 'v_hs vel.mean()', dynamics_v_hs[:,0].mean()
print 'v_hs acc.mean()', dynamics_v_hs[:,1].mean()
print 'v_hs jerk.mean()', dynamics_v_hs[:,2].mean()
print 'v_hs snap.mean()', dynamics_v_hs[:,3].mean()
print
print 'h_hs vel.mean()', dynamics_h_hs[:,0].mean()
print 'h_hs acc.mean()', dynamics_h_hs[:,1].mean()
print 'h_hs jerk.mean()', dynamics_h_hs[:,2].mean()
print 'h_hs snap.mean()', dynamics_h_hs[:,3].mean()
print
print 'vh_hs vel.mean()', dynamics_vh_hs[:,0].mean()
print 'vh_hs acc.mean()', dynamics_vh_hs[:,1].mean()
print 'vh_hs jerk.mean()', dynamics_vh_hs[:,2].mean()
print 'vh_hs snap.mean()', dynamics_vh_hs[:,3].mean()
print
print 'LOW speed'
print 'v_ls vel.mean()', dynamics_v_ls[:,0].mean()
print 'v_ls acc.mean()', dynamics_v_ls[:,1].mean()
print 'v_ls jerk.mean()', dynamics_v_ls[:,2].mean()
print 'v_ls snap.mean()', dynamics_v_ls[:,3].mean()
print
print 'h_ls vel.mean()', dynamics_h_ls[:,0].mean()
print 'h_ls acc.mean()', dynamics_h_ls[:,1].mean()
print 'h_ls jerk.mean()', dynamics_h_ls[:,2].mean()
print 'h_ls snap.mean()', dynamics_h_ls[:,3].mean()
print
print 'vh_ls vel.mean()', dynamics_vh_ls[:,0].mean()
print 'vh_ls acc.mean()', dynamics_vh_ls[:,1].mean()
print 'vh_ls jerk.mean()', dynamics_vh_ls[:,2].mean()
print 'vh_ls snap.mean()', dynamics_vh_ls[:,3].mean()

ax1.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_v_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax1.text(-0.245, 0.11, r'average accel: ' + str(dynamics_v_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax1.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_v_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax1.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_v_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax3.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_h_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax3.text(-0.245, 0.11, r'average accel: ' + str(dynamics_h_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax3.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_h_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax3.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_h_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax5.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_vh_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax5.text(-0.245, 0.11, r'average accel: ' + str(dynamics_vh_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax5.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_vh_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax5.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_vh_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)



ax2.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_v_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax2.text(-0.245, 0.11, r'average accel: ' + str(dynamics_v_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax2.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_v_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax2.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_v_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax4.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_h_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax4.text(-0.245, 0.11, r'average accel: ' + str(dynamics_h_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax4.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_h_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax4.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_h_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax6.text(-0.245, 0.13, r'average vel:   ' + str(dynamics_vh_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax6.text(-0.245, 0.11, r'average accel: ' + str(dynamics_vh_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax6.text(-0.245, 0.09, r'average jerk:  ' + str(dynamics_vh_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax6.text(-0.245, 0.07, r'average snap:  ' + str(dynamics_vh_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)



# print "x_cf_before_landing mean", x_cf_before_landing.mean()
# print "y_cf_before_landing mean", y_cf_before_landing.mean()

# print "accel_lp_v_hs", accel_lp_v_hs.mean()
# print "accel_lp_h_hs", accel_lp_h_hs.mean()
# print "accel_lp_vh_hs", accel_lp_vh_hs.mean()
# print "accel_lp_v_ls", accel_lp_v_ls.mean()
# print "accel_lp_h_ls", accel_lp_h_ls.mean()
# print "accel_lp_vh_ls", accel_lp_vh_ls.mean()



ax1.plot(x_lp_total_v_hs.mean(), y_lp_total_v_hs.mean(), 'o', color='c', markersize=20)
ax3.plot(x_lp_total_h_hs.mean(), y_lp_total_h_hs.mean(), 'o', color='c', markersize=20)
ax5.plot(x_lp_total_vh_hs.mean(), y_lp_total_vh_hs.mean(), 'o', color='c', markersize=20)

ax2.plot(x_lp_total_v_ls.mean(), y_lp_total_v_ls.mean(), 'o', color='c', markersize=20)
ax4.plot(x_lp_total_h_ls.mean(), y_lp_total_h_ls.mean(), 'o', color='c', markersize=20)
ax6.plot(x_lp_total_vh_ls.mean(), y_lp_total_vh_ls.mean(), 'o', color='c', markersize=20)

ax1.axis([-0.25, 0.2, -0.15, 0.15])
ax2.axis([-0.25, 0.2, -0.15, 0.15])
ax3.axis([-0.25, 0.2, -0.15, 0.15])
ax4.axis([-0.25, 0.2, -0.15, 0.15])
ax5.axis([-0.25, 0.2, -0.15, 0.15])
ax6.axis([-0.25, 0.2, -0.15, 0.15])

ax1.set_title('Vision feedback. High landing speed')
ax3.set_title('Haptics feedback. High landing speed')
ax5.set_title('Vision and Haptics feedback. High landing speed')

ax2.set_title('Vision feedback. Low landing speed')
ax4.set_title('Haptics feedback. Low landing speed')
ax6.set_title('Vision and Haptics feedback. Low landing speed')

ax1.axhline(y=0.475, color='k')
ax1.axvline(x=0.468, color='k')
ax2.axhline(y=0.475, color='k')
ax2.axvline(x=0.468, color='k')
ax3.axhline(y=0.475, color='k')
ax3.axvline(x=0.468, color='k')
ax4.axhline(y=0.475, color='k')
ax4.axvline(x=0.468, color='k')
ax5.axhline(y=0.475, color='k')
ax5.axvline(x=0.468, color='k')
ax6.axhline(y=0.475, color='k')
ax6.axvline(x=0.468, color='k')

ax1.set_ylabel('Meters, m')
ax5.set_ylabel('Meters, m')
ax5.set_xlabel('Meters, m')
ax6.set_xlabel('Meters, m')



# plt.show()



# 





















































# TRAJECTORIES #######################################################################
print "\nleft hand trajectories............."

x_cf_before_landing = np.array([])
y_cf_before_landing = np.array([])

x_lp_total_v_hs = np.array([])
y_lp_total_v_hs = np.array([])
lp_total_v_hs_time = np.array([])
accel_lp_v_hs = np.array([])

x_lp_total_v_ls = np.array([])
y_lp_total_v_ls = np.array([])
lp_total_v_ls_time = np.array([])
accel_lp_v_ls = np.array([])

x_lp_total_h_hs = np.array([])
y_lp_total_h_hs = np.array([])
lp_total_h_hs_time = np.array([])
accel_lp_h_hs = np.array([])

x_lp_total_h_ls = np.array([])
y_lp_total_h_ls = np.array([])
lp_total_h_ls_time = np.array([])
accel_lp_h_ls = np.array([])

x_lp_total_vh_hs = np.array([])
y_lp_total_vh_hs = np.array([])
lp_total_vh_hs_time = np.array([])
accel_lp_vh_hs = np.array([])

x_lp_total_vh_ls = np.array([])
y_lp_total_vh_ls = np.array([])
lp_total_vh_ls_time = np.array([])
accel_lp_vh_ls = np.array([])


dynamics_v_hs = np.array([0,0,0,0])
dynamics_h_hs = np.array([0,0,0,0])
dynamics_vh_hs = np.array([0,0,0,0])

dynamics_v_ls = np.array([0,0,0,0])
dynamics_h_ls = np.array([0,0,0,0])
dynamics_vh_ls = np.array([0,0,0,0])


f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
plt.suptitle('Landing pad trajectories while landing, in XY plane', fontsize=18)

for name in subject_name_list:
	print "\nName:", name

	directory_v = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/V/experiment/"
	directory_h = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/H/experiment/"
	directory_vh = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/VH/experiment/"

	list_of_experiments = [directory_v, directory_h, directory_vh]

	for exper_type in list_of_experiments:
		# print "Experiment_type", exper_type
		experiment_list = sorted([x[0] for x in os.walk(exper_type)])
		experiment_list.pop(0)
		# print "experiment_list", experiment_list
		xy_error_list = np.array([])
		displacement_list_x = np.array([])
		displacement_list_y = np.array([])
		
		# for drone in ['100','101']:
		for drone in ['100']:

			for experiment_location in experiment_list:
				# print "experiment_location", experiment_location




				x_cf = np.array([]); y_cf = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_crazyflie'+drone+'_slash_crazyflie'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_cf = np.append(x_cf, float(row[10]))
							y_cf = np.append(y_cf, float(row[11]))


				x_lp = np.array([]); y_lp = np.array([])
				lp_time = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_landing_pad'+drone+'_slash_landing_pad'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_lp = np.append(x_lp, float(row[10]))
							y_lp = np.append(y_lp, float(row[11]))
							lp_time = np.append(lp_time, float(row[0]))








				# x_lp_forvelcalc = x_lp[-200:-30]
				# lp_time = lp_time[-200:-30]/1000000000 # to seconds
				#
				# vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# acc = np.diff(vel) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# jerk = np.diff(acc) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# snap = np.diff(jerk) / np.diff(lp_time)
				# dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
				# # print "dynamics", dynamics


				# print "len x_lp", len(x_lp)
				# print "x_cf before landing", x_cf[-50]
				x_cf_before_landing = np.append(x_cf_before_landing, x_cf[-30])
				y_cf_before_landing = np.append(y_cf_before_landing, y_cf[-30])

				if len(x_lp)<400: # High speed
					frames_to_consider = 200
					# x_lp=x_lp[-frames_to_consider:]
					# y_lp=y_lp[-frames_to_consider:]

					x_lp_forvelcalc = x_lp[-frames_to_consider:-30]
					lp_time = lp_time[-frames_to_consider:-30] / 1000000000  # to seconds

					vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					acc = np.diff(vel) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					jerk = np.diff(acc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					snap = np.diff(jerk) / np.diff(lp_time)
					dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
					# print "dynamics", dynamics

					if "/V/" in exper_type[-16:]:
						# print "V hs"
						ax1.plot(x_lp, y_lp, color = 'g')
						x_lp_total_v_hs = np.append(x_lp_total_v_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_v_hs = np.append(y_lp_total_v_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_v_hs = np.append(accel_lp_v_hs, accel_lp.mean())
						# print "accel_lp_v_hs",accel_lp_v_hs

						dynamics_v_hs = np.vstack([dynamics_v_hs, dynamics])

					if "/H/" in exper_type[-16:]:
						# print "H hs"
						x_lp_total_h_hs = np.append(x_lp_total_h_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_h_hs = np.append(y_lp_total_h_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_h_hs = np.append(accel_lp_h_hs, accel_lp.mean())
						# print "accel_lp_h_hs", accel_lp_h_hs
						ax3.plot(x_lp, y_lp, color = 'b')

						dynamics_h_hs = np.vstack([dynamics_h_hs, dynamics])

					if "/VH/" in exper_type[-16:]:
						# print "VH hs"
						x_lp_total_vh_hs = np.append(x_lp_total_vh_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_vh_hs = np.append(y_lp_total_vh_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_vh_hs = np.append(accel_lp_vh_hs, accel_lp.mean())
						# print "accel_lp_vh_hs", accel_lp_vh_hs
						ax5.plot(x_lp, y_lp, color = 'r')

						dynamics_vh_hs = np.vstack([dynamics_vh_hs, dynamics])


				if len(x_lp)>400: # Low speed
					frames_to_consider = 300
					# x_lp=x_lp[-frames_to_consider:]
					# y_lp=y_lp[-frames_to_consider:]

					x_lp_forvelcalc = x_lp[-frames_to_consider:-40]
					lp_time = lp_time[-frames_to_consider:-40] / 1000000000  # to seconds

					vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					acc = np.diff(vel) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					jerk = np.diff(acc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					snap = np.diff(jerk) / np.diff(lp_time)
					dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
					# print "dynamics", dynamics

					if "/V/" in exper_type[-16:]:
						# print "V ls"
						x_lp_total_v_ls = np.append(x_lp_total_v_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_v_ls = np.append(y_lp_total_v_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_v_ls = np.append(accel_lp_v_ls, accel_lp.mean())
						# print "accel_lp_v_ls", accel_lp_v_ls
						ax2.plot(x_lp, y_lp, color = 'g')
						dynamics_v_ls = np.vstack([dynamics_v_ls, dynamics])
					if "/H/" in exper_type[-16:]:
						# print "H ls"
						x_lp_total_h_ls = np.append(x_lp_total_h_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_h_ls = np.append(y_lp_total_h_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_h_ls = np.append(accel_lp_h_ls, accel_lp.mean())
						# print "accel_lp_h_ls", accel_lp_h_ls
						ax4.plot(x_lp, y_lp, color = 'b')
						dynamics_h_ls = np.vstack([dynamics_h_ls, dynamics])
					if "/VH/" in exper_type[-16:]:
						# print "VH ls"
						x_lp_total_vh_ls = np.append(x_lp_total_vh_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_vh_ls = np.append(y_lp_total_vh_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_vh_ls = np.append(accel_lp_vh_ls, accel_lp.mean())
						# print "accel_lp_vh_ls", accel_lp_vh_ls			
						ax6.plot(x_lp, y_lp, color = 'r')
						dynamics_vh_ls = np.vstack([dynamics_vh_ls, dynamics])

print
print 'HIGH speed'
# print 'dynamics_v_hs', dynamics_v_hs
print 'v_hs vel.mean()', dynamics_v_hs[:,0].mean()
print 'v_hs acc.mean()', dynamics_v_hs[:,1].mean()
print 'v_hs jerk.mean()', dynamics_v_hs[:,2].mean()
print 'v_hs snap.mean()', dynamics_v_hs[:,3].mean()
print
print 'h_hs vel.mean()', dynamics_h_hs[:,0].mean()
print 'h_hs acc.mean()', dynamics_h_hs[:,1].mean()
print 'h_hs jerk.mean()', dynamics_h_hs[:,2].mean()
print 'h_hs snap.mean()', dynamics_h_hs[:,3].mean()
print
print 'vh_hs vel.mean()', dynamics_vh_hs[:,0].mean()
print 'vh_hs acc.mean()', dynamics_vh_hs[:,1].mean()
print 'vh_hs jerk.mean()', dynamics_vh_hs[:,2].mean()
print 'vh_hs snap.mean()', dynamics_vh_hs[:,3].mean()
print
print 'LOW speed'
print 'v_ls vel.mean()', dynamics_v_ls[:,0].mean()
print 'v_ls acc.mean()', dynamics_v_ls[:,1].mean()
print 'v_ls jerk.mean()', dynamics_v_ls[:,2].mean()
print 'v_ls snap.mean()', dynamics_v_ls[:,3].mean()
print
print 'h_ls vel.mean()', dynamics_h_ls[:,0].mean()
print 'h_ls acc.mean()', dynamics_h_ls[:,1].mean()
print 'h_ls jerk.mean()', dynamics_h_ls[:,2].mean()
print 'h_ls snap.mean()', dynamics_h_ls[:,3].mean()
print
print 'vh_ls vel.mean()', dynamics_vh_ls[:,0].mean()
print 'vh_ls acc.mean()', dynamics_vh_ls[:,1].mean()
print 'vh_ls jerk.mean()', dynamics_vh_ls[:,2].mean()
print 'vh_ls snap.mean()', dynamics_vh_ls[:,3].mean()

ax1.text(0.21, 0.63, r'average vel:   ' + str(dynamics_v_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax1.text(0.21, 0.61, r'average accel: ' + str(dynamics_v_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax1.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_v_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax1.text(0.21, 0.57, r'average snap:  ' + str(dynamics_v_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax3.text(0.21, 0.63, r'average vel:   ' + str(dynamics_h_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax3.text(0.21, 0.61, r'average accel: ' + str(dynamics_h_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax3.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_h_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax3.text(0.21, 0.57, r'average snap:  ' + str(dynamics_h_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax5.text(0.21, 0.63, r'average vel:   ' + str(dynamics_vh_hs[:,0].mean())[:6]+" m/s", fontsize=12)
ax5.text(0.21, 0.61, r'average accel: ' + str(dynamics_vh_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax5.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_vh_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax5.text(0.21, 0.57, r'average snap:  ' + str(dynamics_vh_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)



ax2.text(0.21, 0.63, r'average vel:   ' + str(dynamics_v_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax2.text(0.21, 0.61, r'average accel: ' + str(dynamics_v_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax2.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_v_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax2.text(0.21, 0.57, r'average snap:  ' + str(dynamics_v_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax4.text(0.21, 0.63, r'average vel:   ' + str(dynamics_h_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax4.text(0.21, 0.61, r'average accel: ' + str(dynamics_h_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax4.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_h_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax4.text(0.21, 0.57, r'average snap:  ' + str(dynamics_h_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

ax6.text(0.21, 0.63, r'average vel:   ' + str(dynamics_vh_ls[:,0].mean())[:6]+" m/s", fontsize=12)
ax6.text(0.21, 0.61, r'average accel: ' + str(dynamics_vh_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
ax6.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_vh_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
ax6.text(0.21, 0.57, r'average snap:  ' + str(dynamics_vh_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)



print "x_cf_before_landing mean", x_cf_before_landing.mean()
print "y_cf_before_landing mean", y_cf_before_landing.mean()
x_cf_before_landing = x_cf_before_landing.mean()
y_cf_before_landing = y_cf_before_landing.mean()
# print "accel_lp_v_hs", accel_lp_v_hs.mean()
# print "accel_lp_h_hs", accel_lp_h_hs.mean()
# print "accel_lp_vh_hs", accel_lp_vh_hs.mean()
# print "accel_lp_v_ls", accel_lp_v_ls.mean()
# print "accel_lp_h_ls", accel_lp_h_ls.mean()
# print "accel_lp_vh_ls", accel_lp_vh_ls.mean()



ax1.plot(x_lp_total_v_hs.mean(), y_lp_total_v_hs.mean(), 'o', color='c', markersize=20)
ax3.plot(x_lp_total_h_hs.mean(), y_lp_total_h_hs.mean(), 'o', color='c', markersize=20)
ax5.plot(x_lp_total_vh_hs.mean(), y_lp_total_vh_hs.mean(), 'o', color='c', markersize=20)

ax2.plot(x_lp_total_v_ls.mean(), y_lp_total_v_ls.mean(), 'o', color='c', markersize=20)
ax4.plot(x_lp_total_h_ls.mean(), y_lp_total_h_ls.mean(), 'o', color='c', markersize=20)
ax6.plot(x_lp_total_vh_ls.mean(), y_lp_total_vh_ls.mean(), 'o', color='c', markersize=20)


b  = np.array([0,0])

# np.linalg.norm(cf_last_pose-lp_last_pose)


print '\nmean drone-LP distance while landing V hs', np.linalg.norm(np.array([x_lp_total_v_hs.mean()-x_cf_before_landing, y_lp_total_v_hs.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing H hs', np.linalg.norm(np.array([x_lp_total_h_hs.mean()-x_cf_before_landing, y_lp_total_h_hs.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing VH hs', np.linalg.norm(np.array([x_lp_total_vh_hs.mean()-x_cf_before_landing, y_lp_total_vh_hs.mean()-y_cf_before_landing]) - b)

print '\nmean drone-LP distance while landing V ls', np.linalg.norm(np.array([x_lp_total_v_ls.mean()-x_cf_before_landing, y_lp_total_v_ls.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing H ls', np.linalg.norm(np.array([x_lp_total_h_ls.mean()-x_cf_before_landing, y_lp_total_h_ls.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing VH ls', np.linalg.norm(np.array([x_lp_total_vh_ls.mean()-x_cf_before_landing, y_lp_total_vh_ls.mean()-y_cf_before_landing]) - b)




ax1.axis([0.2, 0.6, 0.35, 0.65])
ax2.axis([0.2, 0.6, 0.35, 0.65])
ax3.axis([0.2, 0.6, 0.35, 0.65])
ax4.axis([0.2, 0.6, 0.35, 0.65])
ax5.axis([0.2, 0.6, 0.35, 0.65])
ax6.axis([0.2, 0.6, 0.35, 0.65])

ax1.set_title('Vision feedback. High landing speed')
ax3.set_title('Haptics feedback. High landing speed')
ax5.set_title('Vision and Haptics feedback. High landing speed')

ax2.set_title('Vision feedback. Low landing speed')
ax4.set_title('Haptics feedback. Low landing speed')
ax6.set_title('Vision and Haptics feedback. Low landing speed')

ax1.axhline(y=y_cf_before_landing, color='k')
ax1.axvline(x=x_cf_before_landing, color='k')
ax2.axhline(y=y_cf_before_landing, color='k')
ax2.axvline(x=x_cf_before_landing, color='k')
ax3.axhline(y=y_cf_before_landing, color='k')
ax3.axvline(x=x_cf_before_landing, color='k')
ax4.axhline(y=y_cf_before_landing, color='k')
ax4.axvline(x=x_cf_before_landing, color='k')
ax5.axhline(y=y_cf_before_landing, color='k')
ax5.axvline(x=x_cf_before_landing, color='k')
ax6.axhline(y=y_cf_before_landing, color='k')
ax6.axvline(x=x_cf_before_landing, color='k')

ax1.set_ylabel('Meters, m')
ax5.set_ylabel('Meters, m')
ax5.set_xlabel('Meters, m')
ax6.set_xlabel('Meters, m')



# plt.show()


































# TRAJECTORIES #######################################################################
print "\nRight hand trajectories............."

x_cf_before_landing = np.array([])
y_cf_before_landing = np.array([])

x_lp_total_v_hs = np.array([])
y_lp_total_v_hs = np.array([])
lp_total_v_hs_time = np.array([])
accel_lp_v_hs = np.array([])

x_lp_total_v_ls = np.array([])
y_lp_total_v_ls = np.array([])
lp_total_v_ls_time = np.array([])
accel_lp_v_ls = np.array([])

x_lp_total_h_hs = np.array([])
y_lp_total_h_hs = np.array([])
lp_total_h_hs_time = np.array([])
accel_lp_h_hs = np.array([])

x_lp_total_h_ls = np.array([])
y_lp_total_h_ls = np.array([])
lp_total_h_ls_time = np.array([])
accel_lp_h_ls = np.array([])

x_lp_total_vh_hs = np.array([])
y_lp_total_vh_hs = np.array([])
lp_total_vh_hs_time = np.array([])
accel_lp_vh_hs = np.array([])

x_lp_total_vh_ls = np.array([])
y_lp_total_vh_ls = np.array([])
lp_total_vh_ls_time = np.array([])
accel_lp_vh_ls = np.array([])


dynamics_v_hs = np.array([0,0,0,0])
dynamics_h_hs = np.array([0,0,0,0])
dynamics_vh_hs = np.array([0,0,0,0])

dynamics_v_ls = np.array([0,0,0,0])
dynamics_h_ls = np.array([0,0,0,0])
dynamics_vh_ls = np.array([0,0,0,0])


# f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex='col', sharey='row')
# f, ((ax2, ax4, ax6), (ax1, ax3, ax5)) = plt.subplots(2, 3, sharex='col', sharey='row')
f, ((ax2, ax4, ax6)) = plt.subplots(1, 3, sharex='col', sharey='row')
# plt.suptitle('Landing pad trajectories while landing, in XY plane', fontsize=18)

for name in subject_name_list:
	print "\nName:", name

	directory_v = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/V/experiment/"
	directory_h = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/H/experiment/"
	directory_vh = "/home/drone/Desktop/SwarmSkin/experiment2/"+name+"/VH/experiment/"

	list_of_experiments = [directory_v, directory_h, directory_vh]

	for exper_type in list_of_experiments:
		# print "Experiment_type", exper_type
		experiment_list = sorted([x[0] for x in os.walk(exper_type)])
		experiment_list.pop(0)
		# print "experiment_list", experiment_list
		xy_error_list = np.array([])
		displacement_list_x = np.array([])
		displacement_list_y = np.array([])
		
		# for drone in ['100','101']:
		for drone in ['101']:

			for experiment_location in experiment_list:
				# print "experiment_location", experiment_location








				x_cf = np.array([]); y_cf = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_crazyflie'+drone+'_slash_crazyflie'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_cf = np.append(x_cf, float(row[10]))
							y_cf = np.append(y_cf, float(row[11]))


				x_lp = np.array([]); y_lp = np.array([])
				lp_time = np.array([])
				csv_file_name = experiment_location + '/_slash_vicon_slash_landing_pad'+drone+'_slash_landing_pad'+drone+'.csv'
				with open(csv_file_name) as csvfile:
					reader = csv.reader(csvfile)
					for row in reader:
						if row[10] != "x": # skip the first line
							x_lp = np.append(x_lp, float(row[10]))
							y_lp = np.append(y_lp, float(row[11]))
							lp_time = np.append(lp_time, float(row[0]))








				# x_lp_forvelcalc = x_lp[-200:-30]
				# lp_time = lp_time[-200:-30]/1000000000 # to seconds
				#
				# vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# acc = np.diff(vel) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# jerk = np.diff(acc) / np.diff(lp_time)
				# lp_time = np.delete(lp_time, 0)
				# snap = np.diff(jerk) / np.diff(lp_time)
				# dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
				# # print "dynamics", dynamics


				# print "len x_lp", len(x_lp)
				# print "x_cf before landing", x_cf[-50]
				x_cf_before_landing = np.append(x_cf_before_landing, x_cf[-30])
				y_cf_before_landing = np.append(y_cf_before_landing, y_cf[-30])

				if len(x_lp)<400: # High speed
					frames_to_consider = 200
					# x_lp=x_lp[-frames_to_consider:]
					# y_lp=y_lp[-frames_to_consider:]

					x_lp_forvelcalc = x_lp[-frames_to_consider:-30]
					lp_time = lp_time[-frames_to_consider:-30] / 1000000000  # to seconds

					vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					acc = np.diff(vel) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					jerk = np.diff(acc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					snap = np.diff(jerk) / np.diff(lp_time)
					dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
					# print "dynamics", dynamics

					if "/V/" in exper_type[-16:]:
						# print "V hs"
						ax1.plot(x_lp, y_lp, color = 'g')
						x_lp_total_v_hs = np.append(x_lp_total_v_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_v_hs = np.append(y_lp_total_v_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_v_hs = np.append(accel_lp_v_hs, accel_lp.mean())
						# print "accel_lp_v_hs",accel_lp_v_hs

						dynamics_v_hs = np.vstack([dynamics_v_hs, dynamics])

					if "/H/" in exper_type[-16:]:
						# print "H hs"
						x_lp_total_h_hs = np.append(x_lp_total_h_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_h_hs = np.append(y_lp_total_h_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_h_hs = np.append(accel_lp_h_hs, accel_lp.mean())
						# print "accel_lp_h_hs", accel_lp_h_hs
						ax3.plot(x_lp, y_lp, color = 'b')

						dynamics_h_hs = np.vstack([dynamics_h_hs, dynamics])

					if "/VH/" in exper_type[-16:]:
						# print "VH hs"
						x_lp_total_vh_hs = np.append(x_lp_total_vh_hs, x_lp[-frames_to_consider:-40])
						y_lp_total_vh_hs = np.append(y_lp_total_vh_hs, y_lp[-frames_to_consider:-40])
						# accel_lp_vh_hs = np.append(accel_lp_vh_hs, accel_lp.mean())
						# print "accel_lp_vh_hs", accel_lp_vh_hs
						ax5.plot(x_lp, y_lp, color = 'r')

						dynamics_vh_hs = np.vstack([dynamics_vh_hs, dynamics])


				if len(x_lp)>400: # Low speed
					frames_to_consider = 300
					# x_lp=x_lp[-frames_to_consider:]
					# y_lp=y_lp[-frames_to_consider:]

					x_lp_forvelcalc = x_lp[-frames_to_consider:-40]
					lp_time = lp_time[-frames_to_consider:-40] / 1000000000  # to seconds

					vel = np.diff(x_lp_forvelcalc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					acc = np.diff(vel) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					jerk = np.diff(acc) / np.diff(lp_time)
					lp_time = np.delete(lp_time, 0)
					snap = np.diff(jerk) / np.diff(lp_time)
					dynamics = np.array([abs(vel).mean(), abs(acc).mean(), abs(jerk).mean(), abs(snap).mean()])
					# print "dynamics", dynamics

					if "/V/" in exper_type[-16:]:
						# print "V ls"
						x_lp_total_v_ls = np.append(x_lp_total_v_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_v_ls = np.append(y_lp_total_v_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_v_ls = np.append(accel_lp_v_ls, accel_lp.mean())
						# print "accel_lp_v_ls", accel_lp_v_ls
						ax2.plot(x_lp, y_lp, color = 'g')
						dynamics_v_ls = np.vstack([dynamics_v_ls, dynamics])
					if "/H/" in exper_type[-16:]:
						# print "H ls"
						x_lp_total_h_ls = np.append(x_lp_total_h_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_h_ls = np.append(y_lp_total_h_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_h_ls = np.append(accel_lp_h_ls, accel_lp.mean())
						# print "accel_lp_h_ls", accel_lp_h_ls
						ax4.plot(x_lp, y_lp, color = 'b')
						dynamics_h_ls = np.vstack([dynamics_h_ls, dynamics])
					if "/VH/" in exper_type[-16:]:
						# print "VH ls"
						x_lp_total_vh_ls = np.append(x_lp_total_vh_ls, x_lp[-frames_to_consider:-40])
						y_lp_total_vh_ls = np.append(y_lp_total_vh_ls, y_lp[-frames_to_consider:-40])
						# accel_lp_vh_ls = np.append(accel_lp_vh_ls, accel_lp.mean())
						# print "accel_lp_vh_ls", accel_lp_vh_ls			
						ax6.plot(x_lp, y_lp, color = 'r')
						dynamics_vh_ls = np.vstack([dynamics_vh_ls, dynamics])

print
print 'HIGH speed'
# print 'dynamics_v_hs', dynamics_v_hs
print 'v_hs vel.mean()', dynamics_v_hs[:,0].mean()
print 'v_hs acc.mean()', dynamics_v_hs[:,1].mean()
print 'v_hs jerk.mean()', dynamics_v_hs[:,2].mean()
print 'v_hs snap.mean()', dynamics_v_hs[:,3].mean()
print
print 'h_hs vel.mean()', dynamics_h_hs[:,0].mean()
print 'h_hs acc.mean()', dynamics_h_hs[:,1].mean()
print 'h_hs jerk.mean()', dynamics_h_hs[:,2].mean()
print 'h_hs snap.mean()', dynamics_h_hs[:,3].mean()
print
print 'vh_hs vel.mean()', dynamics_vh_hs[:,0].mean()
print 'vh_hs acc.mean()', dynamics_vh_hs[:,1].mean()
print 'vh_hs jerk.mean()', dynamics_vh_hs[:,2].mean()
print 'vh_hs snap.mean()', dynamics_vh_hs[:,3].mean()
print
print 'LOW speed'
print 'v_ls vel.mean()', dynamics_v_ls[:,0].mean()
print 'v_ls acc.mean()', dynamics_v_ls[:,1].mean()
print 'v_ls jerk.mean()', dynamics_v_ls[:,2].mean()
print 'v_ls snap.mean()', dynamics_v_ls[:,3].mean()
print
print 'h_ls vel.mean()', dynamics_h_ls[:,0].mean()
print 'h_ls acc.mean()', dynamics_h_ls[:,1].mean()
print 'h_ls jerk.mean()', dynamics_h_ls[:,2].mean()
print 'h_ls snap.mean()', dynamics_h_ls[:,3].mean()
print
print 'vh_ls vel.mean()', dynamics_vh_ls[:,0].mean()
print 'vh_ls acc.mean()', dynamics_vh_ls[:,1].mean()
print 'vh_ls jerk.mean()', dynamics_vh_ls[:,2].mean()
print 'vh_ls snap.mean()', dynamics_vh_ls[:,3].mean()

# ax1.text(0.21, 0.63, r'average vel:   ' + str(dynamics_v_hs[:,0].mean())[:6]+" m/s", fontsize=12)
# ax1.text(0.21, 0.61, r'average accel: ' + str(dynamics_v_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax1.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_v_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax1.text(0.21, 0.57, r'average snap:  ' + str(dynamics_v_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

# ax3.text(0.21, 0.63, r'average vel:   ' + str(dynamics_h_hs[:,0].mean())[:6]+" m/s", fontsize=12)
# ax3.text(0.21, 0.61, r'average accel: ' + str(dynamics_h_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax3.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_h_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax3.text(0.21, 0.57, r'average snap:  ' + str(dynamics_h_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)

# ax5.text(0.21, 0.63, r'average vel:   ' + str(dynamics_vh_hs[:,0].mean())[:6]+" m/s", fontsize=12)
# ax5.text(0.21, 0.61, r'average accel: ' + str(dynamics_vh_hs[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax5.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_vh_hs[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax5.text(0.21, 0.57, r'average snap:  ' + str(dynamics_vh_hs[:,3].mean())[:4]+" m/s^4", fontsize=12)



# ax2.text(0.21, 0.63, r'average vel:   ' + str(dynamics_v_ls[:,0].mean())[:6]+" m/s", fontsize=12)
# ax2.text(0.21, 0.61, r'average accel: ' + str(dynamics_v_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax2.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_v_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax2.text(0.21, 0.57, r'average snap:  ' + str(dynamics_v_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

# ax4.text(0.21, 0.63, r'average vel:   ' + str(dynamics_h_ls[:,0].mean())[:6]+" m/s", fontsize=12)
# ax4.text(0.21, 0.61, r'average accel: ' + str(dynamics_h_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax4.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_h_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax4.text(0.21, 0.57, r'average snap:  ' + str(dynamics_h_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)

# ax6.text(0.21, 0.63, r'average vel:   ' + str(dynamics_vh_ls[:,0].mean())[:6]+" m/s", fontsize=12)
# ax6.text(0.21, 0.61, r'average accel: ' + str(dynamics_vh_ls[:,1].mean())[:5]+" m/s^2", fontsize=12)
# ax6.text(0.21, 0.59, r'average jerk:  ' + str(dynamics_vh_ls[:,2].mean())[:4]+" m/s^3", fontsize=12)
# ax6.text(0.21, 0.57, r'average snap:  ' + str(dynamics_vh_ls[:,3].mean())[:4]+" m/s^4", fontsize=12)



print "x_cf_before_landing mean", x_cf_before_landing.mean()
print "y_cf_before_landing mean", y_cf_before_landing.mean()
x_cf_before_landing = x_cf_before_landing.mean()
y_cf_before_landing = y_cf_before_landing.mean()
# print "accel_lp_v_hs", accel_lp_v_hs.mean()
# print "accel_lp_h_hs", accel_lp_h_hs.mean()
# print "accel_lp_vh_hs", accel_lp_vh_hs.mean()
# print "accel_lp_v_ls", accel_lp_v_ls.mean()
# print "accel_lp_h_ls", accel_lp_h_ls.mean()
# print "accel_lp_vh_ls", accel_lp_vh_ls.mean()



# ax1.plot(x_lp_total_v_hs.mean(), y_lp_total_v_hs.mean(), 'o', color='c', markersize=20)
# ax3.plot(x_lp_total_h_hs.mean(), y_lp_total_h_hs.mean(), 'o', color='c', markersize=20)
# ax5.plot(x_lp_total_vh_hs.mean(), y_lp_total_vh_hs.mean(), 'o', color='c', markersize=20)

ax2.plot(x_lp_total_v_ls.mean(), y_lp_total_v_ls.mean(), 'o', color='c', markersize=20)
ax4.plot(x_lp_total_h_ls.mean(), y_lp_total_h_ls.mean(), 'o', color='c', markersize=20)
ax6.plot(x_lp_total_vh_ls.mean(), y_lp_total_vh_ls.mean(), 'o', color='c', markersize=20)


b  = np.array([0,0])

# np.linalg.norm(cf_last_pose-lp_last_pose)


print '\nmean drone-LP distance while landing V hs', np.linalg.norm(np.array([x_lp_total_v_hs.mean()-x_cf_before_landing, y_lp_total_v_hs.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing H hs', np.linalg.norm(np.array([x_lp_total_h_hs.mean()-x_cf_before_landing, y_lp_total_h_hs.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing VH hs', np.linalg.norm(np.array([x_lp_total_vh_hs.mean()-x_cf_before_landing, y_lp_total_vh_hs.mean()-y_cf_before_landing]) - b)

print '\nmean drone-LP distance while landing V ls', np.linalg.norm(np.array([x_lp_total_v_ls.mean()-x_cf_before_landing, y_lp_total_v_ls.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing H ls', np.linalg.norm(np.array([x_lp_total_h_ls.mean()-x_cf_before_landing, y_lp_total_h_ls.mean()-y_cf_before_landing]) - b)
print 'mean drone-LP distance while landing VH ls', np.linalg.norm(np.array([x_lp_total_vh_ls.mean()-x_cf_before_landing, y_lp_total_vh_ls.mean()-y_cf_before_landing]) - b)




# ax1.axis([0.05, 0.45, -0.65, -0.3])
ax2.axis([0.15, 0.45, -0.65, -0.35])
# ax3.axis([0.05, 0.45, -0.65, -0.3])
ax4.axis([0.15, 0.45, -0.65, -0.35])
# ax5.axis([0.05, 0.45, -0.65, -0.3])
ax6.axis([0.15, 0.45, -0.65, -0.35])

# ax1.set_title('Vision feedback. High landing speed')
# ax3.set_title('Haptics feedback. High landing speed')
# ax5.set_title('Vision and Haptics feedback. High landing speed')

ax2.set_title('Vision feedback')
ax4.set_title('Haptics feedback')
ax6.set_title('Vision and Haptics feedback')

# ax1.axhline(y=y_cf_before_landing, color='k')
# ax1.axvline(x=x_cf_before_landing, color='k')
ax2.axhline(y=y_cf_before_landing, color='k')
ax2.axvline(x=x_cf_before_landing, color='k')
# ax3.axhline(y=y_cf_before_landing, color='k')
# ax3.axvline(x=x_cf_before_landing, color='k')
ax4.axhline(y=y_cf_before_landing, color='k')
ax4.axvline(x=x_cf_before_landing, color='k')
# ax5.axhline(y=y_cf_before_landing, color='k')
# ax5.axvline(x=x_cf_before_landing, color='k')
ax6.axhline(y=y_cf_before_landing, color='k')
ax6.axvline(x=x_cf_before_landing, color='k')

# ax1.set_ylabel('Meters, m')
# ax5.set_ylabel('Meters, m')
# ax5.set_xlabel('Meters, m')
ax2.set_xlabel('Meters, m')
ax2.set_ylabel('Meters, m')
ax4.set_xlabel('Meters, m')
ax6.set_xlabel('Meters, m')


plt.show()





































# # EXTRACT the DATA

# x_cf = []
# y_cf = []
# z_cf = []

# with open('_slash_vicon_slash_crazyflie100_slash_crazyflie100.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		if row[10] != "x": # skip the first line
# 			x_cf.append(float(row[10]))
# 			y_cf.append(float(row[11]))
# 			z_cf.append(float(row[12]))

# x_lp = []
# y_lp = []
# z_lp = []
# with open('_slash_vicon_slash_landing_pad_slash_landing_pad.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		if row[10] != "x": # skip the first line
# 			x_lp.append(float(row[10]))
# 			y_lp.append(float(row[11]))
# 			z_lp.append(float(row[12]))






# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# cf_last_pose = np.array([x_cf[-1], y_cf[-1]])
# lp_last_pose = np.array([x_lp[-1], y_lp[-1]])
# xy_error = np.linalg.norm(cf_last_pose-lp_last_pose)
# print "xy_error", xy_error

# # plot a line
# x_cf = np.array(x_cf)
# y_cf = np.array(y_cf)
# z_cf = np.array(z_cf)
# x_lp = np.array(x_lp)
# y_lp = np.array(y_lp)
# z_lp = np.array(z_lp)
# ax.plot(x_cf, y_cf, z_cf, label='drone', c='b')
# ax.plot(x_lp, y_lp, z_lp, label='landing pad', c='g')
# ax.legend()

# # plot dots
# # ax.scatter(x_cf, y_cf, z_cf, c='r', marker='o')
# # ax.scatter(x_lp, y_lp, z_lp, c='g', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()


















# plt.hist(xy_error_list_v, bins='auto')  # arguments are passed to np.histogram
# plt.title("Hist xy_error V total")
# plt.show()

# plt.hist(xy_error_list_h, bins='auto')  # arguments are passed to np.histogram
# plt.title("Hist xy_error H total")
# plt.show()

# plt.hist(xy_error_list_vh, bins='auto')  # arguments are passed to np.histogram
# plt.title("Hist xsy_error VH total")
# plt.show()











# euler = []
# with open('_slash_vicon_slash_landing_pad_slash_landing_pad.csv') as csvfile:
# 	reader = csv.reader(csvfile)
# 	for row in reader:
# 		if row[14] != "x": # skip the first line
# 			quat = (	row[14],
# 						row[15],
# 						row[16],
# 						row[17])
# 			print type(tf.transformations.euler_from_quaternion(quat))
# 			euler.append(tf.transformations.euler_from_quaternion(quat))

# print euler