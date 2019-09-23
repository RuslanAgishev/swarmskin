# -*- coding: utf-8 -*-

"""
Simple example of a swarm using the High level commander.

This example is intended to work with any absolute positioning system.
It aims at documenting how to use the High Level Commander together with
the Swarm class.
"""
import time
import numpy as np
import threading

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.syncLogger import SyncLogger


LAND_TIME = 15.0 # sec
TAKEOFF_HEIGHT = 1.0 # m

class Drone:
    def __init__(self, scf):
        self.scf = scf
        self.pose = None
        self.pose_home = None
        self.start_position_reading()
        self.start_battery_status_reading()
        self.start_zranger_reading()

    def position_callback(self, timestamp, data, logconf):
        x = data['kalman.stateX']
        y = data['kalman.stateY']
        z = data['kalman.stateZ']
        self.pose = np.array([x, y, z])
    def start_position_reading(self):
        log_conf = LogConfig(name='Position', period_in_ms=100) # read position with 10 Hz rate
        log_conf.add_variable('kalman.stateX', 'float')
        log_conf.add_variable('kalman.stateY', 'float')
        log_conf.add_variable('kalman.stateZ', 'float')
        self.scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self.position_callback)
        log_conf.start()

    def zranger_callback(self, timestamp, data, logconf):
        self.zranger_m = data['range.zrange'] / 1000. # [m]
        # print('Z-ranger data: %.2f [m]' %self.zranger_m)
    def start_zranger_reading(self):
        log_conf = LogConfig(name='Z-ranger', period_in_ms=100) # read position with 10 Hz rate
        log_conf.add_variable('range.zrange', 'float')
        self.scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self.zranger_callback)
        log_conf.start()

    def battery_callback(self, timestamp, data, logconf):
        self.V_bat = data['pm.vbat']
        # print('Battery status: %.2f [V]' %self.V_bat)
    def start_battery_status_reading(self):
        log_conf = LogConfig(name='Battery', period_in_ms=100) # read battery status with 10 Hz rate
        log_conf.add_variable('pm.vbat', 'float')
        self.scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self.battery_callback)
        log_conf.start()


def wait_for_position_estimator(scf):
    print('Waiting for estimator to find position...')

    log_config = LogConfig(name='Kalman Variance', period_in_ms=500)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_y_history = [1000] * 10
    var_x_history = [1000] * 10
    var_z_history = [1000] * 10

    threshold = 0.001

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_x_history.append(data['kalman.varPX'])
            var_x_history.pop(0)
            var_y_history.append(data['kalman.varPY'])
            var_y_history.pop(0)
            var_z_history.append(data['kalman.varPZ'])
            var_z_history.pop(0)

            min_x = min(var_x_history)
            max_x = max(var_x_history)
            min_y = min(var_y_history)
            max_y = max(var_y_history)
            min_z = min(var_z_history)
            max_z = max(var_z_history)

            print("{} {} {}".
                  format(max_x - min_x, max_y - min_y, max_z - min_z))

            if (max_x - min_x) < threshold and (
                    max_y - min_y) < threshold and (
                    max_z - min_z) < threshold:
                break
def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    wait_for_position_estimator(scf)
def activate_high_level_commander(scf):
    scf.cf.param.set_value('commander.enHighLevel', '1')
def activate_mellinger_controller(scf, use_mellinger):
    controller = 1
    if use_mellinger:
        controller = 2
    scf.cf.param.set_value('stabilizer.controller', controller)


def run_shared_sequence(scf, drone):
    def land_detector(drone, switch_off_height=0.1):
        while True:
            try:
                if drone.zranger_m < switch_off_height:
                    for _ in range(3): drone.commander.stop()
                    time.sleep(0.1)
                    print('Shutdown the motors')
                    break
            except:
                pass

    activate_mellinger_controller(scf, False)
    drone.commander = scf.cf.high_level_commander

    drone.commander.takeoff(TAKEOFF_HEIGHT, 3.0)
    time.sleep(3)

    flight_time = 5
    goal = drone.waypoint
    print('Going to human position', goal)
    drone.commander.go_to(goal[0], goal[1], goal[2], goal[3]/180*np.pi, flight_time, relative=False)
    # time.sleep(flight_time)

    drone.commander.land(0.0, LAND_TIME)
    land_detector(drone)


# r = 0.3; theta1 = pi/6; theta2 = pi/6
# l = 0.245 # distance between drones (arm length)
# width = 0.55 # between person's shoulders
# human_pose = np.array([-1.0, 0.0]); hx = human_pose[0]; hy = human_pose[1]
# goto_arr = [ [hx+ (r+l)*cos(theta1), hy+ (r+l)*sin(theta1) +width/2, 1.8, 0.0],
#              [hx+ r*cos(theta1),     hy+ r*sin(theta1) +width/2, 1.8, 0.0],
#              [hx+ r*cos(theta2),     hy- r*sin(theta2) -width/2, 1.8, 0.0],
#              [hx+ (r+l)*cos(theta2), hy- (r+l)*sin(theta2) -width/2, 1.8, 0.0] ]

# x[m], y[m], z[m], yaw[deg]
goto_arr = [
            [-0.83, 0.70, TAKEOFF_HEIGHT, 0.0],
            [-0.97, 0.52, TAKEOFF_HEIGHT, 0.0],
            [-0.98, -0.50, TAKEOFF_HEIGHT, 0.0],
            [-0.85, -0.70, TAKEOFF_HEIGHT, 0.0]
            ]

waypoints = [
    goto_arr[0],
    goto_arr[1],
    goto_arr[2],
    goto_arr[3],
]

URI1 = 'radio://0/80/2M/E7E7E7E701'
URI2 = 'radio://0/80/2M/E7E7E7E702'
URI3 = 'radio://0/80/2M/E7E7E7E703'
URI4 = 'radio://0/80/2M/E7E7E7E704'
uris = {
    URI1,
    URI2,
    URI3,
    URI4,
}


if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(activate_high_level_commander)
        swarm.parallel_safe(reset_estimator)

        drones = [None for _ in range(len(uris))]
        i = 0
        for uri, scf in swarm._cfs.items():
            drones[i] = Drone(scf)
            time.sleep(1.0)
            drones[i].pose_home = drones[i].pose
            drones[i].waypoint = waypoints[i]
            i+=1

        wp_args = {
            URI1: [drones[0]],
            URI2: [drones[1]],
            URI3: [drones[2]],
            URI4: [drones[3]],
        }

        input('Press Enter to fly')
        swarm.parallel_safe(run_shared_sequence, args_dict=wp_args)

