# SwarmCloak

A system for landing of a fleet of nano quadrotors on the human arms using light-sensitive landing pads
with vibrotactile feedback. The package contatins a software for drones control and tools to analyse the flight data.
Please, have a look at the [video](https://www.youtube.com/watch?v=2a4XrG_u3RE) for more information.
You can also access the project description at [SIGGRAPH Asia 2019 webpage](https://sa2019.siggraph.org/attend/emerging-technologies/session_slot/231).

<img src="https://github.com/RuslanAgishev/swarmskin/blob/master/figures/swarmcloak.jpg" width=600 />

## Setup
The flight software from this repository requires external positioning system for drones to work.
It could be a motion capture system, for example Vicon, or HTC-Vive/SteamVR lighthouse tracking system.

### Vicon motion capture
In case if you use a motion capture system, install a ROS driver for crazyflie,
[crazyflie_ros](https://github.com/whoenig/crazyflie_ros), follow the instructions bellow.

```
cd ~/catkin_ws/src
git clone https://github.com/whoenig/crazyflie_ros.git
cd crazyflie_ros
git submodule init
git submodule update
```
Use ```catkin_make``` on your workspace to compile.
```
cd ~/catkin_ws/
catkin_make
```

Clone the repository.
```bash
cd ~/Desktop/
git clone https://github.com/RuslanAgishev/swarmskin
```

Adjust drones URIs you use in the launch file [here](https://github.com/RuslanAgishev/swarmskin/blob/master/vicon_scripts/launch/connect1234.launch#L3).
Specify drones landing positions in the [swarmskin.py](https://github.com/RuslanAgishev/swarmskin/blob/master/vicon_scripts/python/swarmskin.py#L106).
And use our repository alongside with crazyflie_ros driver.
```
cp ~/Desktop/swarmskin/vicon_scripts/python/* ~/catkin_ws/src/crazyflie_ros/crazyflie_demo/scripts/
cp ~/Desktop/swarmskin/vicon_scripts/launch/connect1234.launch ~/catkin_ws/src/crazyflie_ros/crazyflie_demo/launch
```
Note, that in order for this setup to work, you also need to track landing pads positions with a motion capture system.
We created a separate object for each landing pad in [Vicon Tracker](https://www.vicon.com/software/tracker/) software and named them "lp1", "lp2", "lp3" and "lp4". You can specify your laning pads objects [here](https://github.com/RuslanAgishev/swarmskin/blob/master/vicon_scripts/python/swarmskin.py#L69).

Connect to crazyflies:
```
roslaunch crazyflie_demo connect1234.launch
```
And run the flight node:
```
rosrun crazyflie_demo swarmskin.py
```

### Lighthouse tracking system
Otherwise, if you are going to fly with the lighthouse tracking system, please refer to the
[documentation](https://wiki.bitcraze.io/doc:lighthouse:setup)
from Bitcraze on how to setup the positioning system.

It is also possible to track distance from a drone to a landing pad with the help of Z-ranger sensor.
However, it is important, when using a lighthouse and optical flow with Z-ranger decks simultaneously,
to specify in the [crazyflie_firmware](https://github.com/bitcraze/crazyflie-firmware) that the flow deck with Z-ranger is not used for a drone localization.
Perform the following two steps to do this.
1.  Disable optical flow:
    this [line](https://github.com/bitcraze/crazyflie-firmware/blob/master/src/deck/drivers/src/flowdeck_v1v2.c#L70)
    in firmware should be set as ```true```:
    ```static bool useFlowDisabled = true;```
2.  Disable Z-ranger (version 2):
    comment this [line](https://github.com/bitcraze/crazyflie-firmware/blob/05315e2ba4b77098b9e05e5f9ad5b48566d658ad/src/deck/drivers/src/zranger2.c#L142):
    ```// rangeEnqueueDownRangeInEstimator(distance, stdDev, xTaskGetTickCount()); ```

In case these links are out-dated, you can download the custom modified [firmware](https://drive.google.com/file/d/1nC26jyhbdd_0MYyDwPrm6ZsK9FyMb7pb/view?usp=sharing).

Ones everything is prepared, attach optical flow and lighthouse decks to your drones and perorm the flight:
```
python lighthouse_demo/swarmskin_lighthouse.py
```

## Citation
If you find this package useful, feel free to cite the work, [arxiv](https://arxiv.org/pdf/1911.09874.pdf).
```
@article{tsykunov2019swarmcloak,
  title={SwarmCloak: Landing of a Swarm of Nano-Quadrotors on Human Arms},
  author={Tsykunov, Evgeny and Agishev, Ruslan and Ibrahimov, Roman and Labazanova, Luiza and Moriyama, Taha and Kajimoto, Hiroyuki and Tsetserukou, Dzmitry},
  journal={arXiv preprint arXiv:1911.09874},
  year={2019}
}
```
