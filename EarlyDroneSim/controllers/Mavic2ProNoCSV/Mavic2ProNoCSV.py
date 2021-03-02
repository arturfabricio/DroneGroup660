from controller import *
import mavic2proHelper
from simple_pid import PID
import csv
import struct
import numpy as np



#floor_width = 50
#floor_height = 50

#PID-Controls:
k_vertical_thrust = 70
k_roll_p = 10
wp_tolerance = 0.05 #Disparity between simulation position and code position.
k_pitch_p = 10

target_altitude = 2 #Change the altitude of the drone.
pitch_Kp = 2
pitch_Ki = 0
pitch_Kd = 2

roll_Kp = 2
roll_Ki = 0.1
roll_Kd = 2

throttle_Kp = 10
throttle_Ki = 0.1
throttle_Kd = 5

yaw_Kp = 2
yaw_Ki = 0
yaw_Kd = 2
yaw_setpoint = -0.99
#altitude_attainmment = 0.0

#params = dict()
#with open("../params.csv", "r") as f:
#	lines = csv.reader(f)
#	for line in lines:
#		params[line[0]] = line[1]


GOAL_POINT = [5.0, 0.0, 1.0]

TIME_STEP = 8
TAKEOFF_THRESHOLD_VELOCITY = 160
M_PI = 3.1415926535897932384626433

robot = Supervisor()

[frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor] = mavic2proHelper.getMotorAll(robot)

timestep = int(robot.getBasicTimeStep())
mavic2proMotors = mavic2proHelper.getMotorAll(robot)
mavic2proHelper.initialiseMotors(robot, 0)
mavic2proHelper.motorsSpeed(robot, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY, TAKEOFF_THRESHOLD_VELOCITY)

#def takeoff(drone):



def get_my_location():
    """Gets the robots translation in the world."""
    #position = robot.getSelf()
    position = gps.getValues()
    return position
#front_left_led = LED("front left led")
#front_right_led = LED("front right led")
camera = Camera("camera")
camera.enable(TIME_STEP)
gps = GPS("gps")
gps.enable(TIME_STEP)
imu = InertialUnit("inertial unit")
imu.enable(TIME_STEP)
compass = Compass("compass")
compass.enable(TIME_STEP)
gyro = Gyro("gyro")
gyro.enable(TIME_STEP)

yaw_setpoint=-1

pitchPID = PID(float(pitch_Kp), float(pitch_Ki), float(pitch_Kd), setpoint=0.0)
rollPID = PID(float(roll_Kp), float(roll_Ki), float(roll_Kd), setpoint=0.0)
throttlePID = PID(float(throttle_Kp), float(throttle_Ki), float(throttle_Kd), setpoint=1.0)
yawPID = PID(float(yaw_Kp), float(yaw_Ki), float(yaw_Kd), setpoint=float(yaw_setpoint))

targetX, targetY, target_altitude = 0.0, 0.0, 1.0

while (robot.step(timestep) != -1):
    pitchDisturbance = 0.5
    initial_location = get_my_location()
    print("Initial location: ", initial_location)
    m_line = np.array(GOAL_POINT) - np.array(initial_location)
    #print(m_line)
	#led_state = int(robot.getTime()) % 2
	#front_left_led.set(led_state)
	#front_right_led.set(int(not(led_state)))
    roll = imu.getRollPitchYaw()[0] + M_PI / 2.0
    pitch = imu.getRollPitchYaw()[1]
    yaw = compass.getValues()[0]
    roll_acceleration = gyro.getValues()[0]
    pitch_acceleration = gyro.getValues()[1]
	
    xGPS = gps.getValues()[2]
    yGPS = gps.getValues()[0]
    zGPS = gps.getValues()[1]

    vertical_input = throttlePID(zGPS)
    yaw_input = yawPID(yaw)

    rollPID.setpoint = targetX
    pitchPID.setpoint = targetY
	
    roll_input = float(k_roll_p) * roll + roll_acceleration + rollPID(xGPS)
    pitch_input = float(k_pitch_p) * pitch - pitch_acceleration + pitchDisturbance
    front_left_motor_input = float(k_vertical_thrust) + vertical_input - roll_input - pitch_input + yaw_input
    front_right_motor_input = float(k_vertical_thrust) + vertical_input + roll_input - pitch_input - yaw_input
    rear_left_motor_input = float(k_vertical_thrust) + vertical_input - roll_input + pitch_input - yaw_input
    rear_right_motor_input = float(k_vertical_thrust) + vertical_input + roll_input + pitch_input + yaw_input

    mavic2proHelper.motorsSpeed(robot, front_left_motor_input, -front_right_motor_input, -rear_left_motor_input, rear_right_motor_input)
    #front_left_motor_input, -front_right_motor_input, -rear_left_motor_input, rear_right_motor_input
