"""droneControl2 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import *
import numpy as np
from simple_pid import PID
from math import sqrt, acos

def get_my_location():
    """Gets the robots translation in the world."""
    position = gps.getValues()
    return position

def getMotorAll(drone):
        frontLeftMotor = robot.getDevice('front left propeller')
        frontRightMotor = robot.getDevice('front right propeller')
        backLeftMotor = robot.getDevice('rear left propeller')
        backRightMotor = robot.getDevice('rear right propeller')
        return [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor]

def motorSpeed(drone, v1, v2, v3, v4):
        [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor] = getMotorAll(robot)
        frontLeftMotor.setVelocity(v1)
        frontRightMotor.setVelocity(v2)
        backLeftMotor.setVelocity(v3)
        backRightMotor.setVelocity(v4)
        return


def initMotors(drone, MAX_PROPSPEED):
        [frontLeftMotor, frontRightMotor, backLeftMotor, backRightMotor] = getMotorAll(robot)
        frontLeftMotor.setPosition(float('inf'))
        frontRightMotor.setPosition(float('inf'))
        backLeftMotor.setPosition(float('inf'))
        backRightMotor.setPosition(float('inf'))
	
        motorSpeed(robot, 0,0,0,0)
        return
       
def distance_to(goal_location):
    my_location = get_my_location()
    dist = np.linalg.norm(np.array(goal_location) - np.array(my_location))
    return dist

def set_velocities(linear_vel, angular_vel):
    """Set linear and angular velocities of the robot.

    Arguments:
    linear_velocity  -- Forward velocity in m/s.
    angular_velocity -- Rotational velocity rad/s. Positive direction is
                        counter-clockwise.
    """
    diff_vel = angular_vel * WHEEL_BASE_RADIUS / WHEEL_RADIUS
    right_vel = (linear_vel + diff_vel) / WHEEL_RADIUS
    left_vel = (linear_vel - diff_vel) / WHEEL_RADIUS
    fastest_wheel = abs(max(right_vel, left_vel))
    if fastest_wheel > MAX_WHEEL_VELOCITY:
        left_vel = left_vel / fastest_wheel * MAX_WHEEL_VELOCITY
        right_vel = right_vel / fastest_wheel * MAX_WHEEL_VELOCITY
    left_motor.setVelocity(left_vel)
    right_motor.setVelocity(right_vel)


def pure_pursuit(x_track_error, angle_error):
    """Follow a line.

    Arguments:
    x_track_error -- Cross-track error. The distance error perpenticular to the
                     reference line. If the robot is located to the left of the
                     line,it gives a positive error; negative to the right.
    angle_error   -- Angle error is the difference between the angle of the line
                     and the heading of the robot. If the reference line points
                     towards North and the robot to East, the error is positive;
                     negative to West.
    """
    ang_vel = -0.05 * x_track_error + 0.3 * angle_error
    set_velocities(TARGET_FORWARD_VELOCITY, ang_vel)

# create the Robot instance.
robot = Supervisor()
TIME_STEP = 8
Mavic2Motors = getMotorAll(robot)
initMotors(robot, 150)
motorSpeed(robot, 160, 160, 160, 160)

GOAL_POINT = [-15.0, 1.0, 0.0]

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

camera = Camera("camera")
camera.enable(TIME_STEP)
camera_roll = Motor("camera roll")
camera_pitch = Motor("camera pitch")
#camera_yaw = Camera("camera yaw")
gps = GPS("gps")
gps.enable(TIME_STEP)
imu = InertialUnit("inertial unit")
imu.enable(TIME_STEP)
compass = Compass("compass")
compass.enable(TIME_STEP)
gyro = Gyro("gyro")
gyro.enable(TIME_STEP)

#PD constants:
k_vertical_thrust = 68.5
k_vert_offset = 0.6
k_vertical_p = 3.0
k_roll_p = 50.0
k_pitch_p = 30.0

target_altitude = 1.0
targetX = 0.0
targetY = 0.0


while robot.step(timestep) != -1:

    initial_location = get_my_location()
    #print(initial_location)
    m_line = np.array(GOAL_POINT) - np.array(initial_location)
    print(m_line)
    
    #Retrieve Robot position using IMU and GPS:
    roll = imu.getRollPitchYaw()[0] + np.pi/2.0
    pitch = imu.getRollPitchYaw()[1]
    yaw = compass.getValues()[0]
    #print("Roll: ", roll, "Pitch: ", pitch, "Yaw: ", yaw)
    LocationX = gps.getValues()[0] 
    LocationY = gps.getValues()[1] 
    LocationZ= gps.getValues()[2] 
    #print("LocationX: ", LocationX, "LocationY: ", LocationY, "LocationZ: ", LocationZ)
    roll_acceleration = gyro.getValues()[0]
    pitch_acceleration = gyro.getValues()[1]

    roll_disturbance = 0.0
    pitch_disturbance = 0.0
    yaw_disturbance =  0.0  


#CameraStabilisation:
    camera_roll.setPosition(-0.115*roll_acceleration)
    camera_pitch.setPosition(-0.1*pitch_acceleration)

    roll_input = k_roll_p * roll + roll_acceleration + roll_disturbance
    pitch_input = k_pitch_p * pitch - pitch_acceleration + pitch_disturbance
    yaw_input = yaw_disturbance
    difference_altitude = target_altitude - LocationY + k_vert_offset
    vertical_input = k_vertical_p * pow(difference_altitude, 3.0)
    
    front_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input;
    front_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input;
    rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input;
    rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input;
    
    motorSpeed(robot, front_left_motor_input, -front_right_motor_input, -rear_left_motor_input, rear_right_motor_input)
    
    
    pass

# Enter here exit cleanup code.
