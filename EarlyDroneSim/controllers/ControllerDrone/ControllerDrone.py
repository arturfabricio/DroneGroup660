from controller import *
from controller import Supervisor, Node, Field, Motor
import numpy as np
from simple_pid import PID
from math import sqrt, acos

GOAL_POINT = [-15.0, 1.0, 0.0]

TIME_STEP = 8
PROPPELLER_RADIUS = 0.013
TARGET_FORWARD_VELOCITY = 1.0 #M/S
MAX_PROPELLER_VELOCITY = 160 #Max is 576

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

#Finding Propellers and set them to velocity control:
frontLeftMotor = robot.getDevice('front left propeller')
frontRightMotor = robot.getDevice('front right propeller')
backLeftMotor = robot.getDevice('rear left propeller')
backRightMotor = robot.getDevice('rear right propeller')

if frontLeftMotor is None or frontRightMotor is None or backLeftMotor is None or backRightMotor is None:
    print("No propellers found.")
    exit()

frontLeftMotor.setPosition(float('inf'))
frontRightMotor.setPosition(float('inf'))
backLeftMotor.setPosition(float('inf'))
backRightMotor.setPosition(float('inf'))
frontLeftMotor.setVelocity(0.0)
frontRightMotor.setVelocity(0.0)
backLeftMotor.setVelocity(0.0)
backRightMotor.setVelocity(0.0)

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

def get_my_location(drone):
    """Gets the robots translation in the world."""
    #position = robot.getSelf()
    position = gps.getValues()
    return position

def main():
    """Main function, runs when program is started"""
    while True:
        initial_location = get_my_location(robot)
        print(initial_location)
        while robot.step(timestep) != 1:
            pass
        #while robot.step(timestep) != -1:
            #print(Hello)
            #initial_location = get_my_location()
            #print(initial_location)

if __name__ == "__main__":
    main()