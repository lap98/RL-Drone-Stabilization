import os
import time
import datetime
import numpy as np
import tensorflow as tf
from lib.plotters import Plotter
from lib.customEnvironment_v0_8 import DroneEnvironment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import TimeLimit
import airsim   
import PySimpleGUI as sg      
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


'''
Given the test foldername as parameter it returns all the data inside the folder
'''
def get_data(foldername):
    testing_data_path = os.getcwd() + "/testing_data"
    for test_path in os.listdir(testing_data_path):
        if(test_path == foldername):
            print(testing_data_path + '/' + test_path)
            for file in os.listdir(testing_data_path + '/' + test_path +'/states'):
                data = np.load(testing_data_path + '/' + test_path +'/states/' + file)
                return data


'''
it returns the time sequence of a particular measurement data depending on the index:
0 -> delta_z
1 -> delta_x
2 -> delta_y
3 -> orient_w_val
4 -> orient_x_val
5 -> orient_y_val
6 -> orient_z_val
7 -> ang_acc_x_val
8 -> ang_acc_y_val
9 -> ang_acc_z_val
10 -> ang_vel_x_val
11 -> ang_vel_y_val
12 -> ang_vel_z_val
13 -> lin_acc_x_val
14 -> lin_acc_y_val
15 -> lin_acc_z_val
16 -> lin_vel_x_val
17 -> lin_vel_y_val
18 -> lin_vel_z_val
'''
def get_sequence_data(data, index):
    seq = []
    for el in data:
        if(index==0):
            seq.append(el[index]*-10)
        elif(index == 3 or index==4 or index ==5 or index ==6):
            seq.append(el[index])
        else:
            seq.append(el[index]*10)
    return seq

'''
Return yaw pitch and roll from data
'''
def get_yaw_pitch_roll(data):
    orient_w = get_sequence_data(data,3)
    orient_x = get_sequence_data(data,4)
    orient_y = get_sequence_data(data,5)
    orient_z = get_sequence_data(data,6)
    yaw_seq = []
    pitch_seq = []
    roll_seq = []
    for i in range(len(orient_w)):
        rot = Rotation.from_quat([orient_x[i],orient_y[i],orient_z[i],orient_w[i]])
        rot_euler = rot.as_euler('xyz', degrees=True)
        roll_seq.append(rot_euler[0])
        pitch_seq.append(rot_euler[1])
        yaw_seq.append(rot_euler[2])
    return yaw_seq, pitch_seq, roll_seq


        
def plotXYZ(data, title = ''):
    plt.title(title)
    x_val_seq = get_sequence_data(data,1)
    y_val_seq = get_sequence_data(data,2)
    z_val_seq = get_sequence_data(data,0)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    plt.xlabel('Timestep')
    plt.ylabel('Delta(m)')
    plt.grid()
    plt.plot(xpoints, label = "Delta X")
    plt.plot(ypoints, label = "Delta Y")
    plt.plot(zpoints, label = "Delta Z")
    plt.legend(loc="upper left")
    plt.show()


def plotYawPitchRoll(data,title = ''):
    plt.title(title)
    yaw,pitch,roll = get_yaw_pitch_roll(data)
    yaw_points = np.array(yaw)
    pitch_points = np.array(pitch)
    roll_points = np.array(roll)
    plt.xlabel('Timestep')
    plt.ylabel('Euler angle')
    plt.grid()
    plt.plot(yaw_points, label = "Yaw")
    plt.plot(pitch_points, label = "Pitch")
    plt.plot(roll_points, label = "Roll")
    plt.legend(loc="upper left")
    plt.show()

def plotAngularAcceleration(data,title=''):
    plt.title(title)
    x_val_seq = get_sequence_data(data,7)
    y_val_seq = get_sequence_data(data,8)
    z_val_seq = get_sequence_data(data,9)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    plt.xlabel('Timestep')
    plt.ylabel('Angular acceleration rad/s2')
    plt.grid()
    plt.plot(xpoints, label = "Angular acceleration X")
    plt.plot(ypoints, label = "Angular acceleration Y")
    plt.plot(zpoints, label = "Angular acceleration Z")
    plt.legend(loc="upper left")
    plt.show()


def plotAngularVelocity(data,title=''):
    plt.title(title)
    x_val_seq = get_sequence_data(data,10)
    y_val_seq = get_sequence_data(data,11)
    z_val_seq = get_sequence_data(data,12)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    plt.xlabel('Timestep')
    plt.ylabel('Angular velocity rad/s')
    plt.grid()
    plt.plot(xpoints, label = "Angular velocity X")
    plt.plot(ypoints, label = "Angular velocity Y")
    plt.plot(zpoints, label = "Angular velocity Z")
    plt.legend(loc="upper left")
    plt.show()

def plotLinearAcceleration(data,title=''):
    plt.title(title)
    x_val_seq = get_sequence_data(data,13)
    y_val_seq = get_sequence_data(data,14)
    z_val_seq = get_sequence_data(data,15)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    plt.xlabel('Timestep')
    plt.ylabel('Linear acceleration m/s2')
    plt.grid()
    plt.plot(xpoints, label = "Linear acceleration X")
    plt.plot(ypoints, label = "Linear acceleration Y")
    plt.plot(zpoints, label = "Linear acceleration Z")
    plt.legend(loc="upper left")
    plt.show()

def plotLinearVelocity(data,title=''):
    plt.title(title)
    x_val_seq = get_sequence_data(data,16)
    y_val_seq = get_sequence_data(data,17)
    z_val_seq = get_sequence_data(data,18)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    plt.xlabel('Timestep')
    plt.ylabel('Linear acceleration m/s')
    plt.grid()
    plt.plot(xpoints, label = "Linear velocity X")
    plt.plot(ypoints, label = "Linear velocity Y")
    plt.plot(zpoints, label = "Linear velocity Z")
    plt.legend(loc="upper left")
    plt.show()    

def plot3D(data,title):
    ax = plt.axes(projection='3d')
    plt.title("TRAJECTORY "+ title)
    x_val_seq = get_sequence_data(data,1)
    y_val_seq = get_sequence_data(data,2)
    z_val_seq = get_sequence_data(data,0)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)
    ax.scatter3D(xpoints[0],  ypoints[0], zpoints[0], cmap='green')
    ax.scatter3D(xpoints[999],  ypoints[999], zpoints[999], cmap='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot3D(xpoints, ypoints, zpoints, 'green')
    plt.show()

def subplot(data,title='',save = False):
    figure = plt.figure(figsize=(18, 18))
    figure.tight_layout(pad=5.0)
    figure.suptitle(title)

    x_val_seq = get_sequence_data(data,1)
    y_val_seq = get_sequence_data(data,2)
    z_val_seq = get_sequence_data(data,0)
    xpoints = np.array(x_val_seq)
    ypoints = np.array(y_val_seq)
    zpoints = np.array(z_val_seq)

    yaw,pitch,roll = get_yaw_pitch_roll(data)
    yaw_points = np.array(yaw)
    pitch_points = np.array(pitch)
    roll_points = np.array(roll)

    ang_acc_x_val_seq = get_sequence_data(data,7)
    ang_acc_y_val_seq = get_sequence_data(data,8)
    ang_acc_z_val_seq = get_sequence_data(data,9)
    ang_acc_xpoints = np.array(ang_acc_x_val_seq)
    ang_acc_ypoints = np.array(ang_acc_y_val_seq)
    ang_acc_zpoints = np.array(ang_acc_z_val_seq)

    ang_vel_x_val_seq = get_sequence_data(data,10)
    ang_vel_y_val_seq = get_sequence_data(data,11)
    ang_vel_z_val_seq = get_sequence_data(data,12)
    ang_vel_xpoints = np.array(ang_vel_x_val_seq)
    ang_vel_ypoints = np.array(ang_vel_y_val_seq)
    ang_vel_zpoints = np.array(ang_vel_z_val_seq)

    lin_acc_x_val_seq = get_sequence_data(data,13)
    lin_acc_y_val_seq = get_sequence_data(data,14)
    lin_acc_z_val_seq = get_sequence_data(data,15)
    lin_acc_xpoints = np.array(lin_acc_x_val_seq)
    lin_acc_ypoints = np.array(lin_acc_y_val_seq)
    lin_acc_zpoints = np.array(lin_acc_z_val_seq)

    lin_vel_x_val_seq = get_sequence_data(data,16)
    lin_vel_y_val_seq = get_sequence_data(data,17)
    lin_vel_z_val_seq = get_sequence_data(data,18)
    lin_vel_xpoints = np.array(lin_vel_x_val_seq)
    lin_vel_ypoints = np.array(lin_vel_y_val_seq)
    lin_vel_zpoints = np.array(lin_vel_z_val_seq)

    ax = plt.subplot(332,projection='3d')
    plt.title("Trajectory")
    ax.scatter3D(xpoints[0],  ypoints[0], zpoints[0], cmap='green',label = "Start point")
    ax.scatter3D(xpoints[999],  ypoints[999], zpoints[999], cmap='red',label = "End point")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot3D(xpoints, ypoints, zpoints, 'green',label = "Trajectory")
    plt.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))

    plt.subplot(334)
    plt.title("Distance from the initial position")
    plt.ylabel('Delta[m]')
    plt.grid()
    plt.plot(xpoints, label = "Delta X")
    plt.plot(ypoints, label = "Delta Y")
    plt.plot(zpoints, label = "Delta Z")
    plt.legend(loc="best")

    plt.subplot(335)
    plt.title("Orientation")
    plt.ylabel('Angle[°]')
    plt.grid()
    plt.plot(yaw_points, label = "Yaw")
    plt.plot(pitch_points, label = "Pitch")
    plt.plot(roll_points, label = "Roll")
    plt.legend(loc="best")

    plt.subplot(336)
    plt.title("Angular acceleration")
    plt.ylabel('Angular acceleration [rad/s2]')
    plt.grid()
    plt.plot(ang_acc_xpoints, label = "Angular acceleration X")
    plt.plot(ang_acc_ypoints, label = "Angular acceleration Y")
    plt.plot(ang_acc_zpoints, label = "Angular acceleration Z")
    plt.legend(loc="best")

    plt.subplot(337)
    plt.title("Angular velocity")
    plt.ylabel('Angular velocity [rad/s]')
    plt.grid()
    plt.plot(ang_vel_xpoints, label = "Angular velocity X")
    plt.plot(ang_vel_ypoints, label = "Angular velocity Y")
    plt.plot(ang_vel_zpoints, label = "Angular velocity Z")
    plt.legend(loc="best")

    plt.subplot(338)
    plt.title("Linear acceleration")
    plt.ylabel('Linear acceleration [m/s2]')
    plt.grid()
    plt.plot(lin_acc_xpoints, label = "Linear acceleration X")
    plt.plot(lin_acc_ypoints, label = "Linear acceleration Y")
    plt.plot(lin_acc_zpoints, label = "Linear acceleration Z")
    plt.legend(loc="best")

    plt.subplot(339)
    plt.title("Linear velocity")
    plt.ylabel('Linear velocity [m/s]')
    plt.grid()
    plt.plot(lin_vel_xpoints, label = "Linear velocity X")
    plt.plot(lin_vel_ypoints, label = "Linear velocity Y")
    plt.plot(lin_vel_zpoints, label = "Linear velocity Z")
    plt.legend(loc="best")

    if(save):
        directory = os.getcwd() + "/plots"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + '/'+ title)
    else:
        plt.show()


    
    
"""
Return the title of the window given as imput the folder name
"""
def get_title(foldername):
    title = foldername
    if("orientation" in title):
        title = title + '°'
    elif("pose" in title):
        title = title + ' m'
    elif("wind" in title):
        if(title != "no_wind"):
            title = title + ' mps'
    title = title.replace('_', ' ').replace('=', ' = ')
    title = title.replace('orientation', 'initial').replace('pose', 'initial value of')
    title = title.capitalize()

    return title

# directory = 'p'
# data = get_data(directory)
# subplot(data,title=get_title(directory),save=False)

test_directory = os.getcwd() + '/testing_data'
for directory in os.listdir(test_directory):
    data = get_data(directory)
    subplot(data,title=get_title(directory),save=True)