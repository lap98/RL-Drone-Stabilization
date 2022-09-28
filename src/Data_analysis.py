#################################################
# Imports
#################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


#################################################
# Reinforcement Learning parameters
#################################################

save_path = "C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/drone_deploy/drone_data/states20220924_183928.npy"

states = np.load(save_path)
# delta_z/10, delta_x/10, delta_y/10, orient.w_val, orient.x_val, orient.y_val, orient.z_val, ang_acc.x_val/10, ang_acc.y_val/10, ang_acc.z_val/10,
# ang_vel.x_val/10, ang_vel.y_val/10, ang_vel.z_val/10, lin_acc.x_val/10, lin_acc.y_val/10, lin_acc.z_val/10, lin_vel.x_val/10, lin_vel.y_val/10, lin_vel.z_val/10
print(states.shape)

'''
rotations = []
for i in range(len(states)):
    rot = R.from_quat([states[i,4], states[i,5], states[i,6], states[i,3]])
    rotations.append(rot.as_euler('xyz', degrees=True))
rotations = np.array(rotations)
plt.title("roll pitch yaw degrees")
plt.plot(rotations[:,0], label="roll")
plt.plot(rotations[:,1], label="pitch")
plt.plot(rotations[:,2], label="yaw")
plt.legend()
plt.show()
'''
plt.title("lin velocity integrated m/s")
plt.plot(10*states[:,16],label="x lin vel")
plt.plot(10*states[:,17],label="y lin vel")
plt.plot(10*states[:,18],label="z lin vel")
plt.legend()
plt.show()
'''
plt.title("lin acc m/s2?")
plt.plot(10*states[:,13],label="x acc")
plt.plot(10*states[:,14],label="y acc")
plt.plot(10*states[:,15],label="z acc")
plt.legend()
plt.show()

plt.title("angular acc")
plt.plot(10*states[:,7],label="x ang acc")
plt.plot(10*states[:,8],label="y ang acc")
plt.plot(10*states[:,9],label="z ang acc")
plt.legend()
plt.show()
'''


dump_path = "C:/Users/aless/Downloads/Uni/Advanced_Deep_Learning_Models_and_Methods/Project/python_code/drone_deploy/drone_data/data_dump20220924_183928.npy"
states_drone = np.load(dump_path)
# x_roll y_pitch z_yaw v_x v_y v_z acc_x acc_y acc_z gyr_x gyr_y gyr_z gyr_x gyr_y acc_x acc_y acc_z gyr_x gyr_y gyr_z gyr_offset [vision vel x y z]
print(states_drone.shape)

'''
plt.title("pitch roll yaw degrees")
plt.plot(states_drone[:,0], label="roll")
plt.plot(states_drone[:,1], label="pitch")
plt.plot(states_drone[:,2], label="yaw")
plt.legend()
plt.show()

plt.title("lin velocity mm/s")
plt.plot(states_drone[:,3], label="x")
plt.plot(states_drone[:,4], label="y")
plt.plot(states_drone[:,5], label="z")
'''
plt.title("lin velocity vision")
plt.plot(states_drone[:,21], label="vis_x")
plt.plot(states_drone[:,22], label="vis_y")
plt.plot(states_drone[:,23], label="vis_z")
plt.legend()
plt.show()
'''
plt.title("acc lsb")
plt.plot(states_drone[:,6], label="acc_x")
plt.plot(states_drone[:,7], label="acc_y")
plt.plot(states_drone[:,8], label="acc_z")
plt.legend()
plt.show()

def smooth(x):
    x = np.pad(x, (0, 2), 'constant', constant_values=(0, 0))
    x = np.reshape(x, (-1, 4))
    x = np.average(x, axis=1)
    return x
def filter(x):
    return x[abs(x) < 1e2]

print(len(states_drone[:,14]))
print(len(filter(states_drone[:,14])))

plt.title("acc phys")
#plt.hist(states_drone[:,14], bins=1000)
plt.plot(filter(states_drone[:,14]), label="acc_x")
plt.plot(filter(states_drone[:,15]), label="acc_y")
plt.plot(filter(states_drone[:,16]), label="acc_z")
plt.legend()
plt.show()

plt.title("gyro phys")
#plt.hist(states_drone[:,14], bins=1000)
plt.plot(filter(states_drone[:,17]), label="gyro_x")
plt.plot(filter(states_drone[:,18]), label="gyro_y")
plt.plot(filter(states_drone[:,19]), label="gyro_z")
plt.legend()
plt.show()
'''