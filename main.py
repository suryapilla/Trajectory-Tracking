#%%
from time import time
import numpy as np
from utils import visualize
# from casadi import *
from cec_controller import *
#%%
# Simulation params
np.random.seed(10)
time_step = 0.5 # time between steps in seconds
sim_time = 120    # simulation time

# Car params
x_init = 1.5
y_init = 0.0
theta_init = np.pi/2
v_max = 1
v_min = 0
w_max = 1
w_min = -1

# This function implement the car dynamics
def car_next_state(time_step, cur_state, control, noise = True):
    theta = cur_state[2]
    rot_3d_z = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    f = rot_3d_z @ control
    mu, sigma = 0, 0.04 # mean and standard deviation for (x,y)
    w_xy = np.random.normal(mu, sigma, 2)
    mu, sigma = 0, 0.004  # mean and standard deviation for theta
    w_theta = np.random.normal(mu, sigma, 1)
    w = np.concatenate((w_xy, w_theta))
    if noise:
        return cur_state + time_step*f.flatten() + w
    else:
        return cur_state + time_step*f.flatten()
#%%
if __name__ == '__main__':
    # Obstacles in the environment
    obstacles = np.array([[-2,-2,0.5], [1,2,0.5]])
    # Params
    traj = lissajous
    ref_traj = []
    error = 0.0
    error_total =0.0
    car_states = []
    times = []
    # Start main loop
    main_loop = time()  # return time in sec
    # Initialize state
    cur_state = np.array([x_init, y_init, theta_init])
    cur_iter = 0
    # Main loop
    Q_p,q,R_p,N = 30,5,10,20
    cnt=0
    while (cur_iter * time_step < sim_time):
        cnt=cnt+1
        t1 = time()
        # Get reference state
        cur_time = cur_iter*time_step
        cur_ref = traj(cur_iter)
        # Save current state and reference state for visualization
        ref_traj.append(cur_ref)
        car_states.append(cur_state)

        ################################################################
        # Generate control input
        # TODO: Replace this simple controller with your own controller

        control = cec_controller(cur_state, cur_ref,cur_iter,Q_p,q,R_p,N)

        
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<")
        # print("[v,w]", control)
        v = control[0]
        w = control[1]
        v = np.clip(v, v_min, v_max)
        w = np.clip(w,w_min, w_max)
        ################################################################

        # Apply control input
        next_state = car_next_state(time_step, cur_state, [v,w], noise=True)
        # Update current state
        
        cur_state = next_state
        # error = error + np.linalg.norm(cur_state - cur_ref)
        error= error + np.linalg.norm([cur_state[0] - cur_ref[0], cur_state[1] - cur_ref[1], 
        (cur_state[2] - cur_ref[2] + np.pi) % (2*np.pi) - np.pi])
        
        # Loop time
        t2 = time()
        times.append(t2-t1)
        
        cur_iter = cur_iter + 1

    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('Average iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('Final error: ', error)

    # # Visualization
    ref_traj = np.array(ref_traj)
    car_states = np.array(car_states)
    times = np.array(times)
    visualize(car_states, ref_traj, obstacles, times, time_step, save=True)

