from casadi import *
import numpy as np

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



def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    # print(T)
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]
traj = lissajous

def cec_controller(cur_state, ref,cur_iter,Q_p,q,R_p,N):
     
     # Q_p,q,R_p = 7,5,2
     # Q_p,q,R_p = 2,30,5
     Q = Q_p*np.eye(2)
     R = R_p * np.eye(2)
     
     cur_ref = ref
     err_x = cur_state[0] - cur_ref[0]
     err_y = cur_state[1] - cur_ref[1]
     err_theta = cur_state[2] - cur_ref[2]
     err_theta = (err_theta + np.pi) % (2 * np.pi) - np.pi
     opti = Opti()
     # N = 5
     U = opti.variable(2,N)
     E = opti.variable(3,N+1)
     error = opti.parameter(3,1)
     opti.set_value(error, vcat([err_x, err_y, err_theta]))
     v = U[0,:]
     w = U[1,:]
     obj = 0
     #For loop to set objective function
     P_t = E[:2,:]
     theta_t = E[2,:]
     for i in range(N):
          obj += (P_t[:,i].T @ Q @ P_t[:,i] + q * (1 - cos(theta_t[i]))**2 + U[:,i].T @ R @ U[:,i])
     obj += E[:,-1].T @ E[:,-1]
     opti.minimize(obj)
     opti.subject_to(opti.bounded(0, U[0,:], 1))
     opti.subject_to(opti.bounded(-1, U[1,:], 1))
     opti.subject_to(E[:,0] == error)
     for i in range(1,N+1):
        
        ref_t = traj(cur_iter + i)
        ref_t_1 = traj(cur_iter + i - 1)
        
        opti.subject_to(E[:,i] == E[:,i-1] + vertcat(hcat([0.5 * cos(E[2, i-1] + ref_t_1[2]), 0]), hcat([0.5 * sin(E[2,i-1] + ref_t_1[2]), 0]), 
        hcat([0, 0.5])) @ U[:,i-1] - vcat([ref_t[0] - ref_t_1[0], ref_t[1] - ref_t_1[1], (ref_t[2] - ref_t_1[2] + np.pi) % (2 * np.pi) - np.pi]))

        opti.subject_to(opti.bounded(vcat([-3,-3]), E[:2, i] + vcat([ref_t[0], ref_t[1]]), vcat([3,3])))
        opti.subject_to((E[0,i] + ref_t[0] +  2)**2 + (E[1,i]+ ref_t[1] + 2)**2 > 0.5**2)
        opti.subject_to((E[0,i] + ref_t[0]- 1)**2 + (E[1,i] + ref_t[1] - 2)**2 > 0.5**2)

     opti.solver('ipopt')
     sol = opti.solve()
     return sol.value(U)[:,0]