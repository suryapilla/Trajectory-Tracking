import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal
# from main import lissajous
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

class GPI_controller():

    def __init__(self,traj,obs):

        self.ref = traj
        self.obstacles = obs
        self.time_step = 0.5
        self.num_x  = 8
        self.num_y = 8
        self.num_theta = 8
        self.N = 100
        self.vel = 10
        self.w = 10
        self.gamma = 0.95
        self.max_iters = 1000
        self.T = self.N * self.num_x * self.num_y * self.num_theta
        self.xlim = np.array([-3,3])
        self.ylim = np.array([-3,3])
        self.theta_lim = np.array([-np.pi,np.pi])
        self.vel_lim = np.array([0,1])
        self.w_lim = np.array([-1,1])
        self.err_ss = self.err_ss_matrix()
        self.err_cs = self.err_cs_matrix()
        self.stage_cost_matrix = self.generate_stage_cost()
        self.transition_state_idx, self.transition_probs = self.generate_transition_prob_matrix()
        self.optimal_policy = self.value_iteration()


    def err_ss_matrix(self):
        e_x = np.linspace(-3, 3, self.num_x)
        e_y = np.linspace(-3, 3, self.num_y)
        e_theta = np.linspace(-np.pi, np.pi, self.num_theta)

        grid_x, grid_y, grid_theta = np.meshgrid(e_x, e_y, e_theta)
        grid_x = grid_x.flatten()
        grid_y = grid_y.flatten()
        grid_theta = grid_theta.flatten()

        curr_refs = self.ref(np.arange(self.N))
        curr_refs = curr_refs[:, :2]

        obstacle_distances = np.sqrt((grid_x[:, np.newaxis] - self.obstacles[:, 0]) ** 2 +
                                    (grid_y[:, np.newaxis] - self.obstacles[:, 1]) ** 2)

        outside_obstacles = np.all(obstacle_distances >= self.obstacles[:, 2], axis=1)

        valid_indices = np.logical_and(np.logical_and(grid_x >= -3, grid_x <= 3),
                                    np.logical_and(grid_y >= -3, grid_y <= 3))

        valid_indices = np.logical_and(valid_indices, outside_obstacles)

        t_indices = np.repeat(np.arange(self.N), self.num_x * self.num_y * self.num_theta)
        err_ss = np.column_stack((t_indices[valid_indices],
                                            grid_x[valid_indices] + curr_refs[:, 0],
                                            grid_y[valid_indices] + curr_refs[:, 1],
                                            grid_theta[valid_indices]))

        np.save('errorStateSpace.npy', err_ss)
        return err_ss

    

    def err_cs_matrix(self):
        vel = np.linspace(self.vel_lim[0],self.vel_lim[1],self.vel)
        omega = np.linspace(self.w_lim[0],self.w_lim[1],self.w)
        err_cs = np.array(np.meshgrid(vel,omega)).T.reshape(-1, 2)
        u = np.zeros((2,self.w * self.vel))
        s = 0
        for i in tqdm(range(err_cs.shape[0])):
            u[:,s] = err_cs[i]
            s=s+1
        np.save('controlSpace.npy',u)
        return u
    

    def motion_model(self, x, u, ref_pos_curr, ref_pos_next, ref_theta_curr, ref_theta_next):
        updated_x = np.copy(x)

        updated_x[0] += u[0] * self.time_step * np.cos(x[2] + ref_theta_curr) + ref_pos_curr[0] - ref_pos_next[0]  # + np.random.normal(0,0.04)
        updated_x[1] += u[0] * self.time_step * np.sin(x[2] + ref_theta_curr) + ref_pos_curr[1] - ref_pos_next[1]  # + np.random.normal(0,0.04)
        updated_x[2] += u[1] * self.time_step + ref_theta_curr - ref_theta_next  # + np.random.normal(0,0.004)

        return updated_x



    def generate_stage_cost(self):
        Q = np.eye(2)
        R = np.eye(2)
        q = 1

        err_ss_1_3 = self.err_ss[1:3]

        stage_cost = np.dot(np.dot(err_ss_1_3.T, Q), err_ss_1_3) + q * (1 - np.cos(self.err_ss[3])) ** 2
        stage_cost += np.sum(np.dot(np.dot(self.err_cs.T, R), self.err_cs), axis=0)

        np.save('stageCost.npy', stage_cost)
        return stage_cost


    

    def value_iteration(self):
        P = np.load('transition_probabilities_matrix.npy')
        transition_idx = np.load('transition_states_idx.npy')

        Val_func = np.zeros(self.T)
        optimal_policy = np.zeros(self.T)

        L = np.load('stage_cost_matrix.npy')

        print("Starting Value Iteration...")

        for iteration in tqdm(range(self.max_iters)):
            V_old = Val_func.copy()
            exp_val_cost = np.sum(P * Val_func[transition_idx.astype(np.int64)], axis=2)
            Q = L + self.gamma * exp_val_cost
            V_new = np.min(Q, axis=1)
            optimal_policy = np.argmin(Q, axis=1)
            Val_func = V_new.copy()
            diff = np.linalg.norm(V_new - V_old)
            print("Iter: {}, Difference: {}".format(iteration, diff))
            if diff <= 0.001:
                break
        np.save('op.npy', optimal_policy)
        return optimal_policy

