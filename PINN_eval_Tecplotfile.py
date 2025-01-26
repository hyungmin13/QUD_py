#%%
import jax
import jax.numpy as jnp
from jax import random
from time import time
import optax
from jax import value_and_grad
from functools import partial
from jax import jit
from tqdm import tqdm
from typing import Any
from flax import struct
from flax.serialization import to_state_dict, from_state_dict
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.interpolate import interpn
import h5py
from scipy.io import loadmat
import argparse
from Tecplot_mesh import tecplot_Mesh
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)

class PINNbase:
    def __init__(self,c):
        c.get_outdirs()
        c.save_constants_file()
        self.c=c
class PINN(PINNbase):
    def test(self):
        all_params = {"domain":{}, "data":{}, "network":{}, "problem":{}}
        all_params["domain"] = self.c.domain.init_params(**self.c.domain_init_kwargs)
        all_params["data"] = self.c.data.init_params(**self.c.data_init_kwargs)
        global_key = random.PRNGKey(42)
        all_params["network"] = self.c.network.init_params(**self.c.network_init_kwargs)
        all_params["problem"] = self.c.problem.init_params(**self.c.problem_init_kwargs)
        optimiser = self.c.optimization_init_kwargs["optimiser"](self.c.optimization_init_kwargs["learning_rate"])
        grids, all_params = self.c.domain.sampler(all_params)
        train_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
    
def equ_func(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    def u_tt(batch):
        return jax.jvp(u_t,(batch,), (cotangent, ))[1]
    out_x, out_xx = jax.jvp(u_tt, (g_batch,), (cotangent,))
    return out_x, out_xx

def equ_func2(all_params, g_batch, cotangent, model_fns):
    def u_t(batch):
        return model_fns(all_params, batch)
    out, out_t = jax.jvp(u_t, (g_batch,), (cotangent,))
    return out, out_t

def Derivatives(dynamic_params, all_params, g_batch, model_fns):
    keys = ['u_ref', 'v_ref', 'w_ref', 'u_ref']

    all_params["network"]["layers"] = dynamic_params
    out, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    out, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)    
    uvwp = np.concatenate([out[:,k:(k+1)]*all_params["data"][keys[k]] for k in range(len(keys))],1)
    uvwp[:,-1] = 1.185*uvwp[:,-1]
    uxs = np.concatenate([out_x[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,1] for k in range(len(keys))],1)
    uys = np.concatenate([out_y[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,2] for k in range(len(keys))],1)
    uzs = np.concatenate([out_z[:,k:(k+1)]*all_params["data"][keys[k]]/all_params["domain"]["in_max"][0,3] for k in range(len(keys))],1)
    deriv_mat = np.concatenate([np.expand_dims(uxs,2),np.expand_dims(uys,2),np.expand_dims(uzs,2)],2)
    vor_mag = np.sqrt((deriv_mat[:,1,2]-deriv_mat[:,2,1])**2+
                      (deriv_mat[:,2,0]-deriv_mat[:,0,2])**2+
                      (deriv_mat[:,0,1]-deriv_mat[:,1,0])**2)
    Q = 0.5 * sum(-np.abs(0.5 * (deriv_mat[:, i, j] + deriv_mat[:, j, i]))**2 +
                  np.abs(0.5 * (deriv_mat[:, i, j] - deriv_mat[:, j, i]))**2 
                  for i in range(3) for j in range(3))
    return uvwp, vor_mag, Q
#%%
print(all_params["domain"]["in_max"])
#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    #parser = argparse.ArgumentParser(description='TBL_PINN')
    #parser.add_argument('-t', '--timestep', type=int, help='timestep', default=25)
    #args = parser.parse_args()
    #timestep = args.timestep
    u_tau = 15*10**(-6)/36.2/10**(-6)
    u_ref_n = 4.9968*10**(-2)/u_tau
    delta = 36.2*10**(-6)
    x_ref_n = 1.0006*10**(-3)/delta

    checkpoint_fol = "TBL_run_06"
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = '/scratch/hyun/TBL/'
    a['problem_init_kwargs']['path_s'] = '/scratch/hyun/Ground/'
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
        pickle.dump(a,f)

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)

    with open(run.c.model_out_dir + "saved_dic_340000.pkl","rb") as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params

#%%
    timestep = 25
    pos_ref = all_params["domain"]["in_max"].flatten()
    vel_ref = np.array([all_params["data"]["u_ref"],
                        all_params["data"]["v_ref"],
                        all_params["data"]["w_ref"]])
#%%
    ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref']
    ref_data = {ref_key[i]:ref_val for i, ref_val in enumerate(np.concatenate([pos_ref,vel_ref]))}
    datapath = 'eval_grid/PG_TBL_dnsinterp.mat'
    data = loadmat(datapath)
    eval_key = ['x', 'y', 'z', 'x_pred', 'y_pred', 'z_pred', 'u1', 'v1', 'w1', 'p1', 'um', 'vm', 'wm']
    DNS_grid = (0.001*data['y'][:,0,0], 0.001*data['x'][0,:,0], 0.001*data['z'][0,0,:])
    eval_grid = np.concatenate([0.001*data['y_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1),
                                0.001*data['x_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1),
                                0.001*data['z_pred'].reshape(32,88,410)[:31,:,:].reshape(-1,1)],1)
    vel_ground = [interpn(DNS_grid, data[eval_key[i+6]], eval_grid).reshape(31,88,410) for i in range(3)]
    fluc_ground = [interpn(DNS_grid, data[eval_key[i+6]], eval_grid).reshape(31,88,410) - data[eval_key[i+10]].reshape(32,88,410)[:31,:,:] for i in range(3)]
    p_cent = interpn(DNS_grid, data['p1'], eval_grid).reshape(31,88,410)
    fluc_ground.append(p_cent-np.mean(p_cent))
    V_mag = np.sqrt(fluc_ground[0]**2+fluc_ground[1]**2+fluc_ground[2]**2)
    div_list = [V_mag, V_mag, V_mag, fluc_ground[-1]]
    fluc_ground[-1] = (fluc_ground[-1]*u_ref_n**2 - 0.0025*eval_grid[:,1].reshape(31,88,410)[0,0,:]*x_ref_n)/u_ref_n**2

    eval_grid_n = np.concatenate([np.zeros((eval_grid.shape[0],1))+timestep*(1/17954),
                                eval_grid[:,1:2], eval_grid[:,0:1], eval_grid[:,2:3]],1)
    for i in range(eval_grid_n.shape[1]): eval_grid_n[:,i] = eval_grid_n[:,i]/ref_data[ref_key[i]] 
#%%
    eval_grid_z = eval_grid.reshape(31,88,410,3)
    x_e = eval_grid_z[:,:,:,1]
    y_e = eval_grid_z[:,:,:,0]
    z_e = eval_grid_z[:,:,:,2]

#%%
    dynamic_params = all_params["network"].pop("layers")
    uvwp, vor_mag, Q = zip(*[Derivatives(dynamic_params, all_params, eval_grid_n[i:i+10000], model_fn) 
                             for i in range(0, eval_grid_n.shape[0], 10000)])
    uvwp = np.concatenate(uvwp, axis=0)
    vor_mag = np.concatenate(vor_mag, axis=0)
    Q = np.concatenate(Q, axis=0)

    u_fluc = uvwp[:,0].reshape(31,88,410) - data[eval_key[10]].reshape(32,88,410)[:31,:,:]
    v_fluc = uvwp[:,1].reshape(31,88,410) - data[eval_key[11]].reshape(32,88,410)[:31,:,:]
    w_fluc = uvwp[:,2].reshape(31,88,410) - data[eval_key[12]].reshape(32,88,410)[:31,:,:]
    p_cent = uvwp[:,3].reshape(31,88,410) - np.mean(uvwp[:,3].reshape(31,88,410))

    u_error = np.sqrt(np.square(uvwp[:,0].reshape(31,88,410) - vel_ground[0]))
    v_error = np.sqrt(np.square(uvwp[:,1].reshape(31,88,410) - vel_ground[1]))
    w_error = np.sqrt(np.square(uvwp[:,2].reshape(31,88,410) - vel_ground[2]))
    p_error = np.sqrt(np.square(uvwp[:,3].reshape(31,88,410) - fluc_ground[3]))
#%%
    filename = "datas/"+checkpoint_fol+"/TBL_eval_"+str(timestep)+".dat"
    if "datas/"+checkpoint_fol:
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)
    X, Y, Z = (x_e[0,0,:].shape[0], y_e[0,:,0].shape[0], z_e[:31,0,0].shape[0])
    vars = [('u_ground[m/s]', np.float32(vel_ground[0].reshape(-1))), ('v_ground[m/s]', np.float32(vel_ground[1].reshape(-1))), ('w_ground[m/s]', np.float32(vel_ground[2].reshape(-1))), ('p_ground[Pa]', np.float32(fluc_ground[-1].reshape(-1))),
            ('u_fluc_ground[m/s]', np.float32(fluc_ground[0].reshape(-1))), ('v_fluc_ground[m/s]', np.float32(fluc_ground[1].reshape(-1))), ('w_fluc_ground[m/s]', np.float32(fluc_ground[2].reshape(-1))),
            ('u_pred[m/s]',np.float32(uvwp[:,0].reshape(31,88,410).reshape(-1))), ('v_pred[m/s]',uvwp[:,1].reshape(31,88,410).reshape(-1)),
            ('w_pred[m/s]',uvwp[:,2].reshape(31,88,410).reshape(-1)), ('p_pred[Pa]',uvwp[:,3].reshape(-1)),
            ('u_fluc[m/s]',u_fluc.reshape(-1)), ('v_fluc[m/s]',v_fluc.reshape(-1)), ('w_fluc[m/s]',w_fluc.reshape(-1)),
            ('u_error[m/s]',u_error.reshape(-1)), ('v_error[m/s]',v_error.reshape(-1)), ('w_error[m/s]',w_error.reshape(-1)), ('p_error[Pa]',p_error.reshape(-1)),
            ('vormag[1/s]',vor_mag.reshape(-1)), ('Q[1/s^2]', Q.reshape(-1))]
    fw = 27
    tecplot_Mesh(filename, X, Y, Z, x_e.reshape(-1), y_e.reshape(-1), z_e.reshape(-1), vars, fw)

