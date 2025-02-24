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
from scipy.io import loadmat
from scipy.interpolate import interpn
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as st
import h5py

def error_metric(pred, test, div):
    out = np.linalg.norm(pred-test)/np.linalg.norm(div)
    return out

def error_metric2(pred, test):
    f = np.concatenate([(pred[0]-test[0]).reshape(-1,1),
                        (pred[1]-test[1]).reshape(-1,1),
                        (pred[2]-test[2]).reshape(-1,1)],1)
    div = np.concatenate([(test[0]).reshape(-1,1),
                        (test[1]).reshape(-1,1),
                        (test[2]).reshape(-1,1)],1)
    return np.linalg.norm(f, ord='fro')/np.linalg.norm(div, ord='fro')

def NRMSE(pred, test, div):
    out = np.sqrt(np.mean(np.square(pred-test))/np.mean(np.square(div)))
    return out

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

def acc_cal(dynamic_params, all_params, g_batch, model_fns):
    all_params["network"]["layers"] = dynamic_params
    weights = all_params["problem"]["loss_weights"]
    out, out_t = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[1.0, 0.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_x = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 1.0, 0.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_y = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 1.0, 0.0]]),(g_batch.shape[0],1)),model_fns)
    _, out_z = equ_func2(all_params, g_batch, jnp.tile(jnp.array([[0.0, 0.0, 0.0, 1.0]]),(g_batch.shape[0],1)),model_fns)

    u = all_params["data"]['u_ref']*out[:,0:1]
    v = all_params["data"]['v_ref']*out[:,1:2]
    w = all_params["data"]['w_ref']*out[:,2:3]

    ut = all_params["data"]['u_ref']*out_t[:,0:1]/all_params["data"]["domain_range"]["t"][1]
    vt = all_params["data"]['v_ref']*out_t[:,1:2]/all_params["data"]["domain_range"]["t"][1]
    wt = all_params["data"]['w_ref']*out_t[:,2:3]/all_params["data"]["domain_range"]["t"][1]

    ux = all_params["data"]['u_ref']*out_x[:,0:1]/all_params["data"]["domain_range"]["x"][1]
    vx = all_params["data"]['v_ref']*out_x[:,1:2]/all_params["data"]["domain_range"]["x"][1]
    wx = all_params["data"]['w_ref']*out_x[:,2:3]/all_params["data"]["domain_range"]["x"][1]
    px = all_params["data"]['u_ref']*out_x[:,3:4]/all_params["data"]["domain_range"]["x"][1]

    uy = all_params["data"]['u_ref']*out_y[:,0:1]/all_params["data"]["domain_range"]["y"][1]
    vy = all_params["data"]['v_ref']*out_y[:,1:2]/all_params["data"]["domain_range"]["y"][1]
    wy = all_params["data"]['w_ref']*out_y[:,2:3]/all_params["data"]["domain_range"]["y"][1]
    py = all_params["data"]['u_ref']*out_y[:,3:4]/all_params["data"]["domain_range"]["y"][1]

    uz = all_params["data"]['u_ref']*out_z[:,0:1]/all_params["data"]["domain_range"]["z"][1]
    vz = all_params["data"]['v_ref']*out_z[:,1:2]/all_params["data"]["domain_range"]["z"][1]
    wz = all_params["data"]['w_ref']*out_z[:,2:3]/all_params["data"]["domain_range"]["z"][1]
    pz = all_params["data"]['u_ref']*out_z[:,3:4]/all_params["data"]["domain_range"]["z"][1]
    

    acc_x = ut + u*ux + v*uy + w*uz
    acc_y = vt + u*vx + v*vy + w*vz
    acc_z = wt + u*wx + v*wy + w*wz

    return acc_x, acc_y, acc_z

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
        train_data, valid_data, all_params = self.c.data.train_data(all_params)
        all_params = self.c.problem.constraints(all_params)
        #valid_data = self.c.problem.exact_solution(all_params)
        model_fn = c.network.network_fn
        return all_params, model_fn, train_data, valid_data
#%%
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *
    from PINN_constants import *
    from PINN_problem import *
    import argparse
    from glob import glob
    #checkpoint_fol = "QUD_run_01"
    parser = argparse.ArgumentParser(description='QUD_PINN')
    parser.add_argument('-c', '--checkpoint', type=str, help='checkpoint', default="")
    args = parser.parse_args()
    checkpoint_fol = args.checkpoint
    print(checkpoint_fol, type(checkpoint_fol))
    path = "results/summaries/"
    with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','rb') as f:
        a = pickle.load(f)
    a['data_init_kwargs']['path'] = 'UrbanRescue/run065/'

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    all_params, model_fn, train_data, valid_data = run.test()
    

    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    vel_error_list_t = []
    acc_error_list_t = []
    vel_error_list_v = []
    acc_error_list_v = []
    pos_ref = all_params["domain"]["in_max"].flatten()
    vel_ref = np.array([all_params["data"]["u_ref"],
                        all_params["data"]["v_ref"],
                        all_params["data"]["w_ref"]])
    ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref']
    ref_data = {ref_key[i]:ref_val for i, ref_val in enumerate(np.concatenate([pos_ref,vel_ref]))}

    indexes_v, counts_v = np.unique(valid_data['pos'][:,0], return_counts=True)
    indexes_t, counts_t = np.unique(train_data['pos'][:,0], return_counts=True)
    g = 0
    timestep = 25
    for checkpoint in checkpoint_list:
        print(g)
        with open(checkpoint,'rb') as f:
            a = pickle.load(f)
        model = Model(all_params["network"]["layers"], model_fn)
        all_params["network"]["layers"] = from_state_dict(model, a).params

        ref_key = ['t_ref', 'x_ref', 'y_ref', 'z_ref', 'u_ref', 'v_ref', 'w_ref', 'u_ref']
        acc_t = np.concatenate([acc_cal(all_params["network"]["layers"], 
                                        all_params, 
                                        train_data['pos'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][10000*s:10000*(s+1)], 
                                        model_fn) 
                                for s in range(train_data['pos'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])].shape[0]//10000+1)],0)
        pred_t = np.concatenate([model_fn(all_params, 
                                          train_data['pos'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][10000*s:10000*(s+1)]) 
                                 for s in range(train_data['pos'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])].shape[0]//10000+1)],0)
        acc_v = np.concatenate([acc_cal(all_params["network"]["layers"], 
                                        all_params, 
                                        valid_data['pos'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][10000*s:10000*(s+1)], 
                                        model_fn) 
                                for s in range(valid_data['pos'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])].shape[0]//10000+1)],0)
        pred_v = np.concatenate([model_fn(all_params, 
                                          valid_data['pos'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][10000*s:10000*(s+1)]) 
                                 for s in range(valid_data['pos'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])].shape[0]//10000+1)],0)
        
        pred_t = np.concatenate([pred_t[:,i:i+1]*ref_data[ref_key[i+4]] for i in range(4)],1)
        pred_t[:,-1] = pred_t[:,-1]*1.185
        pred_t[:,-1] = pred_t[:,-1] - np.mean(pred_t[:,-1])
        pred_v = np.concatenate([pred_v[:,i:i+1]*ref_data[ref_key[i+4]] for i in range(4)],1)
        pred_v[:,-1] = pred_v[:,-1]*1.185
        pred_v[:,-1] = pred_v[:,-1] - np.mean(pred_v[:,-1])    

        f_t = np.concatenate([(pred_t[:,0]-train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,0]).reshape(-1,1), 
                            (pred_t[:,1]-train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,1]).reshape(-1,1), 
                            (pred_t[:,2]-train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,2]).reshape(-1,1)],1)
        div_t = np.concatenate([train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,0].reshape(-1,1), 
                              train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,1].reshape(-1,1), 
                              train_data['vel'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,2].reshape(-1,1)],1)
        f_v = np.concatenate([(pred_v[:,0]-valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,0]).reshape(-1,1), 
                            (pred_v[:,1]-valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,1]).reshape(-1,1), 
                            (pred_v[:,2]-valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,2]).reshape(-1,1)],1)
        div_v = np.concatenate([valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,0].reshape(-1,1), 
                              valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,1].reshape(-1,1), 
                              valid_data['vel'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,2].reshape(-1,1)],1)
        f_acc_t = np.concatenate([(acc_t[:,0]-train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,0]).reshape(-1,1), 
                                  (acc_t[:,1]-train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,1]).reshape(-1,1), 
                                  (acc_t[:,2]-train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,2]).reshape(-1,1)],1)
        div_acc_t = np.concatenate([train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,0].reshape(-1,1), 
                                    train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,1].reshape(-1,1), 
                                    train_data['acc'][np.sum(counts_t[:timestep]):np.sum(counts_t[:(timestep+1)])][:,2].reshape(-1,1)],1) 
        f_acc_v = np.concatenate([(acc_v[:,0]-valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,0]).reshape(-1,1), 
                                  (acc_v[:,1]-valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,1]).reshape(-1,1), 
                                  (acc_v[:,2]-valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,2]).reshape(-1,1)],1)
        div_acc_v = np.concatenate([valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,0].reshape(-1,1), 
                                    valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,1].reshape(-1,1), 
                                    valid_data['acc'][np.sum(counts_v[:timestep]):np.sum(counts_v[:(timestep+1)])][:,2].reshape(-1,1)],1)  

        vel_error_list_t.append(np.linalg.norm(f_t, ord='fro')/np.linalg.norm(div_t,ord='fro'))
        acc_error_list_t.append(np.linalg.norm(f_acc_t, ord='fro')/np.linalg.norm(div_acc_t,ord='fro'))
        vel_error_list_v.append(np.linalg.norm(f_v, ord='fro')/np.linalg.norm(div_v,ord='fro'))
        acc_error_list_v.append(np.linalg.norm(f_acc_v, ord='fro')/np.linalg.norm(div_acc_v,ord='fro'))
        g = g+1

    vel_error_t = np.array(vel_error_list_t)
    acc_error_t = np.array(acc_error_list_t)
    vel_error_v = np.array(vel_error_list_v)
    acc_error_v = np.array(acc_error_list_v)
    tol_error = np.concatenate([vel_error_t.reshape(-1,1),acc_error_t.reshape(-1,1),
                                vel_error_v.reshape(-1,1),acc_error_v.reshape(-1,1)],1)

    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)

    with open("datas/"+checkpoint_fol+"/error_evolution.pkl","wb") as f:
        pickle.dump(tol_error,f)
    f.close()

