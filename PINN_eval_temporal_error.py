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
import h5py
class Model(struct.PyTreeNode):
    params: Any
    forward: callable = struct.field(pytree_node=False)
    def __apply__(self,*args):
        return self.forward(*args)
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
    #a['problem_init_kwargs']['path_s'] = 'Ground/'
    #with open(path+checkpoint_fol+'/constants_'+ str(checkpoint_fol) +'.pickle','wb') as f:
    #    pickle.dump(a,f)

    values = list(a.values())

    c = Constants(run = values[0],
                domain_init_kwargs = values[1],
                data_init_kwargs = values[2],
                network_init_kwargs = values[3],
                problem_init_kwargs = values[4],
                optimization_init_kwargs = values[5],)
    run = PINN(c)
    #checkpoint_list = np.sort(glob(run.c.model_out_dir+'/*.pkl'))
    #with open(run.c.model_out_dir + "saved_dic_720000.pkl","rb") as f:
    
    checkpoint_list = sorted(glob(run.c.model_out_dir+'/*.pkl'), key=lambda x: int(x.split('_')[4].split('.')[0]))
    print(checkpoint_list)
    with open(checkpoint_list[-1],'rb') as f:
        a = pickle.load(f)
    all_params, model_fn, train_data, valid_data = run.test()

    model = Model(all_params["network"]["layers"], model_fn)
    all_params["network"]["layers"] = from_state_dict(model, a).params
#%% temporal error는 51개의 시간단계에대해서 [:,0]는 velocity error, [:,1]은 pressure error
    uni, counts = np.unique(valid_data['pos'][:,0],return_counts=True)
    temporal_error_vel_t_list = []
    temporal_error_vel_v_list = []
    temporal_error_acc_t_list = []
    temporal_error_acc_v_list = []
    c = 0
    dynamic_params = all_params["network"].pop("layers")
    for j in range(51):
        print(j)
        index_t = np.where(train_data['pos'][:,3]<0.4)
        train_data = {data_keys[i]:train_data[data_keys[i]][index_t[0],:] for i in range(len(data_keys))}
        index_v = np.where(valid_data['pos'][:,3]<0.4)
        valid_data = {data_keys[i]:valid_data[data_keys[i]][index_v[0],:] for i in range(len(data_keys))}

        pred_t = model_fn(all_params, train_data['pos'][c:counts[j]+c,:])
        pred_v = model_fn(all_params, valid_data['pos'][c:counts[j]+c,:])
        acc_x_t, acc_y_t, acc_z_t = acc_cal(dynamic_params, all_params, train_data['pos'][c:counts[j]+c,:], model_fn)
        acc_x_v, acc_y_v, acc_z_v = acc_cal(dynamic_params, all_params, valid_data['pos'][c:counts[j]+c,:], model_fn)
        output_keys = ['u', 'v', 'w', 'p']
        output_unnorm = [all_params["data"]['u_ref'],all_params["data"]['v_ref'],
                        all_params["data"]['w_ref'],1.185*all_params["data"]['u_ref']]
        
        outputs_t = {output_keys[i]:pred_t[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs_t['p'] = outputs_t['p'] - np.mean(outputs_t['p'])
        output_ext_t = {output_keys[i]:train_data['vel'][c:counts[j]+c,i] for i in range(len(output_keys)-1)}
        output_ext_acc_t = {output_keys[i]:train_data['acc'][c:counts[j]+c,i] for i in range(len(output_keys)-1)}

        outputs_v = {output_keys[i]:pred_v[:,i]*output_unnorm[i] for i in range(len(output_keys))}
        outputs_v['p'] = outputs_v['p'] - np.mean(outputs_v['p'])
        output_ext_v = {output_keys[i]:valid_data['vel'][c:counts[j]+c,i] for i in range(len(output_keys)-1)}
        output_ext_acc_v = {output_keys[i]:valid_data['acc'][c:counts[j]+c,i] for i in range(len(output_keys)-1)}
        
        c = c + counts[j]
        f_t = np.concatenate([(outputs_t['u']-output_ext_t['u']).reshape(-1,1), 
                            (outputs_t['v']-output_ext_t['v']).reshape(-1,1), 
                            (outputs_t['w']-output_ext_t['w']).reshape(-1,1)],1)
        div_t = np.concatenate([output_ext_t['u'].reshape(-1,1), output_ext_t['v'].reshape(-1,1), 
                            output_ext_t['w'].reshape(-1,1)],1)

        f_at = np.concatenate([(acc_x_t-output_ext_acc_t['u']).reshape(-1,1), 
                            (acc_y_t-output_ext_acc_t['v']).reshape(-1,1), 
                            (acc_z_t-output_ext_acc_t['w']).reshape(-1,1)],1)
        div_at = np.concatenate([output_ext_acc_t['u'].reshape(-1,1), output_ext_acc_t['v'].reshape(-1,1), 
                            output_ext_acc_t['w'].reshape(-1,1)],1)        

        f_v = np.concatenate([(outputs_v['u']-output_ext_v['u']).reshape(-1,1), 
                            (outputs_v['v']-output_ext_v['v']).reshape(-1,1), 
                            (outputs_v['w']-output_ext_v['w']).reshape(-1,1)],1)
        div_v = np.concatenate([output_ext_v['u'].reshape(-1,1), output_ext_v['v'].reshape(-1,1), 
                            output_ext_v['w'].reshape(-1,1)],1)

        f_av = np.concatenate([(acc_x_v-output_ext_acc_v['u']).reshape(-1,1), 
                            (acc_y_v-output_ext_acc_v['v']).reshape(-1,1), 
                            (acc_z_v-output_ext_acc_v['w']).reshape(-1,1)],1)
        div_av = np.concatenate([output_ext_acc_v['u'].reshape(-1,1), output_ext_acc_v['v'].reshape(-1,1), 
                            output_ext_acc_v['w'].reshape(-1,1)],1) 

        temporal_error_vel_t_list.append(np.linalg.norm(f_t, ord='fro')/np.linalg.norm(div_t,ord='fro'))
        temporal_error_vel_v_list.append(np.linalg.norm(f_v, ord='fro')/np.linalg.norm(div_v,ord='fro'))
        temporal_error_acc_t_list.append(np.linalg.norm(f_at, ord='fro')/np.linalg.norm(div_at,ord='fro'))
        temporal_error_acc_v_list.append(np.linalg.norm(f_av, ord='fro')/np.linalg.norm(div_av,ord='fro'))  

    temporal_error = np.concatenate([np.array(temporal_error_vel_t_list).reshape(-1,1),
                                     np.array(temporal_error_vel_v_list).reshape(-1,1),
                                     np.array(temporal_error_acc_t_list).reshape(-1,1),
                                     np.array(temporal_error_acc_v_list).reshape(-1,1)],1)

    if os.path.isdir("datas/"+checkpoint_fol):
        pass
    else:
        os.mkdir("datas/"+checkpoint_fol)

    with open("datas/"+checkpoint_fol+"/temporal_error.pkl","wb") as f:
        pickle.dump(temporal_error,f)
    f.close()

# %%
