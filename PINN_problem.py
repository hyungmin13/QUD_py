#%%
import jax.nn
import jax.numpy as jnp
import numpy as np
import h5py
from glob import glob

class Problem:
    @staticmethod
    def init_params(*args):
        raise NotImplementedError
    @staticmethod
    def exact_solution(all_params):
        raise NotImplementedError

class QUD(Problem):
    @staticmethod
    def init_params(domain_range, viscosity, loss_weights, path_s, frequency, problem_name):
        problem_params = {"domain_range":domain_range, "viscosity":viscosity,
                          "loss_weights":loss_weights, "path_s":path_s, "frequency":frequency,
                          "problem_name":problem_name}
        return problem_params


    
    @staticmethod
    def constraints(all_params):
        dimension = 4
        cotangents = [jnp.eye(dimension)[i:i+1,:] for i in range(dimension)]
        vel_unnorm_val = ('u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref')
        pos_unnorm_val = ((), (), (), (),
                      ('t'), ('t'), ('t'), ('t'),
                      ('x'), ('x'), ('x'), ('x'),
                      ('x', 'x'), ('x', 'x'), ('x', 'x'), ('x', 'x'),
                      ('y'), ('y'), ('y'), ('y'),
                      ('y', 'y'), ('y', 'y'), ('y', 'y'), ('y', 'y'),
                      ('z'), ('z'), ('z'), ('z'),
                      ('z', 'z'), ('z', 'z'), ('z', 'z'), ('z', 'z'))
        all_params["problem"]["cotangents"] = cotangents
        all_params["problem"]["vel_unnorm_val"] = vel_unnorm_val
        all_params["problem"]["pos_unnorm_val"] = pos_unnorm_val
        return all_params

class TBL(Problem):
    @staticmethod
    def init_params(domain_range, viscosity, loss_weights, path_s, frequency, problem_name):
        problem_params = {"domain_range":domain_range, "viscosity":viscosity,
                          "loss_weights":loss_weights, "path_s":path_s, "frequency":frequency,
                          "problem_name":problem_name}
        return problem_params

    @staticmethod
    def exact_solution(all_params):
        frequency = all_params["problem"]["frequency"]
        domain_range = all_params["problem"]["domain_range"]
        filenames = np.sort(glob(all_params["problem"]["path_s"]+'*.npy'))
        pos = []
        val = []
        for t in range(int(domain_range['t'][1]*frequency)+1):
            data = np.load(filenames[t]).reshape(-1,7)
            pos_ = np.concatenate([np.zeros(data[:,0:1].shape).reshape(-1,1)+t/frequency,
                                  0.001*data[:,0:1].reshape(-1,1),
                                  0.001*data[:,1:2].reshape(-1,1),
                                  0.001*data[:,2:3].reshape(-1,1)],1)
            val_ = np.concatenate([data[:,3:4].reshape(-1,1),
                                  data[:,4:5].reshape(-1,1),
                                  data[:,5:6].reshape(-1,1),
                                  data[:,6:7].reshape(-1,1),],1)
            pos.append(pos_)
            val.append(val_)
        pos = np.concatenate(pos,0)
        val = np.concatenate(val,0)
        key = ['t', 'x', 'y', 'z']
        for i in range(pos.shape[1]):
            pos[:,i] = pos[:,i]/domain_range[key[i]][1]

        return {"pos":pos, "vel":val}
    
    @staticmethod
    def constraints(all_params):
        dimension = 4
        cotangents = [jnp.eye(dimension)[i:i+1,:] for i in range(dimension)]
        vel_unnorm_val = ('u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref')
        pos_unnorm_val = ((), (), (), (),
                      ('t'), ('t'), ('t'), ('t'),
                      ('x'), ('x'), ('x'), ('x'),
                      ('x', 'x'), ('x', 'x'), ('x', 'x'), ('x', 'x'),
                      ('y'), ('y'), ('y'), ('y'),
                      ('y', 'y'), ('y', 'y'), ('y', 'y'), ('y', 'y'),
                      ('z'), ('z'), ('z'), ('z'),
                      ('z', 'z'), ('z', 'z'), ('z', 'z'), ('z', 'z'))
        all_params["problem"]["cotangents"] = cotangents
        all_params["problem"]["vel_unnorm_val"] = vel_unnorm_val
        all_params["problem"]["pos_unnorm_val"] = pos_unnorm_val
        return all_params

class HIT(Problem):
    @staticmethod
    def init_params(domain_range, viscosity, loss_weights, path_s, frequency):
        problem_params = {"domain_range":domain_range, "viscosity":viscosity,
                          "loss_weights":loss_weights, "path_s":path_s, "frequency":frequency}
        return problem_params
    
    #@staticmethod
    #def sample_constraints(all_params, ):
    #    x_batch_phys = 
    
    @staticmethod
    def exact_solution(all_params):
        frequency = all_params["problem"]["frequency"]
        domain_range = all_params["problem"]["domain_range"]
        datas = h5py.File(all_params["problem"]["path_s"],'r')
        datakeys = ['t','x','y','z','u','v','w','p']
        datas = {datakey:np.array(datas[datakey],dtype=np.float32) for datakey in datakeys}
        pos = []
        for t in range(int(domain_range['t'][1]*frequency)+1):
            pos_ = np.concatenate([np.zeros(datas['x'].shape).reshape(-1,1)+t/frequency,
                                  0.001*datas['x'].reshape(-1,1),
                                  0.001*datas['y'].reshape(-1,1),
                                  0.001*datas['z'].reshape(-1,1)],1)
            pos.append(pos_)
        pos = np.concatenate(pos,0)
        key = ['t', 'x', 'y', 'z']
        for i in range(pos.shape[1]):
            pos[:,i] = pos[:,i]/domain_range[key[i]][1]
        val = np.concatenate([0.001*datas['u'].reshape(-1,1),
                              0.001*datas['v'].reshape(-1,1),
                              0.001*datas['w'].reshape(-1,1),
                              datas['p'].reshape(-1,1),],1)
        return {"pos":pos, "vel":val}
    
    @staticmethod
    def constraints(all_params):
        dimension = 4
        cotangents = [jnp.eye(dimension)[i:i+1,:] for i in range(dimension)]
        vel_unnorm_val = ('u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref',
                        'u_ref', 'v_ref', 'w_ref', 'u_ref')
        pos_unnorm_val = ((), (), (), (),
                      ('t'), ('t'), ('t'), ('t'),
                      ('x'), ('x'), ('x'), ('x'),
                      ('x', 'x'), ('x', 'x'), ('x', 'x'), ('x', 'x'),
                      ('y'), ('y'), ('y'), ('y'),
                      ('y', 'y'), ('y', 'y'), ('y', 'y'), ('y', 'y'),
                      ('z'), ('z'), ('z'), ('z'),
                      ('z', 'z'), ('z', 'z'), ('z', 'z'), ('z', 'z'))
        all_params["problem"]["cotangents"] = cotangents
        all_params["problem"]["vel_unnorm_val"] = vel_unnorm_val
        all_params["problem"]["pos_unnorm_val"] = pos_unnorm_val
        return all_params

    #@staticmethod
    #def loss_fn(all_params, ):
if __name__ == "__main__":
    all_params = {"problem":{}}
    #frequency = 1250
    frequency = 17594
    domain_range = {'t':(0,50/frequency), 'x':(0,0.056), 'y':(0,0.012), 'z':(0,0.00424)}
    viscosity = 15*10e-6
    loss_weights = (1,1,1,0.00001,0.00001,0.00001,0.00001)
    constraints = ('first_order_diff', 'second_order_diff', 'second_order_diff', 'second_order_diff')
    #path_s = '/home/bussard/hyun_sh/TBL_PINN/data/HIT/IsoturbFlow.mat'
    path_s = '/scratch/hyun/Ground/'
    #all_params["problem"] = HIT.init_params(domain_range, viscosity, loss_weights, path_s, frequency)
    #all_params = HIT.constraints(all_params)
    #datas = HIT.exact_solution(all_params)
    all_params["problem"] = TBL.init_params(domain_range, viscosity, loss_weights, path_s, frequency)
    all_params = TBL.constraints(all_params)
    datas = TBL.exact_solution(all_params)


