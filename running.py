import numpy as np
from sklearn.preprocessing import normalize
from scipy.linalg import orth
import time
import os
from yaml import safe_load as load
import pathos.multiprocessing as mp
import gc
from pathlib import Path

### __main__ configuration ###
ncpu = mp.cpu_count()
parallel_procs = 1
computation_database_path = 'computation-database/'
basepath = 'hyperparameters/Test'


### ACTIVATION FUNCTIONS and TARGETS###
activation_dict = {
    'relu': lambda x: np.maximum(x,0),
    'H3': lambda x: x**3 - 3*x,
    'H4': lambda x: x**4 - 6*x**2 + 3,
    'tanh_H3': lambda x: np.tanh(x**3 - 3*x),
    'tanh': np.tanh,
    'H2': lambda x: x**2 - 1,
    'kstar4': lambda x: x**2 * np.exp(-x**2)
}
activation_derivative_dict = {
    'relu': lambda x: (x>0).astype(float),
    'H3': lambda x: 3*x**2 - 3,
    'H4': lambda x: 4*x**3 - 12*x,
    'tanh_H3': lambda x: 1/np.cosh(x**3 - 3*x)**2,
    'tanh': lambda x: 1/np.cosh(x)**2
}
exponents_learning_rate = {
    'sgd_optimal': -3/2,
    'sam': -1.,
    'kstar4': -2.,
    'constant': 0.,
    'kstar3': -1.5
}
targets = {
    'fig2a': lambda z: z[0] + z[0]*z[1],
    'fig2b': lambda z: np.tanh(z[0] + (z[1]**3 - 3*z[1])),
    'fig2bnotanh': lambda z: z[0] + z[1]**3 - 3*z[1],
    'fig2bstaircase': lambda z: z[0] + z[0]*(z[1]**3 - 3*z[1]),
    'fig2c':  lambda z: np.tanh(z[0]) + np.tanh(z[1]),
    'fig2cH3': lambda z: z[0]**3 - 3*z[0] + z[1]**3 - 3*z[1],
    'fig2cRelu': lambda z: np.maximum(z[0],0) + np.maximum(z[1],0),
    'fig2cAubin': lambda z: np.sign(z[0]) + np.sign(z[1]),
    'sign(z1z2z3)': lambda z: np.sign(z[0]*z[1]*z[2]),
    'sign(z1z2z3)_enchanted': lambda z: 5*np.sign(z[0]*z[1]*z[2]),
    'z1z2z3': lambda z: z[0]*z[1]*z[2],
    '4index': lambda z: (z[0]**3 -3*z[0]) + z[1]*z[2]*z[3], 
    'sign(z1z2)': lambda z: np.sign(z[0]*z[1]), 
    'z1z2': lambda z: z[0]*z[1],
    'relu': lambda x: np.maximum(x,0),
    'H3z1z2z3': lambda z: z[0]**3 - 3*z[0] + z[0]*z[1]*z[2],
    'sqstairs': lambda z: z[0]**2 + np.sign(z[0]*z[1]*z[2]),
    'sqstairs_enchanted': lambda z: (z[0]**2-1) + 5 * np.sign(z[0]*z[1]*z[2]),
    'sqstairsH3': lambda z: z[0]**3 - 3*z[0] + np.sign(z[0]*z[1]*z[2]),
    'sqstairsH4': lambda z: z[0]**4-6*z[0]**2+3 + np.sign(z[0]*z[1]*z[2]),
    'sqstairsH4_enchanted': lambda z: z[0]**4-6*z[0]**2+3 + 5*np.sign(z[0]*z[1]*z[2])
}

def run_path(
        path,
        read_all = True,
        recursive = True,
        worker_run = False,
        runner_params = None,
        force_run = False
    ):
    # Getting the file list
    pathlist = Path(basepath).glob(f"{'**/' if recursive else ''}*.yaml")
    if read_all:
        run_filespaths = [str(path) for path in pathlist]
        print('Files detetched: ', len(run_filespaths))

    for filepath in run_filespaths:
        run_file(filepath, worker_run = worker_run, runner_params = runner_params, force_run = force_run)

def run_file(filepath, worker_run = False, runner_params = None, force_run = False):
    if worker_run:
        if runner_params is None:
            raise ValueError('runner_params must be provided if worker_run is True')
    flag = str(filepath).split(sep="/")[-1].removesuffix(".yaml")
    yaml_file = open(filepath, 'r')
    hyperparams = load(yaml_file)
    ps = hyperparams['ps']
    student_activation_choices = hyperparams['student_activation_choices'] # student is always a committee
    choice_gammas = hyperparams['choice_gammas']
    algo_methods = hyperparams['algo_methods']
    batch_size_choices = hyperparams['batch_size_choices']
    k = hyperparams['k']
    noise = hyperparams['noise']
    predictor_interaction = hyperparams['predictor_interaction']
    nseeds = hyperparams['nseeds'] 
    prefactors = hyperparams['prefactors']
    coefficient_time = hyperparams['coefficient_time']
    ds = np.array(hyperparams['ds'], dtype=int)
    choice_a = str(hyperparams['choice_a'])
    choice_time_scaling = str(hyperparams['choice_time_scaling'])
    choice_init = str(hyperparams['choice_init'])
    spherical_flag = hyperparams['spherical']
    rho_prefactors = hyperparams['rho_prefactors']

    try:
        teacher_kind = hyperparams['teacher_kind']
        target_choices = hyperparams['target_choices']
    except KeyError:
        teacher_kind = 'committees'
        teacher_activation_choices = hyperparams['teacher_activation_choices']
    ### RELEVANT DICTIONARIES ### (Batch size is defined depending on d, and second layer as a function of p)
    paths_to_folders = {
        'gd': computation_database_path+'gd_runs/gd_run',
        'proxy_sam': computation_database_path+'proxy_sam_runs/sam_run',
        'true_sam': computation_database_path+'sam_runs/sam_run',
        'no_resample': computation_database_path+'no_resample_runs/no_resample_run',
        'exponential_gd': computation_database_path+'exponential_gd/exponential_gd'
    }
    if teacher_kind == 'committees':
        targets_to_run = { teacher_activation_choice: lambda local_fields: 1/k * np.sum(activation_dict[teacher_activation_choice](local_fields),axis=-1)  for teacher_activation_choice in teacher_activation_choices}
    else:
        targets_to_run = { target_choice: targets[target_choice] for target_choice in target_choices}
        
    ### MAIN LOOP ###
    for p in ps:
        for target_choice, target in targets_to_run.items(): 
            for student_activation_choice in student_activation_choices:
                activation = activation_dict[student_activation_choice] # def student activation
                activation_derivative = activation_derivative_dict[student_activation_choice]  # def student act der
                for prefactor in prefactors:
                    for choice_gamma in choice_gammas:
                        for algo_method in algo_methods:
                            path_to_folder = paths_to_folders[algo_method]
                            for batch_size_choice in batch_size_choices:
                                for rp_counter, rho_prefactor in enumerate(rho_prefactors):
                                    if algo_method != 'true_sam' and rp_counter != 0:
                                        continue
                                    for i,d in enumerate(ds):
                                        batch_sizes_fractions = {'one': d, 'tenth': 10, 'fifth': 5, 'half': 2, 'full': 1,  'double': 0.5}
                                        initializations = {'random': 1/np.sqrt(d), 'almost_zero': 1e-10, 'zero': 0.0, 'first_direction': None}
                                        coefficient_batch_size = batch_sizes_fractions[batch_size_choice]
                                        n = int(d/coefficient_batch_size)
                                        exponent_learning_rate = exponents_learning_rate[choice_gamma]
                                        gamma = prefactor * n * np.power(d,exponent_learning_rate) * p
                                        time_scalings = {'kstar3': int(coefficient_time*d**2/n), 'multiple_pass_easy': int(coefficient_time*d/n), 'dlogd': int(coefficient_time*d*np.log(d)/n), 'kstar4': int(coefficient_time*d**3/n) }
                                        T = time_scalings[choice_time_scaling]
                                        if choice_init in ['random', 'almost_zero', 'zero']:
                                            t = initializations[choice_init]
                                        else:
                                            t = None
                                        for seed in range(nseeds):
                                            if algo_method == 'true_sam':
                                                rhoflag = rho_prefactor
                                            else:
                                                rhoflag = 'None'
                                            filename = f'{path_to_folder}_(name={flag})d={d}_p={p}_choice_gamma={prefactor}_{choice_gamma}_teacher={target_choice}_student={student_activation_choice}_batch_size={batch_size_choice}_choice_a={choice_a}_seed={seed}_init={choice_init}_k={k}_spherical={spherical_flag}_rhopref={rhoflag}_CT={coefficient_time}.npz'
                                            if os.path.exists(filename) and (not force_run):
                                                print(f'File {filename} exists. Skipping this simulation', flush = True)
                                            else:
                                                print(f'File {filename} does not exist. Running simulation', flush = True)
                                                ic_seed = seed^25081926
                                                rng = np.random.default_rng(ic_seed)
                                                Wtarget = orth((normalize(rng.normal(size=(k,d)), axis=1, norm='l2')).T).T
                                                Wtild = normalize(rng.normal(size=(p,d)), axis=1, norm='l2')
                                                Wtild_target = np.einsum('ji,ri,rh->jh', Wtild , Wtarget ,Wtarget)
                                                W0_orth = normalize(Wtild - Wtild_target, axis=1, norm='l2')
                                                if t is not None:
                                                    W0 = (t*normalize(Wtild_target,norm='l2',axis=1) + np.sqrt(1-t**2)*W0_orth)
                                                else:
                                                    if choice_init == 'first_direction':
                                                        W0 = np.stack([np.copy(Wtarget[0]) for _ in range(p)], axis=0)
                                                    else:
                                                        raise ValueError(f'Initialization not known: {choice_init}')
                                                second_layers = {'rademacher': np.sign(rng.normal(size=(p,))), 'gaussian': rng.normal(size=(p,)), 'ones': np.ones(p)}
                                                a0 = second_layers[choice_a]
                                                P = Wtarget @ Wtarget.T
                                                M0 = W0 @ Wtarget.T
                                                Q0 = W0 @ W0.T
                                                param = (
                                                    target, Wtarget, n, activation, W0, a0, activation_derivative, gamma, noise, predictor_interaction, seed, T, filename, choice_time_scaling, algo_method, spherical_flag, rho_prefactor 
                                                )
                                                if worker_run:
                                                    runner_params.append(param)
                                                else:
                                                    do_run(param)
                                    

def do_run(params):
    target, Wtarget, n, activation, W0, a0, activation_derivative, gamma, noise, predictor_interaction, seed, T, filename, choice_time_scaling, algo_method, spherical_flag, rho_prefactor = params
    if spherical_flag:
        from giant_learning.gradient_descent import SphericalGradientDescent as GDAlgo
        from giant_learning.gradient_descent import SphericalDisplacedSGD as DisplacedSGDAlgo
    else:
        from giant_learning.gradient_descent import GradientDescent as GDAlgo
        from giant_learning.gradient_descent import DisplacedSGD as DisplacedSGDAlgo
        from giant_learning.gradient_descent import ExponentialLossGradientDescent as ExpGDAlgo
    start = time.process_time()
    if algo_method == 'gd':
        algo = GDAlgo(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'skip', resample_every = 1, seed = seed, lazy_memory = True
        )
    elif algo_method == 'proxy_sam':
        algo = GDAlgo(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'skip', resample_every = 2, seed = seed, lazy_memory = True
        )
    elif algo_method == 'true_sam':
        algo = DisplacedSGDAlgo(
                    target, Wtarget, n,
                    activation, W0, a0, activation_derivative,
                    gamma, noise, rho_prefactor, predictor_interaction = not predictor_interaction,
                    test_size = None, analytical_error= 'skip', 
                    resample_every = 1, seed = seed, lazy_memory = True
                    )
    elif algo_method == 'no_resample':
        algo = GDAlgo(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise, predictor_interaction = not predictor_interaction,
            test_size = None, analytical_error= 'skip', resample_every = 0, seed = seed, lazy_memory = True
        )
    elif algo_method == 'exponential_gd':
        algo = ExpGDAlgo(
            target, Wtarget, n,
            activation, W0, a0, activation_derivative,
            gamma, noise,
            test_size = None, analytical_error= 'skip', resample_every = 1, seed = seed, lazy_memory = True
        )
    else:
        raise ValueError(f'algo_method = {algo_method} not known!')
    
    algo.train(T)
    np.savez(filename, test_errors = algo.test_errors, Ms = np.array(algo.Ms), Qs = np.array(algo.Qs), T = T, choice_time_scaling = choice_time_scaling)
    print(f'Elapsed time = {time.process_time() - start} -- algo = {algo_method} -- seed = {seed} -- d = {W0.shape[1]} -- p{algo.p}', flush = True)
    del algo
    gc.collect()

if __name__ == '__main__':
    import os
    os.environ["OMP_NUM_THREADS"] = str(parallel_procs)
    os.environ["MKL_NUM_THREADS"] = str(parallel_procs)
    os.environ["OPENBLAS_NUM_THREADS"] = str(parallel_procs)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(parallel_procs)
    os.environ["NUMEXPR_NUM_THREADS"] = str(parallel_procs)

    runner_params = []
    run_path(basepath, worker_run = True, runner_params = runner_params)
    print('Running on ', ncpu, ' cores')

    with mp.Pool(ncpu) as pool:
        pool.map(do_run, runner_params, chunksize=1)
