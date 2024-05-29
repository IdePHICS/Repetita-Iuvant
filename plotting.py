import numpy as np 
import matplotlib.pyplot as plt
import os
from yaml import safe_load as load
import warnings
from tqdm import tqdm

def main(argc, argv):
    index_flag = "MultiIndex"
    algo_flag = "Plain"
    batch_flag = "BatchOne"
    new_flag = "Reusing"
    last_flag = "Fig2b"
    filepath = f'hyperparameters/{index_flag}/{algo_flag}/{batch_flag}/{new_flag}/{last_flag}.yaml' # This is the date and hour of the hyperparameters file

    filepath = 'hyperparameters/definitive/fig_staircase/SQStairH4_long.yaml'
    prefactor_index = 0
    rho_prefactor_index = 0
    p_index = 0

    if argc > 1:
        filepath = argv[1]
        if argc > 3:
            prefactor_index = int(argv[2])
            rho_prefactor_index = int(argv[3])

    plot_from_hyperparams(
        filepath,
        prefactor_index = prefactor_index,
        rho_prefactor_index = rho_prefactor_index,
        p_index = p_index,
        what_to_plot = 'cosine_similarity',
        how_to_plot = 'averaged',
        plotting_CT = None,
        save_flag = True,
        show_flag = False,
        plot_random_lines = True,
        joint_plot = False,
        latex_flag = False,
        debug = True,
        figname = None,
        figtype = 'png',
        computation_database_path = 'computation-database-definitive/'
    )   

paths_to_folders = {
    'gd': '/gd_runs/gd_run', 
    'proxy_sam': '/proxy_sam_runs/sam_run', 
    'true_sam': '/sam_runs/sam_run', 
    'no_resample': '/no_resample_runs/no_resample_run',
    'exponential_gd': 'exponential_gd/exponential_gd'
} 
colors = ['mediumblue', 'orangered', 'magenta', 'mediumseagreen', 'c', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'olive', 'cyan']
markers = ['o', 'x', 's', 'D', '^', 'v', '>', '<', 'p', 'h', 'H', 'd', '*', '+', '|', '_']
latex_time_scaling = {
    'kstar3': 'd^2',
    'multiple_pass_easy': 'd',
    'dlogd': 'd\log d'
}
legend_algo_method = {
    'gd': 'SGD',
    'true_sam': 'EgD',
    'proxy_sam': 'LaD $k=2$',
    'no_resample': 'GD',
    'exponential_gd': 'Exp. Loss SGD'
}

target_beutystring = {
    'sqstairsH4_enchanted': r'$h^*_{stair}$',
    'sign(z1z2z3)_enchanted': r'$h^*_{sign}$',
    'sqstairs_enchanted': r'$h^*_{stair}$'
}

def plot_from_hyperparams(
        filepath,                           # configuration file
        prefactor_index = 0,                # index of the prefactor to plot
        rho_prefactor_index = 0,            # index of the rho_prefactor to plot
        p_index = 0,                        # index of the p to plot
        what_to_plot = 'cosine_similarity', # 'magnetization' or 'cosine_similarity'
        how_to_plot = 'averaged',           # 'averaged' or 'non_averaged'
        plotting_CT = None,                 # number of steps to plot
        save_flag = True,                   # save the figure on a file 
        show_flag = False,                  # show the figure as a pop-up
        plot_random_lines = True,           # plot the random performance lines for comparison
        plotting_dimensions = 'all',        # 'all' or 'last', choose which d to plot
        joint_plot = False,                 # plot all the k lines on the same plot
        latex_flag = True,                  # use latex for the text
        debug = False,                      # print debug information under and over the plot
        figname = None,                     # name of the figure. If None, it is automatically generated and place in computation-database/figures
        figtype = 'pdf',                    # file type for the figure: 'pdf' or 'png'
        figsize = 5,                        # size in inch of the figure
        singletarget = 0,                   # index of the single target to be plotted. If -1, all targets are used
        ylabel = True,                      # plot the ylabel
        legend = 'yes',                     # 'yes', 'no', 'only[r]': 'yes' and no are 'clear'. 'only[r]' means only the r-th axis is used.
        computation_database_path = 'computation-database/' # path to the computation database
    ):
    flag = str(filepath).split(sep="/")[-1].removesuffix(".yaml")
    yaml_file = open(filepath, 'r')
    hyperparams = load(yaml_file)
    p = hyperparams['ps'][p_index]
    student_activation_choice = hyperparams['student_activation_choices'][0]
    choice_gamma = hyperparams['choice_gammas'][0]
    batch_size_choice = hyperparams['batch_size_choices'][0]
    k = hyperparams['k']
    noise = hyperparams['noise']
    predictor_interaction = hyperparams['predictor_interaction']
    nseeds = hyperparams['nseeds'] 
    spherical_flag = hyperparams['spherical']
    prefactor = hyperparams['prefactors'][prefactor_index]
    coefficient_time = hyperparams['coefficient_time']
    plotting_CT = coefficient_time if plotting_CT == None else plotting_CT
    ds = hyperparams['ds']
    if plotting_dimensions == 'last':
        ds = [ds[-1]]
    choice_a = hyperparams['choice_a']
    choice_time_scaling = hyperparams['choice_time_scaling']
    choice_init = hyperparams['choice_init']
    rho_prefactor = hyperparams['rho_prefactors'][rho_prefactor_index]
    algo_methods = hyperparams['algo_methods']
    try:
        target_choices = hyperparams['teacher_activation_choices']
    except:
        target_choices = hyperparams['target_choices']
    if singletarget != -1:
        target_choices = [target_choices[singletarget]]
    try:
        M_rotation = np.array(hyperparams['M_rotation'])
        assert(len(M_rotation.shape) == 2)
    except KeyError:
        M_rotation = np.eye(k)

    if legend.startswith('only'):
        only_legend = int(legend[4:])
    else:
        only_legend = None
    if plotting_CT == None:
        plotting_CT = coefficient_time
    reducing_T_factor = plotting_CT / coefficient_time
    ### LOAD & PLOT ###
    plt.rcParams['text.usetex'] = latex_flag

    if joint_plot:
        fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(figsize,figsize))
        axes = [ax for _ in range(k)]
    else:
        fig, axes = plt.subplots(nrows=1,ncols=k,figsize=(figsize*k,figsize))
        if k == 1:
            axes = [axes]
    for target_index, target_choice in enumerate(target_choices):
        for algo_index, algo_method in enumerate(algo_methods):
            for dimension_index, d in enumerate(ds):
                print(f'Plotting algo = {algo_method} -- d = {d}')
                batch_sizes_fractions = {'one': d, 'tenth': 10, 'fifth': 5, 'half': 2, 'full': 1,  'double': 0.5}
                coefficient_batch_size = batch_sizes_fractions[batch_size_choice]
                n = int(d/coefficient_batch_size)
                time_scaling_coefficients = {'kstar3': n/d**2, 'multiple_pass_easy': n/d, 'dlogd': n/(d*np.log(d))}
                Ms = []
                Qs = []
                T = None
                skip_factor = None
                time_scaling = None
                time_scaling_coefficient = time_scaling_coefficients[str(choice_time_scaling)]
                for seed in tqdm(range(nseeds), mininterval=2):
                    path_to_folder = computation_database_path + paths_to_folders[algo_method]
                    rhoflag = rho_prefactor if algo_method == 'true_sam' else 'None'
                    filename = f'{path_to_folder}_(name={flag})d={d}_p={p}_choice_gamma={prefactor}_{choice_gamma}_teacher={target_choice}_student={student_activation_choice}_batch_size={batch_size_choice}_choice_a={choice_a}_seed={seed}_init={choice_init}_k={k}_spherical={spherical_flag}_rhopref={rhoflag}_CT={coefficient_time}.npz'
                    if os.path.exists(filename):
                        try:
                            data = np.load(filename, allow_pickle=True)
                        except:
                            print(f'{filename} unreadable!')
                            continue

                        if T == None:
                            time_scaling = data['choice_time_scaling']
                            T = data['T']
                            plotting_T = int(T*reducing_T_factor)
                            skip_factor = max(plotting_T // (6*figsize),1)
                        else:
                            assert(time_scaling == data['choice_time_scaling'])
                            assert(T == data['T'])

                        Ms.append(data['Ms'][:plotting_T:skip_factor,:,:])
                        Qs.append(data['Qs'][:plotting_T:skip_factor,:,:])
                if len(Ms) == 0:
                    warnings.warn(f'No files found for d = {d} -- algo = {algo_method}\nLast Tried: {filename}')
                    continue
                Ms = np.array(Ms)
                Qs = np.array(Qs)

                xaxis = np.arange(plotting_T+1)[::skip_factor] * time_scaling_coefficient
                assert(xaxis.shape[0] == Ms.shape[1])

                Ms = np.dot(
                    Ms, M_rotation
                )
                if what_to_plot == 'magnetization':
                    yaxis = Ms[:,:,:,:] # shape(n_seeds, skippedT, p, k)
                else:
                    # Normalize Ms by its norm
                    Qdiag = np.diagonal(Qs, axis1=-2, axis2=-1)
                    CosSim= np.einsum(
                        '...tjk, ...tj -> ...tjk', Ms, 1/np.sqrt(Qdiag)
                    )
                    yaxis = CosSim[:,:,:,:] # shape(n_seeds, skippedT, p, k)
                skippedT = yaxis.shape[-3]
                yaxis = abs(yaxis) # take the abs for having the best between +/- a_i
                yaxis = np.max(yaxis, axis=-2) # shape(n_seeds, skippedT, k)
                # assert(yaxis.shape == (nseeds, skippedT, k))
                yaxis = np.swapaxes(yaxis, 0, 2) # shape (k, skippedT, n_seeds)
                
                marker_index = (algo_index+target_index)*(algo_index+target_index+1)//2 + target_index
                for r in range(k):
                    color_index = (dimension_index+r)*(dimension_index+r+1)//2 + dimension_index
                    if how_to_plot == 'non_averaged':
                        axes[r].plot(
                            xaxis[:], yaxis[r,:],
                            color=colors[color_index],
                            marker = markers[marker_index],
                            ls = '-'
                        )
                    elif how_to_plot == 'averaged':
                        yaxis_mean = np.mean(yaxis[r], axis=-1)
                        yaxis_err = np.std(yaxis[r], axis=-1)/np.sqrt(yaxis.shape[-1])
                        # yaxis_err = np.sqrt(yaxis_mean*(1-yaxis_mean)/yaxis.shape[-1])
                        axes[r].errorbar(
                            x = xaxis, y = yaxis_mean, yerr = yaxis_err,
                            capsize=3,
                            color=colors[color_index],
                            marker = markers[marker_index],
                            ls = ''
                        )

                    # Empty plot for legend
                    if (joint_plot or r==0) and (only_legend == None or only_legend == r):
                        label = ''
                        if singletarget == -1:
                            label += f'{target_beutystring[target_choice]} '
                        if plotting_dimensions != 'last':
                            label += f'$d={d}$ '
                        label += f'{legend_algo_method[algo_method]}'
                        if joint_plot:
                            label += f' $w^*_{r+1}$' 
                        if how_to_plot == 'non_averaged':
                            # label += r' $n_\\text{seeds} = '+f'{nseeds}$'
                            pass
                        axes[r].plot([],[],
                            color=colors[color_index],
                            marker = markers[marker_index],
                            label=label)
    if debug:
        axes[0].set_title(f'teacher={target_choice} - student={student_activation_choice} - how_to_plot = {how_to_plot} - p={p} - batch_size={batch_size_choice} - rho = {rho_prefactor} - gamma = {prefactor}')
        axes[-1].set_xlabel(f'Normalized  Steps -- time scaling = {choice_time_scaling} -- skip factor = {skip_factor}')
     
    for r in range(k):
        # ylabel
        if ylabel:
            if p == 1:
                starting_label = '$'
            else:
                starting_label = '$max_j \,\,\, '
            if joint_plot:
                if what_to_plot == 'magnetization':
                    axes[r].set_ylabel(starting_label+r'\,\,\, M_{j,r}$')
                elif what_to_plot == 'cosine_similarity':
                    axes[r].set_ylabel(starting_label+r'CosSim_(w_j,w^*_{r})$')
            else:
                if what_to_plot == 'magnetization':
                    axes[r].set_ylabel(starting_label+r'M_{j,'+f'{r+1}'+'}$')
                elif what_to_plot == 'cosine_similarity':
                    axes[r].set_ylabel(starting_label+r'CosSim_(w_j,w^*_{'+f'{r+1}'+'})$')
        xlabel = '$T'
        if n > 1:
            xlabel += r'\cdot n_b'
        xlabel += f'/{latex_time_scaling[choice_time_scaling]}$'
        axes[r].set_xlabel(xlabel)
        if plot_random_lines: 
            for dimension_index, d in enumerate(ds):
                dimension_index = (dimension_index+r)*(dimension_index+r+1)//2 + dimension_index
                if plotting_dimensions == 'last':
                    if r > 0 and joint_plot:
                        continue
                    color = 'black'
                    label = r'$\frac{1}{\sqrt{d}}$'
                else:
                    color = colors[dimension_index]
                    label = r'Rand. Perf. $\frac{1}{\sqrt{'+str(d)+'}}$'

                if only_legend == None:
                    axes[r].axhline(y = 1/np.sqrt(d), ls='--', color = color, label=label)
                else:
                    axes[r].axhline(y = 1/np.sqrt(d), ls='--', color = color)

        if legend != 'no':
            if r == 0 or joint_plot == False:
                axes[r].legend()
    ### SAVE ###
    if save_flag:
        default_fig_name = f'computation-database/figures/({flag})algos{algo_methods}_p{p}_ds{ds}_T{plotting_CT}_gamma{prefactor}_{choice_gamma}_teacher{target_choice}_student{student_activation_choice}_batch_size{batch_size_choice}_choice_a{choice_a}_init{choice_init}_k{k}_spherical{spherical_flag}_rhopref{rho_prefactor}_plot{what_to_plot}_{how_to_plot}'
        if figname == None:
            figname = default_fig_name
        fig.savefig(
            figname+'.'+figtype,
            format=figtype,
            dpi=300,
            bbox_inches='tight'
        )
    if show_flag:
        plt.show()

if __name__ == '__main__':
    import sys
    main(len(sys.argv), sys.argv)