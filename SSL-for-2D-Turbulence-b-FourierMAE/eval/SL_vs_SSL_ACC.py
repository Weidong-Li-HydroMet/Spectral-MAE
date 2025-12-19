import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import yaml

import analyze_model


def main(config, ckpt_root, run_num, ckpts, leadtimes):
    """
    Main function to perform SL vs SSL analysis.
    """

    # Turn off long_analysis
    config['long_analysis_params']['save_data_length'] = 1
    config['long_analysis_params']['analysis_length'] = 1
    config['long_analysis_params']['temporal_mean'] = False
    config['long_analysis_params']['zonal_mean'] = False
    config['long_analysis_params']['zonal_eof'] = False
    config['long_analysis_params']['div'] = False
    config['long_analysis_params']['return_period'] = False
    config['long_analysis_params']['PDF'] = False
    config['long_analysis_params']['video'] = False

    # Don't want to overwrite existing short analysis results
    config['short_analysis_params']['save_short_analysis'] = False
    
    # SL performance
    print(f'\nSL analysis ...')

    results_SL = analyze_model.main(config)

    # SSL performance
    print(f'\nSSL analysis ...')

    results_SSL = {}
    config["dataset_params"]["root_dir"] = ckpt_root
    config["dataset_params"]["run_num"] = run_num

    for ckpt in ckpts:
        print(f'On epoch-ckpt: {ckpt}')

        # Set model filename in config
        config["dataset_params"]["model_filename"] = f'training_checkpoints/ckpt_{ckpt}.tar'
    
        results_SSL[ckpt] = analyze_model.main(config)

    # SL vs SSL comparison
    print(f'Plotting results ...')

    fig, axs = plt.subplots(nrows=2, ncols=2,
                            sharex=True, sharey=True)
    axs = axs.flatten()

    font = {'size': 14}
    mpl.rc('font', **font)

    for i, lead in enumerate(leadtimes):
        # Plot SL
        acc_median_SL = results_SL['acc_u_median'][lead] * np.ones(len(ckpts))
        acc_uq_SL = results_SL['acc_u_uq'][lead] * np.ones(len(ckpts))
        acc_lq_SL = results_SL['acc_u_lq'][lead] * np.ones(len(ckpts))
        axs[i].plot(ckpts, acc_median_SL, '-k')
        axs[i].fill_between(ckpts, acc_lq_SL, acc_uq_SL, color='k', alpha=0.1)

        # Plot SSL
        acc_median_SSL = [results_SSL[ckpt]['acc_u_median'][lead] for ckpt in ckpts]
        acc_uq_SSL = [results_SSL[ckpt]['acc_u_uq'][lead] for ckpt in ckpts]
        acc_lq_SSL = [results_SSL[ckpt]['acc_u_lq'][lead] for ckpt in ckpts]
        acc_SSL = [[acc_lq_SSL[j], acc_median_SSL[j], acc_uq_SSL[j]] for j in range(len(ckpts))]
        box_widths = 0.15 * ((ckpts[-1] - ckpts[0]) / len(ckpts))
        axs[i].boxplot(
                acc_SSL,
                positions=ckpts,
                vert=True,
                patch_artist=True,
                showmeans=False,
                notch=False,
                widths=box_widths
                )

        #axs[i].set_xlabel('Number of finetuing epochs')
        #axs[i].set_ylabel('ACC')
        axs[i].grid(True)
        x_pad = (0.2 / 0.15) * box_widths
        axs[i].set_xlim([ckpts[0]-x_pad, ckpts[-1]+x_pad])
        axs[i].set_ylim([0., 1.15])
        axs[i].set_title(f'{lead+1} {r"$\Delta t$"}', fontsize=8)
        axs[i].tick_params(axis='x', labelrotation=45)
        #axs[i].text(0, -1, f'{lead} {r"$\Delta t$"}', fontsize=10, 
        #            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
   
    fig.supxlabel('Number of finetuning epochs')
    fig.supylabel('ACC')
    fig.set_tight_layout(True)
    fig.savefig(f'SL_vs_SSL_ACC_U_{run_num}.svg')


if __name__ == "__main__":
    
    # Load configuration file for SL
    
    config_fp = str(sys.argv[1])
    with open("config/" + config_fp, "r") as f:
        config = yaml.safe_load(f)

    # Checkpoint parameters for SSL (assuming same config as for SL)

    ckpt_root = str(sys.argv[2])   # Directory for all model checkpoints
    run_num = str(sys.argv[3])   # Run num for SSL

    # List analysis parameters
    ckpts = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]   # List of epochs-checkpoints
    leadtimes = [0, 32, 65, 99]   # Prediction lead times for which to plot ACC


    main(config, ckpt_root, run_num, ckpts, leadtimes)
