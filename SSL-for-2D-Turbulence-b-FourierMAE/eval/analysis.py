# Test skill of trained model.
import sys
import torch
from torch import nn
import torch.distributed as dist
import os
import ast
from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
from py2d.initialize import initialize_wavenumbers_rfft2, gridgen
from py2d.derivative import derivative
from py2d.convert import Omega2Psi, Psi2UV
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from ruamel.yaml import YAML
import imageio

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(parent_dir)
sys.path.append(src_dir)
from src.models.vision_transformer import ViT
from src.utils.data_loaders import get_dataloader


# ================================================================================ #

def n_step_rollout(model, ic, n=1, train_tendencies=False):
    """Produce an n-step forward roll-out
    prediction.
    Args:
        model: trained pytorch model.
        ic: [B=1, C, T, X, Y] initial condition for prediction.
        n (int): number of steps to predict for.
    Returns:
        pred: [B=n, C, T, X, Y] n-step model prediction (time along dim=0)."""

    pred = []
    with torch.no_grad():
        idx = torch.tensor([0])
        if train_tendencies:
            # WARNING: if num_out_frames > 1, only top frame is kept in auto-regressive rollout
            if ic.shape[2] > 1:
                for i in range(n):
                    # Use index_select to prevent reducing along dim of size 1
                    pred_temp = torch.index_select(ic, 2, index=idx) + torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
            else:
                for i in range(n):
                    pred_temp = ic + torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = pred_temp
        else:
            if ic.shape[2] > 1:
                for i in range(n):
                    pred_temp = torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)          
                    ic = torch.cat([pred_temp, ic[:,:,:-1,:,:]], dim=2)
            else:
                for i in range(n):
                    pred_temp = torch.index_select(model(ic), 2, index=idx)
                    pred.append(pred_temp)
                    ic = pred_temp

    pred = torch.cat(pred, dim=0)

    return pred

def get_rmse(y, y_hat, climo=None):
    
    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    y_anom = y - climo
    y_hat_anom = y_hat - climo
    err = (y_anom - y_hat_anom) ** 2
    err = np.mean(err, axis=(-1, -2))
    rmse = np.sqrt(err)
    
    return rmse

def get_acc(y, y_hat, climo=None):
    """
    Args:
        y, y_hat: [B=n_steps, X, Y]
    """

    if climo is None:
        climo = np.zeros((y.shape[-2], y.shape[-1]))

    corr = []
    for i in range(y.shape[0]):
        y_i = y[i] - climo
        y_hat_i = y_hat[i] - climo
        #acc = (
        #        np.sum(y_i * y_hat_i) /
        #        np.sqrt(
        #            np.sum(y_i ** 2) * np.sum(y_hat_i ** 2)
        #            )
        #        )
        #corr.append(acc)
        corr.append(np.corrcoef(y_i.flatten(), y_hat_i.flatten())[1, 0])

    return np.array(corr)

def spectrum_zonal_average_2D_FHIT(U,V):
  """
  Zonal averaged spectrum for 2D flow variables

  Args:
    U: 2D square matrix, velocity
    V: 2D square matrix, velocity

  Returns:
    E_hat: 1D array
    wavenumber: 1D array
  """

  # Check input shape
  if U.ndim != 2 and V.ndim != 2:
    raise ValueError("Input flow variable is not 2D. Please input 2D matrix.")
  if U.shape[0] != U.shape[1] and V.shape[0] != V.shape[1]:
    raise ValueError("Dimension mismatch for flow variable. Flow variable should be a square matrix.")

  N_LES = U.shape[0]

  # fft of velocities along the first dimension
  U_hat = np.fft.rfft(U, axis=1)/ N_LES  #axis=1
  V_hat = np.fft.rfft(V, axis=1)/ N_LES  #axis=1

  # Energy
  #E_hat = 0.5 * U_hat * np.conj(U_hat) + 0.5 * V_hat * np.conj(V_hat)
  E_hat = U_hat

  # Average over the second dimension
  # Multiplying by 2 to account for the negative wavenumbers
  E_hat = np.mean(np.abs(E_hat)*2, axis=0) #axis=0
  wavenumbers = np.linspace(0, N_LES//2, N_LES//2+1)

  return E_hat, wavenumbers

def get_spectra(U, V):
    """
    Args:
        U, V: [B=n_steps, X, Y]
    Returns:
        spectra: [B=n_steps, k]
    """
    
    spectra = []
    for i in range(U.shape[0]):
        E_hat, wavenumbers = spectrum_zonal_average_2D_FHIT(U[i], V[i])
        spectra.append(E_hat)

    spectra = np.stack(spectra, axis=0)

    return spectra, wavenumbers

def get_zonal_PCA(zdata, n_comp=1):
    """
    Compute PCA of zonally-averaged fields.
    Args:
        data: [B=n_steps, X, Y] np.array of data
    Returns:
        pcs: [B, n_comp]
        eofs: [n_comp, X]
    """

    # Zonally average data
    zdata = np.mean(zdata, axis=-1) 
    print(f'zdata.shape: {zdata.shape}')

    # initiate PCA
    pca = PCA(n_components=n_comp)

    pcs = pca.fit_transform(zdata)      # [B, n_comp]
    eofs = pca.components_              # [n_comp, X]
    print(f'pcs.shape: {pcs.shape}')
    print(f'eofs.shape: {eofs.shape}')

    return pcs, eofs

def get_div(U, V):
    """
    Args:
        U: [B=n_steps, X, Y] 
        V: [B=n_steps, X, Y]
    Returns:
        div: [B,] divergence vs time
    """
   
    Lx, Ly = 2*np.pi, 2*np.pi
    Nx, Ny = U.shape[1], U.shape[2]
    Lx, Ly, X, Y, dx, dy = gridgen(Lx, Ly, Nx, Ny, INDEXING='ij')

    Kx, Ky, Kabs, Ksq, invKsq = initialize_wavenumbers_rfft2(Nx, Ny, Lx, Ly, INDEXING='ij')

    div = []
    for i in range(U.shape[0]):
        Dx = derivative(U[i,:,:], [0,1], Kx, Ky, spectral=False) #[1,0]
        Dy = derivative(V[i,:,:], [1,0], Kx, Ky, spectral=False) #[0,1]
        div.append(np.mean(np.abs(Dx + Dy)))

    return np.array(div)

def make_video(pred, tar):
    """
    Args:
        pred, tar: [B=n_steps, C, X, Y]
    """

    frames = []
    for t in range(pred.shape[0]):
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        axs = axs.flatten()
        data = [pred[t, 0, :, :], pred[t, 1, :, :], tar[t, 0, :, :], tar[t, 1, :, :]]
        titles = ['ML: U', 'ML: V', 'Truth: U', 'Truth: V']
        for i, ax in enumerate(axs):
            data_i = data[i] #.transpose((-1,-2))
            im = ax.imshow(data_i, cmap='bwr', vmin=-5, vmax=5, aspect='equal')
            xlen = data_i.shape[-1]
            ax.set_title(titles[i])
            ax.set_xticks([0, xlen/2, xlen], [0, r'$\pi$', r'$2\pi$']) 
            ax.set_yticks([0, xlen/2, xlen], [0, r'$\pi$', r'$2\pi$'])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.suptitle(rf'{t+1}$\Delta t$')
        fig.savefig('temp_frame.png', bbox_inches='tight')
        plt.close()

        frames.append(imageio.imread('temp_frame.png'))

    imageio.mimsave(f'Video_' + run_num + '.gif', frames, fps=5)

def perform_analysis(model, dataloader, dataloader_climo, dataloader_video, dataset, analysis_dict, params):
    """
    Returns:
        results: dictionary of {'rmse','rmse_per','acc','acc_per','spectra'}
    """

    climo_data, _ = next(iter(dataloader_climo))
    print(f'climo_data.shape: {climo_data.shape}')

    climo_data = climo_data.transpose(-1, -2).squeeze().detach().cpu().numpy()                     # [B=n_steps, C, X, Y]
    climo_u = climo_data[:,0].mean(axis=0)                                       # [X, Y]
    climo_v = climo_data[:,1].mean(axis=0) 
    climo_u_zonal = climo_data[:,0].mean(axis=(0,2))
    print(f'climo_u.shape: {climo_u.shape}')                                     # should be [X, Y]
    print(f'clim_u_zonal.shape: {climo_u_zonal.shape}')

    rmse_u, rmse_u_per = [], []
    rmse_v, rmse_v_per = [], []
    acc_u, acc_u_per = [], []
    acc_v, acc_v_per = [], []
    spectra, spectra_tar, wavenumbers = [], [], []
    for i, batch in enumerate(dataloader):

        print(f'Prediction iteration: {i}\n')
        inputs, targets = batch
        ic = inputs[0].unsqueeze(dim=0)
        print(f'ic.shape: {ic.shape}')
        n_steps = inputs.shape[0]

        pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=params["train_tendencies"])
        per_pred = inputs[0].repeat(n_steps, 1, 1, 1, 1)[:,:,0,:,:]

        print(f'pred.shape: {pred.shape}')                                      # should be: [B=n_steps, C, T=1, X, Y]
        print(f'per_pred.shape: {per_pred.shape}')
        print(f'targets.shape: {targets.shape}')

        pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()                            # [B=n_steps, C, X, Y]
        per_pred = per_pred.transpose(-1,-2).squeeze().detach().cpu().numpy()  
        targets = targets.transpose(-1,-2).squeeze().detach().cpu().numpy()

        pred_u = pred[:,0]                                                      # [B=n_steps, X, Y]
        pred_v = pred[:,1]
        per_pred_u = per_pred[:,0]
        per_pred_v = per_pred[:,1]
        tar_u = targets[:,0]
        tar_v = targets[:,1]

        # Unnormalize data
        pred_u = (pred_u * dataset.input_std[0]) + dataset.input_mean[0]
        pred_v = (pred_v * dataset.input_std[1]) + dataset.input_mean[1]
        tar_u = (tar_u * dataset.label_std[0]) + dataset.label_mean[0]
        tar_v = (tar_v * dataset.label_std[1]) + dataset.label_mean[1]


        if analysis_dict['rmse']:
            rmse_u.append(get_rmse(tar_u, pred_u, climo=climo_u))
            rmse_u_per.append(get_rmse(tar_u, per_pred_u, climo=climo_u))
            rmse_v.append(get_rmse(tar_v, pred_v, climo=climo_v))
            rmse_v_per.append(get_rmse(tar_v, per_pred_v, climo=climo_v))
            print(f'rmse_u[-1].shape: {rmse_u[-1].shape}\n')

        if analysis_dict['acc']:
            acc_u.append(get_acc(tar_u, pred_u, climo_u))                       # [B=n_steps,]
            acc_u_per.append(get_acc(tar_u, per_pred_u, climo_u))
            acc_v.append(get_acc(tar_v, pred_v, climo_v))
            acc_v_per.append(get_acc(tar_v, per_pred_v, climo_v))
            print(f'acc_u[-1].shape: {acc_u[-1].shape}\n')

        if analysis_dict['spectra']:
            spectra_temp, wavenumbers = get_spectra(pred_u, pred_v)             # [B=n_steps, k]
            spectra_tar_temp, _ = get_spectra(tar_u, tar_v)
            spectra.append(spectra_temp)
            spectra_tar.append(spectra_tar_temp)
            print(f'spectra[-1].shape: {spectra[-1].shape}\n')

    # Average over all predictions and Save results
    results = {}
    if analysis_dict['rmse']:
        results['rmse_u_median'] = np.quantile(np.stack(rmse_u, axis=0), 0.5, axis=0)
        results['rmse_u_uq'] = np.quantile(np.stack(rmse_u, axis=0), 0.75, axis=0)
        results['rmse_u_lq'] = np.quantile(np.stack(rmse_u, axis=0), 0.25, axis=0)
        results['rmse_u_per_median'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.5, axis=0)
        results['rmse_u_per_uq'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.75, axis=0)
        results['rmse_u_per_lq'] = np.quantile(np.stack(rmse_u_per, axis=0), 0.25, axis=0)
        results['rmse_v_median'] = np.quantile(np.stack(rmse_v, axis=0), 0.5, axis=0)
        results['rmse_v_uq'] = np.quantile(np.stack(rmse_v, axis=0), 0.75, axis=0)
        results['rmse_v_lq'] = np.quantile(np.stack(rmse_v, axis=0), 0.25, axis=0)
        results['rmse_v_per_median'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.5, axis=0)
        results['rmse_v_per_uq'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.75, axis=0)
        results['rmse_v_per_lq'] = np.quantile(np.stack(rmse_v_per, axis=0), 0.25, axis=0)

        results['rmse_u_mean'] = np.mean(np.stack(rmse_u, axis=0), axis=0)
        results['rmse_u_std'] = np.std(np.stack(rmse_u, axis=0), axis=0)
        results['rmse_u_per_mean'] = np.mean(np.stack(rmse_u_per, axis=0), axis=0)
        results['rmse_u_per_std'] = np.std(np.stack(rmse_u_per, axis=0), axis=0)
        results['rmse_v_mean'] = np.mean(np.stack(rmse_v, axis=0), axis=0)
        results['rmse_v_std'] = np.std(np.stack(rmse_v, axis=0), axis=0)
        results['rmse_v_per_mean'] = np.mean(np.stack(rmse_v_per, axis=0), axis=0)
        results['rmse_v_per_std'] = np.std(np.stack(rmse_v_per, axis=0), axis=0)
    if analysis_dict['acc']:
        results['acc_u_median'] = np.quantile(np.stack(acc_u, axis=0), 0.5, axis=0)
        results['acc_u_uq'] = np.quantile(np.stack(acc_u, axis=0), 0.75, axis=0)
        results['acc_u_lq'] = np.quantile(np.stack(acc_u, axis=0), 0.25, axis=0)
        results['acc_u_per_median'] = np.quantile(np.stack(acc_u_per, axis=0), 0.5, axis=0)
        results['acc_u_per_uq'] = np.quantile(np.stack(acc_u_per, axis=0), 0.75, axis=0)
        results['acc_u_per_lq'] = np.quantile(np.stack(acc_u_per, axis=0), 0.25, axis=0)
        results['acc_v_median'] = np.quantile(np.stack(acc_v, axis=0), 0.5, axis=0)
        results['acc_v_uq'] = np.quantile(np.stack(acc_v, axis=0), 0.75, axis=0)
        results['acc_v_lq'] = np.quantile(np.stack(acc_v, axis=0), 0.25, axis=0)
        results['acc_v_per_median'] = np.quantile(np.stack(acc_v_per, axis=0), 0.5, axis=0)
        results['acc_v_per_uq'] = np.quantile(np.stack(acc_v_per, axis=0), 0.75, axis=0)
        results['acc_v_per_lq'] = np.quantile(np.stack(acc_v_per, axis=0), 0.25, axis=0)

        results['acc_u_mean'] = np.mean(np.stack(acc_u, axis=0), axis=0)
        results['acc_u_std'] = np.std(np.stack(acc_u, axis=0), axis=0)
        results['acc_u_per_mean'] = np.mean(np.stack(acc_u_per, axis=0), axis=0)
        results['acc_u_per_std'] = np.std(np.stack(acc_u_per, axis=0), axis=0)
        results['acc_v_mean'] = np.mean(np.stack(acc_v, axis=0), axis=0)
        results['acc_v_std'] = np.std(np.stack(acc_v, axis=0), axis=0)
        results['acc_v_per_mean'] = np.mean(np.stack(acc_v_per, axis=0), axis=0)
        results['acc_v_per_std'] = np.std(np.stack(acc_v_per, axis=0), axis=0)
    if analysis_dict['spectra']:
        results['spectra'] = np.mean(np.stack(spectra, axis=0), axis=0)          # [B=n_steps, k]
        results['spectra_tar'] = np.mean(np.stack(spectra_tar, axis=0), axis=0)
        results['wavenumbers'] = wavenumbers

    long_analysis = any([analysis_dict['video'], analysis_dict['zonal_pca'], analysis_dict['div']])
    if long_analysis:
        inp, tar = next(iter(dataloader_video))
        ic = inp[0].unsqueeze(dim=0)
        n_steps = inp.shape[0]

        pred = n_step_rollout(model, ic, n=n_steps, train_tendencies=params["train_tendencies"])
        #pred = pred[1:]

        pred = pred.transpose(-1,-2).squeeze().detach().cpu().numpy()
        tar = tar.transpose(-1,-2).squeeze().detach().cpu().numpy()

        # Unnormalize data
        pred[:,0] = (pred[:,0] * dataset.input_std[0]) + dataset.input_mean[0]
        pred[:,1] = (pred[:,1] * dataset.input_std[1]) + dataset.input_mean[1]
        tar[:,0] = (tar[:,0] * dataset.label_std[0]) + dataset.label_mean[0]
        tar[:,1] = (tar[:,1] * dataset.label_std[1]) + dataset.label_mean[1]

        pred_u = pred[:,0]
        pred_v = pred[:,1]
        tar_u = tar[:,0]
        tar_v = tar[:,1]

        if analysis_dict['video']:
            print(f'Making long roll-out video.')
            make_video(pred, tar)

        if analysis_dict['zonal_pca']:

            pred_u_zonal = np.mean(pred_u, axis=-1) - climo_u_zonal
            tar_u_zonal = np.mean(tar_u, axis=-1) - climo_u_zonal

            pred_u_pc, pred_u_eof = get_zonal_PCA(pred_u,
                                                    n_comp=analysis_dict['pca_ncomp'])
            tar_u_pc, tar_u_eof = get_zonal_PCA(tar_u,
                                                    n_comp=analysis_dict['pca_ncomp'])
            print(f'pred_u_pc.shape: {pred_u_pc.shape}\n')
            print(f'pred_u_eof.shape: {pred_u_eof.shape}\n')
            
            results['pred_u_pc'] = pred_u_pc
            results['pred_u_eof'] = pred_u_eof
            results['tar_u_pc'] = tar_u_pc
            results['tar_u_eof'] = tar_u_eof

        if analysis_dict['div']:
            print(f'div. tar_u.shape, tar_v.shape: {tar_u.shape}, {tar_v.shape}\n')
            print(f'div. pred_u.shape, pred_v.shape: {pred_u.shape}. {pred_v.shape}\n')

            # Rescale preds/tars for div calculation
            pred_div = get_div(pred_u, pred_v)
            tar_div = get_div(tar_u, tar_v)

            results['pred_div'] = pred_div
            results['tar_div'] = tar_div

    return results

def plot_analysis(results, analysis_dict):

    font = {'size': 14}
    mpl.rc('font', **font)

    if analysis_dict['rmse']:
        # U
        fig, ax = plt.subplots()
        x = np.arange(1, 1+len(results['rmse_u_median'])) 
        ax.plot(x, results['rmse_u_median'], '-k', label='ML')
        upper = results['rmse_u_uq'] # + results['rmse_u_std']
        lower = results['rmse_u_lq'] # - results['rmse_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_u_per_median'], '--k', label='Persistence')
        upper = results['rmse_u_per_uq'] # + results['rmse_u_per_std']
        lower = results['rmse_u_per_lq'] # - results['rmse_u_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, 100])
        ax.legend()
        plt.tight_layout()
        fig.savefig('RMSE_U_' + run_num + '.svg')
        # V
        fig, ax = plt.subplots()
        ax.plot(x, results['rmse_v_median'], '-k', label='ML')
        upper = results['rmse_v_uq'] # + results['rmse_v_std']
        lower = results['rmse_v_lq'] # - results['rmse_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['rmse_v_per_median'], '--k', label='Persistence')
        upper = results['rmse_v_per_uq'] # + results['rmse_v_per_std']
        lower = results['rmse_v_per_lq'] # - results['rmse_v_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('RMSE')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([0, 3.5])
        ax.set_xlim([0, 100])
        ax.legend()
        plt.tight_layout()
        fig.savefig('RMSE_V_' + run_num + '.svg')

    if analysis_dict['acc']:
        # U
        fig, ax = plt.subplots()
        x = np.arange(1, 1+len(results['acc_u_median'])) 
        ax.plot(x, results['acc_u_median'], '-k', label='ML')
        upper = results['acc_u_uq'] # + results['acc_u_std']
        lower = results['acc_u_lq'] # - results['acc_u_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_u_per_median'], '--k', label='Persistence')
        upper = results['acc_u_per_uq'] # + results['acc_u_per_std']
        lower = results['acc_u_per_lq'] # - results['acc_u_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 100])
        ax.legend()
        plt.tight_layout()
        fig.savefig('ACC_U_' + run_num + '.svg')
        # V
        fig, ax = plt.subplots()
        ax.plot(x, results['acc_v_median'], '-k', label='ML')
        upper = results['acc_v_uq'] # + results['acc_v_std']
        lower = results['acc_v_lq'] # - results['acc_v_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.plot(x, results['acc_v_per_median'], '--k', label='Persistence')
        upper = results['acc_v_per_uq'] # + results['acc_v_per_std']
        lower = results['acc_v_per_lq'] # - results['acc_v_per_std']
        ax.fill_between(x, lower, upper, color='k', alpha=0.1)
        ax.set_ylabel('ACC')
        ax.set_xlabel(rf'Lead time ($\Delta t$)')
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 100])
        ax.legend()
        plt.tight_layout()
        fig.savefig('ACC_V_' + run_num + '.svg')

    if analysis_dict['spectra']:
        fig, ax = plt.subplots()
        x = results['wavenumbers']
        ax.plot(x, results['spectra_tar'][0], '-k', label='Truth')
        for lead in analysis_dict['spectra_leadtimes']:
            spec = results['spectra'][lead]
            label = rf'{lead+1}$\Delta t$' 
            ax.plot(x, spec, label=label)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Wavenumbers')
            ax.set_ylabel('Power')
            ax.set_xlim([0.8, 200])
            ax.set_ylim([10**(-9), 10])
            ax.legend()
            plt.tight_layout()
            fig.savefig('Power_Spectra_' + run_num + '.svg')

    if analysis_dict['zonal_pca']:
        # Plot EOFs
        pred_u_pcs = results['pred_u_pc']
        pred_u_eofs = results['pred_u_eof']
        tar_u_pcs = results['tar_u_pc']
        tar_u_eofs = results['tar_u_eof']
        eofs = [pred_u_eofs, tar_u_eofs]
        colors = ['k', 'r', 'b', 'g']
        x = np.linspace(0, 2*np.pi, pred_u_eofs.shape[1])
        for i in range(pred_u_eofs.shape[0]):
            fig, ax = plt.subplots()
            ax.plot(pred_u_eofs[i, :], x, f'--{colors[i]}', label=f'ML EOF{i+1}')
            ax.plot(tar_u_eofs[i, :], x, f'-{colors[i]}', label=f'Truth EOF{i+1}')
            ax.set_xlim([-0.25, 0.25])
            ax.set_ylabel('x')
            ax.set_title(f'EOF{i+1} of zonally-averaged U')
            ax.legend()
            plt.tight_layout()
            fig.savefig(f'EOF{i+1}_' + run_num + '.svg')

    if analysis_dict['div']:
        fig, ax = plt.subplots()
        x = np.arange(1, 1+results['pred_div'].shape[0])
        ax.plot(x, results['pred_div'], '--k', label='ML')
        ax.plot(x, results['tar_div'], '-k', label='Truth')
        ax.set_xlabel('ML timestep')
        #ax.set_ylim([-1, 1])
        ax.legend()
        plt.tight_layout()
        fig.savefig('Div_' + run_num + '.svg')

def main(root_dir, model_filename, params_filename, test_length, num_tests, test_file_start_idx,
        analysis_dict, test_length_climo=10000, test_length_video=500):

    # Read in params file
    params_fp = os.path.join(root_dir, params_filename)
    yaml = YAML(typ='safe')
    with open(params_fp, 'r') as f:
        params_temp = yaml.load(f)
    
    params = {}
    for key, val in params_temp.items():
        try:
            params[key] = ast.literal_eval(val)
        except:
            params[key] = val
    print(f'params: {params}\n')

    # Initiate model and load weights
    model_fp = os.path.join(root_dir, model_filename)
    model = ViT(
        img_size=params["img_size"],
        patch_size=params["patch_size"],
        num_frames=params["num_frames"],
        tubelet_size=params["tubelet_size"],
        in_chans=params["in_chans"],
        encoder_embed_dim=params["encoder_embed_dim"],
        encoder_depth=params["encoder_depth"],
        encoder_num_heads=params["encoder_num_heads"],
        decoder_embed_dim=params["decoder_embed_dim"],
        decoder_depth=params["decoder_depth"],
        decoder_num_heads=params["decoder_num_heads"],
        mlp_ratio=params["mlp_ratio"],
        num_out_frames=params["num_out_frames"],
        patch_recovery=params["patch_recovery"]
        )
    ckpt_temp = torch.load(model_fp, map_location=torch.device('cpu'))['model_state']
    ckpt = {}
    for key, val in ckpt_temp.items():
        key_new = key[7:]
        ckpt[key_new] = val
    
    model.load_state_dict(ckpt)
    model.eval()

    # Initiate dataloaders
    test_file_range = (test_file_start_idx, test_file_start_idx+(test_length*num_tests*params["target_step"])-1) #+params["target_step"])
    dataloader, dataset = get_dataloader(data_dir=params["data_dir"],
                                    file_range=test_file_range,
                                    target_step=params["target_step"],
                                    train_tendencies=params["train_tendencies"],
                                    batch_size=test_length,
                                    train=False,
                                    stride=params["target_step"],
                                    distributed=dist.is_initialized(),
                                    num_frames=params["num_frames"],
                                    num_out_frames=1, #params["num_out_frames"],
                                    num_workers=2,
                                    pin_memory=params["pin_memory"])
    test_file_range = (test_file_start_idx, test_file_start_idx+test_length_climo)
    dataloader_climo, dataset_climo = get_dataloader(data_dir=params["data_dir"],
                                    file_range=test_file_range,
                                    target_step=1,
                                    train_tendencies=params["train_tendencies"],
                                    batch_size=test_length_climo,
                                    train=False,
                                    stride=1,
                                    distributed=dist.is_initialized(),
                                    num_frames=1, #params["num_frames"],
                                    num_out_frames=1, #params["num_out_frames"],
                                    num_workers=2,
                                    pin_memory=params["pin_memory"])
    test_file_range = (test_file_start_idx, test_file_start_idx+(test_length_video*params["target_step"])+params["target_step"])
    dataloader_video, datasaet_video = get_dataloader(data_dir=params["data_dir"],
                                    file_range=test_file_range,
                                    target_step=params["target_step"],
                                    train_tendencies=params["train_tendencies"],
                                    batch_size=test_length_video,
                                    train=False,
                                    stride=params["target_step"],
                                    distributed=dist.is_initialized(),
                                    num_frames=params["num_frames"],
                                    num_out_frames=1, #params["num_out_frames"],
                                    num_workers=2,
                                    pin_memory=params["pin_memory"])

    print(f'len(dataset): {len(dataset)}')
    print(f'len(dataset_climo): {len(dataset_climo)}')
    print(f'len(dataset_video): {len(datasaet_video)}')

    results = perform_analysis(model, dataloader, dataloader_climo, dataloader_video, dataset, analysis_dict, params)

    plot_analysis(results, analysis_dict)


# ================================================================================ #

# File Paths
root_dir = '/scratch/user/u.dp200518/SSL-2DTurb/MAE_FINETUNE/xxx1/'
model_filename = 'training_checkpoints/best_ckpt.tar' #ckpt_200.tar best_ckpt.tar'
params_filename = 'hyperparams.yaml'
run_num = 'mae_ft_xxx1'

# Test Parameters
test_length = 100   # will be batch size
num_tests = 10
test_file_start_idx = 350000 

test_length_climo = 10000

test_length_video = 1000

pca_ncomp = 2

# Analysis
analysis_dict = {
        'rmse': True,
        'acc': True,
        'spectra': True,
        'spectra_leadtimes': [0, 4, 9, 39, 49],
        'zonal_pca': False,
        'pca_ncomp': pca_ncomp,
        'video': False,
        'div': False,
        'long_rollout_length': test_length_video
        }

main(root_dir, model_filename, params_filename, test_length, num_tests, test_file_start_idx,
        analysis_dict, test_length_climo=test_length_climo, test_length_video=test_length_video) 
