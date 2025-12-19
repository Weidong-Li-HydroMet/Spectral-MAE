import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize


def grad_norm(model):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def grad_max(model):
    max_grad = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_max = torch.max(torch.abs(p.grad.detach().data))
        if max_grad < param_max.item():
            max_grad = param_max.item()
    return param_max


def apply_colormap(img, cmap='bwr'):
    """
    Return PIL Image from input img (np.array) after applying colormap cmap.
    """

    norm = Normalize(vmin=-3, vmax=3)
    normed_img = norm(img.T)

    return plt.cm.get_cmap(cmap)(normed_img)


def log_input_target_prediction(input, target, prediction, table, epoch, batch_members=[0, -1]):
    """
    Log input, target and prediction images into a wandb table.
    input, target, prediction shapes: [b, c, t, h, w]
    batch_members: list of batch indices to select batch members to log
    """

    # Get first and last batch members
    input = input[batch_members, :, 0].cpu().numpy()
    target = target[batch_members, :, 0].cpu().numpy()
    prediction = prediction[batch_members, :, 0].cpu().numpy()

    # Separate into U and V
    input_u, input_v = input[:, 0], input[:, 1]
    target_u, target_v = target[:, 0], target[:, 1]
    prediction_u, prediction_v = prediction[:, 0], prediction[:, 1]

    for j, (u, v) in enumerate(zip(zip(input_u, target_u, prediction_u), zip(input_v, target_v, prediction_v))):
        
        # Add U
        i, t, p = u
        img_id = f'epoch_{epoch} bm_{batch_members[j]}: U'
        table.add_data(img_id, 
                           wandb.Image(apply_colormap(i)), 
                           wandb.Image(apply_colormap(t)), 
                           wandb.Image(apply_colormap(p)),
                           wandb.Image(apply_colormap(t) - apply_colormap(p)))

        # Add V
        i, t, p = v
        img_id = f'epoch_{epoch} bm_{batch_members[j]}: V'
        table.add_data(img_id, 
                           wandb.Image(apply_colormap(i)), 
                           wandb.Image(apply_colormap(t)), 
                           wandb.Image(apply_colormap(p)),
                           wandb.Image(apply_colormap(t) - apply_colormap(p)))
        
    return table