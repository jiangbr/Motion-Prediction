# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional, Union
from scipy import special
import numpy as np
import torch
import json
import os

"""
Utility function
"""


def Gaussian2d(x: torch.Tensor) -> torch.Tensor:
    """Computes the parameters of a bivariate 2D Gaussian"""
    x_mean = x[:, :, 0]
    y_mean = x[:, :, 1]
    sigma_x = torch.exp(x[:, :, 2])
    sigma_y = torch.exp(x[:, :, 3])
    rho = torch.tanh(x[:, :, 4])
    return torch.stack([x_mean, y_mean, sigma_x, sigma_y, rho], dim=2)


def logsumexp(inputs: torch.Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False) -> torch.Tensor:
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def nll_loss_multimodes(pred: torch.Tensor, truth: torch.Tensor, mask: torch.Tensor, modes_pred: torch.Tensor,
                        noise: Optional[float] = 0.0, index: Optional[np.array] = None) -> torch.Tensor:
    """NLL loss multimodes for training.
    Args:
    pred is [mode, fut_len, num_agents, 2]
    truth is [fut_len, num_agents, 2]
    mask is [fut_len, num_agents, 1]
    mode_pred is [num_agents, mode]
    noise is optional
    index denote which agent is to be calculated
    """
    # add modes
    truth = torch.unsqueeze(truth, 0)
    mask = torch.unsqueeze(mask, 0)
    # if index is not None, then extract corresponding agents
    if index is not None:
        pred_list = []
        mode_list = []
        for item in index:
            pred_list.append(pred[:, :, item:item+1, :])
            mode_list.append(modes_pred[item:item+1, :])
        pred = torch.cat(pred_list, dim=2)
        modes_pred = torch.cat(mode_list, dim=0)

    # reduce coordinate and use mask, error is [mode, fut_len, num_agents]
    error = torch.sum(((truth - pred) * mask) ** 2, dim=-1)
    # when confidence is 0 log goes to -inf, but we're fine with it
    # error is [mode, num_agents]
    error = torch.log(modes_pred + 1e-6).permute(1, 0) - 0.5 * torch.sum(error, dim=-2)
    # use max aggregator on modes for numerical stability
    max_value = torch.max(error, dim=0, keepdim=True).values
    # reduce modes
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=0)) - max_value[0]
    # return average of all the agents
    if index is None:
        loss = torch.mean(error)
    else:
        loss = torch.sum(error)
    return loss


################################################################################
def load_json_file(json_filename: str) -> dict:
    with open(json_filename) as json_file:
        json_dictionary = json.load(json_file)
    return json_dictionary


def write_json_file(json_filename: str, json_dict: dict, pretty: Optional[bool] = False) -> None:
    with open(os.path.expanduser(json_filename), 'w') as outfile:
        if pretty:
            json.dump(json_dict, outfile, sort_keys=True, indent=2)
        else:
            json.dump(json_dict, outfile, sort_keys=True, )


def pi(obj: Union[torch.Tensor, np.ndarray]) -> None:
    """ Prints out some info."""
    if isinstance(obj, torch.Tensor):
        print(str(obj.shape), end=' ')
        print(str(obj.device), end=' ')
        print('min:', float(obj.min()), end=' ')
        print('max:', float(obj.max()), end=' ')
        print('std:', float(obj.std()), end=' ')
        print(str(obj.dtype))
    elif isinstance(obj, np.ndarray):
        print(str(obj.shape), end=' ')
        print('min:', float(obj.min()), end=' ')
        print('max:', float(obj.max()), end=' ')
        print('std:', float(obj.std()), end=' ')
        print(str(obj.dtype))


def compute_angles(x_mean: torch.Tensor, num_steps: int = 3) -> torch.Tensor:
    """Compute the 2d angle of trajectories.
  Args:
    x_mean is [nSteps, nObjs, dim]
  """
    nSteps, nObjs, dim = x_mean.shape
    thetas = np.zeros((nObjs, num_steps))
    for k in range(num_steps):
        for o in range(nObjs):
            diff = x_mean[k + 1, o, :] - x_mean[k, o, :]
            thetas[o, k] = np.arctan2(diff[1], diff[0])
    return thetas.mean(axis=1)


def rotate_to(data: np.ndarray, theta0: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Rotate data about location x0 with theta0 in radians.
  Args:
    data is [nSteps, dim] or [nSteps, nObjs, dim]
  """
    rot = np.array([[np.cos(theta0), np.sin(theta0)],
                    [-np.sin(theta0), np.cos(theta0)]])
    if len(data.shape) == 2:
        return np.dot(data - x0, rot.T)
    else:
        nSteps, nObjs, dim = data.shape
        return np.dot(data.reshape((-1, dim)) - x0, rot.T).reshape((nSteps, nObjs, dim))


def rotate_to_inv(data: np.ndarray, theta0: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Inverse rotate data about location x0 with theta0 in radians.
  Args:
    data is [nSteps, dim] or [nSteps, nObjs, dim]
  """
    rot = np.array([[np.cos(theta0), -np.sin(theta0)],
                    [np.sin(theta0), np.cos(theta0)]])
    if len(data.shape) == 2:
        return np.dot(data, rot.T) + x0
    else:
        nSteps, nObjs, dim = data.shape
        return (np.dot(data.reshape((-1, dim)), rot.T) + x0).reshape((nSteps, nObjs, dim))
