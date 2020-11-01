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


def nll_loss(pred: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NLL averages across steps, samples, and dimensions(x,y)"""
    x_mean = pred[:, :, 0]
    y_mean = pred[:, :, 1]
    x_sigma = pred[:, :, 2]
    y_sigma = pred[:, :, 3]
    rho = pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)
    x = data[:, :, 0]
    y = data[:, :, 1]
    results = torch.pow(ohr, 2) * (
            torch.pow(x_sigma, 2) * torch.pow(x - x_mean, 2) + torch.pow(y_sigma, 2) * torch.pow(y - y_mean, 2)
            - 2 * rho * torch.pow(x_sigma, 1) * torch.pow(y_sigma, 1) * (x - x_mean) * (y - y_mean)) - torch.log(
        x_sigma * y_sigma * ohr)

    results = results * mask[:, :, 0]
    assert torch.sum(mask) > 0.0
    return torch.sum(results) / torch.sum(mask[:, :, 0])


def nll_loss_per_sample(pred: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """NLL averages across steps and dimensions, but not samples (agents)."""
    x_mean = pred[:, :, 0]
    y_mean = pred[:, :, 1]
    x_sigma = pred[:, :, 2]
    y_sigma = pred[:, :, 3]
    rho = pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)  # type: ignore
    x = data[:, :, 0]
    y = data[:, :, 1]
    results = torch.pow(ohr, 2) * (
            torch.pow(x_sigma, 2) * torch.pow(x - x_mean, 2) + torch.pow(y_sigma, 2) * torch.pow(y - y_mean, 2)
            - 2 * rho * torch.pow(x_sigma, 1) * torch.pow(y_sigma, 1) * (x - x_mean) * (y - y_mean)) - torch.log(
        x_sigma * y_sigma * ohr)
    results = results * mask[:, :, 0]  # nSteps by nBatch
    return torch.sum(results, dim=0) / torch.sum(mask[:, :, 0], dim=0)


def nll_loss_test(pred: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """NLL for testing cases, returns a vector over future timesteps."""
    x_mean = pred[:, :, 0]
    y_mean = pred[:, :, 1]
    x_sigma = pred[:, :, 2]
    y_sigma = pred[:, :, 3]
    rho = pred[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)  # type: ignore
    x = data[:, :, 0]
    y = data[:, :, 1]
    results = torch.pow(ohr, 2) * (
            torch.pow(x_sigma, 2) * torch.pow(x - x_mean, 2) + torch.pow(y_sigma, 2) * torch.pow(y - y_mean, 2)
            - 2 * rho * torch.pow(x_sigma, 1) * torch.pow(y_sigma, 1) * (x - x_mean) * (y - y_mean)) - torch.log(
        x_sigma * y_sigma * ohr)
    results = results * mask[:, :, 0]  # nSteps by nBatch
    assert torch.sum(mask) > 0.0
    counts = torch.sum(mask[:, :, 0], dim=1)
    return torch.sum(results, dim=1), counts


def mse_loss(pred: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss."""
    x_mean = pred[:, :, 0]
    y_mean = pred[:, :, 1]
    x = data[:, :, 0]
    y = data[:, :, 1]
    results = torch.pow(x - x_mean, 2) + torch.pow(y - y_mean, 2)
    results = results * mask[:, :, 0]
    return torch.sum(results) / torch.sum(mask[:, :, 0])


def mse_loss_test(pred: torch.Tensor, data: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error loss for test time."""
    x_mean = pred[:, :, 0]
    y_mean = pred[:, :, 1]
    x = data[:, :, 0]
    y = data[:, :, 1]
    results = torch.pow(x - x_mean, 2) + torch.pow(y - y_mean, 2)
    results = results * mask[:, :, 0]
    counts = torch.sum(mask[:, :, 0], dim=1)
    lossVal = torch.sum(results, dim=1)
    return lossVal, counts


def logsumexp(inputs: torch.Tensor, dim: Optional[int] = None, keepdim: Optional[bool] = False) -> torch.Tensor:
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


def nll_loss_multimodes(pred: List[torch.Tensor], data: torch.Tensor, mask: torch.Tensor, modes_pred: torch.Tensor,
                        noise: Optional[float] = 0.0) -> float:
    """NLL loss multimodes for training.
  Args:
    pred is a list (with N modes) of predictions
    data is ground truth    
    noise is optional
  """

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
