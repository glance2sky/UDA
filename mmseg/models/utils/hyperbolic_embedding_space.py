import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn

import torch
import math


def torch_exp_map_zero(inputs, c, EPS=1e-7):
    """
    PyTorch implementation of exponential mapping from Euclidean to hyperbolic space (Poincaré ball model).

    Args:
        inputs: Input tensor of shape [n, d] (Euclidean coordinates)
        c: Curvature of the hyperbolic space (positive scalar)
        EPS: Small constant for numerical stability

    Returns:
        Projected points in the Poincaré ball
    """
    sqrt_c = torch.sqrt(torch.tensor(c, device=inputs.device))

    # Add epsilon to avoid division by zero
    inputs = inputs + EPS

    # Compute norm along the last dimension
    norm = torch.norm(inputs, p=2, dim=-1, keepdim=False)

    # Compute scaling factor gamma
    gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)

    # Scale the input vectors
    scaled_inputs = gamma.unsqueeze(-1) * inputs

    # Project to Poincaré ball
    return torch_project_hyp_vecs(scaled_inputs, c, dim=-1)


def torch_project_hyp_vecs(x, c, dim=-1):
    """
    Project hyperbolic vectors to ensure they stay within the Poincaré ball.

    Args:
        x: Input tensor
        c: Curvature
        dim: Dimension to compute norm over

    Returns:
        Clipped tensor within the Poincaré ball
    """
    PROJ_EPS = 1e-5
    max_norm = (1.0 - PROJ_EPS) / math.sqrt(c)

    # Compute norms along specified dimension
    norms = torch.norm(x, p=2, dim=dim, keepdim=True)

    # Clip norms
    clipped = torch.clamp(norms, max=max_norm)

    # Project vectors
    return x * (clipped / (norms + PROJ_EPS))
class HyperbolicEmbeddingSpace(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,embeddings, curvature=0):
        projected_embedding = torch_exp_map_zero(embeddings, c=curvature)
        logits = hyp_mlr(projected_embedding,c=curvature,P_mlr, A_mlr)