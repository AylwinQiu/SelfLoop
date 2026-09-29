from sympy.abc import delta
import torch as tc
import numpy as np
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, x_dim:int, y_dim:int, h_id_dim:int, h_i_dim:int, h_j_dim:int, H_ij_dim:int, delta_f_dim:int, project_head_num:int=5):
        super().__init__()
        self.fc1 = nn.Linear(x_dim+y_dim+h_id_dim+h_i_dim+h_j_dim+H_ij_dim, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, y_dim+delta_f_dim+H_ij_dim)
        # Get model parameters number
        f_parameter_num = 2 #TODO
        self.project_heads = tc.rand((project_head_num, delta_f_dim, f_parameter_num))