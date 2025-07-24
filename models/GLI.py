import numpy as np
import os
from tqdm import tqdm
import itertools
import time
import torch
import torch.nn as nn



class TopologyBatch(nn.Module):
    def __init__(self, kinematic_chain, nb_joints):
        super().__init__()
        self.kinematic_chain = kinematic_chain
        self.nb_joints = nb_joints

    def gauss_integral_vectorized(self, s1, e1, s2, e2):
        """
        Optimized calculation of the Gauss Link Integral (GLI) for arrays of particle chains.

        Parameters:
            s1, e1, s2, e2: torch.tensor (..., 3)
                3D position vectors representing the start and end points of two chains.

        Returns:
            torch.tensor: The computed GLI values for all input chains.
        """
        # Compute vectors between points
        r13 = s2 - s1
        r14 = e2 - s1
        r23 = s2 - e1
        r24 = e2 - e1
        r12 = e1 - s1
        r34 = e2 - s2

        # Calculate face normals and their norms
        faces = torch.stack([
            torch.cross(r13, r14, dim=-1),
            torch.cross(r14, r24, dim=-1),
            torch.cross(r24, r23, dim=-1),
            torch.cross(r23, r13, dim=-1)
        ], dim=-2)  # Shape: (..., 4, 3)

        norms = torch.norm(faces, dim=-1, keepdim=True)  # Shape: (..., 4, 1)
        normalized_faces = torch.where(norms > 0, faces / norms, torch.zeros_like(faces))

        # Compute GLI using arcsin of dot products
        dots = torch.sum(normalized_faces * torch.roll(normalized_faces, shifts=-1, dims=-2), dim=-1)  # (..., 4)
        dots = torch.clip(dots, -1.0 + (1e-7), 1.0 - (1e-7))  # Clip to handle numerical issues
        gli = torch.sum(torch.arcsin(dots), dim=-1)  # (...)

        # Determine the sign of GLI using cross-product and dot-product
        sign = torch.sum(torch.cross(r34, r12, dim=-1) * r13, dim=-1)  # (...)
        gli *= torch.where(sign <= 0, -1.0, 1.0)

        # Scale by the normalization factor
        return gli / (4.0 * torch.pi)

    def gauss_integral_motion_optimized(self, motion1, motion2):
        """
        Optimized computation of Gauss Link Integral motion for entire sequences of poses.

        Parameters:
            motion1, motion2: torch.tensor, shape (batch, frame_num, joint_num, 3),
                              describing the positions of joints across frames for two motions.

        Returns:
            torch.tensor: Shape (batch, frame_num-1,), containing the maximum absolute velocity of GLI changes across frames.
        """
        # Define paths from the kinematic chain
        if self.nb_joints == 22:
            paths = [chain[1:] for chain in self.kinematic_chain]
        elif self.nb_joints == 55:
            paths = [chain[1:] for chain in self.kinematic_chain[:5]]
            paths_hand = [chain for chain in self.kinematic_chain[5:]]
            paths = paths + paths_hand
        else:
            raise ValueError("Unsupported number of joints: {}".format(self.nb_joints))


        # Extract frames and path indices
        batch_num, frame_num = motion1.shape[:2]
        path_num = len(paths)

        # Prepare tensors to store GLI values
        GLI_motion = torch.zeros((batch_num, frame_num, path_num, path_num)).to(motion1.device)

        # Convert paths to padded tensors for indexing

        # Gather all path positions across frames
        for i in range(len(paths)):
            path1 = paths[i]
            for j in range(len(paths)):
                path2 = paths[j]
                motion1_start = motion1[:, :, path1[:-1]].unsqueeze(3).expand(-1, -1, -1, len(path2) - 1, -1)
                motion1_end = motion1[:, :, path1[1:]].unsqueeze(3).expand(-1, -1, -1, len(path2) - 1, -1)
                motion2_start = motion2[:, :, path2[:-1]].unsqueeze(2).expand(-1, -1, len(path1) - 1, -1, -1)
                motion2_end = motion2[:, :, path2[1:]].unsqueeze(2).expand(-1, -1, len(path1) - 1, -1, -1)
                gli_motion_path = self.gauss_integral_vectorized(motion1_start, motion1_end, motion2_start, motion2_end)
                GLI_motion[:, :, i, j] += torch.sum(gli_motion_path, dim=[-1, -2])

        # Perform vectorized GLI computation for all frames

        # Compute frame-to-frame GLI velocity
        GLI_abs_vel = torch.abs(GLI_motion[:, 1:] - GLI_motion[:, :-1])
        GLI_abs_vel = GLI_abs_vel.view(batch_num, frame_num - 1, -1)
        GLI_abs_vel_max = torch.max(GLI_abs_vel, dim=-1)[0]

        return GLI_abs_vel_max

    def forward(self, motion1, motion2):
        return self.gauss_integral_motion_optimized(motion1, motion2)
