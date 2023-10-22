import torch

def pose_distance(pose_b44):
    """
    DVMVS frame pose distance.
    """

    R = pose_b44[:, :3, :3]
    t = pose_b44[:, :3, 3]
    R_trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    R_measure = torch.sqrt(2 * (1 - torch.minimum(torch.ones_like(R_trace)*3.0, R_trace) / 3))
    t_measure = torch.norm(t, dim=1)
    combined_measure = torch.sqrt(t_measure ** 2 + R_measure ** 2)
    # [B*7], [B*7], [B*7]
    return combined_measure, R_measure, t_measure