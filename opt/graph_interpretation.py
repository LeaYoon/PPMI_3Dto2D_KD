import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
from scipy.cluster.hierarchy import dendrogram, linkage

def save_graph_diff(save_dir, s_repr, t_repr, sim_func, idx_Z=None):
    """
    ::s_repr: in (B, D)
    ::t_repr: in (B, D)
    """
    if sim_func is None:
        sim_func = torch.mm
 
    # B*B similarity matrix on s_repr
    s_repr = s_repr / torch.norm(s_repr, dim=1, keepdim=True)
    if sim_func is torch.mm:
        s_sim_mat = sim_func(s_repr, s_repr.transpose(0, 1))
    else:
        s_sim_mat = sim_func(s_repr)
    s_sim_mat = s_sim_mat / torch.max(s_sim_mat)

    # B*B similarity matrix on t_repr
    t_repr = t_repr / torch.norm(t_repr, dim=1, keepdim=True)
    if sim_func is torch.mm:
        t_sim_mat = sim_func(t_repr, t_repr.transpose(0, 1))
    else:
        t_sim_mat = sim_func(t_repr)
    t_sim_mat = t_sim_mat / torch.max(t_sim_mat)
    
    diff = t_sim_mat-s_sim_mat

    ### dendrogram [!] 
        
    # linkage
    Z = linkage(t_sim_mat.detach().cpu().numpy(), method="ward") # diff -> t_sim_mat
    if idx_Z is not None:
        idx_Z = idx_Z
    else:
        idx_Z = dendrogram(Z, no_plot=True)['leaves']

    # Z_X = linkage(s_sim_mat, method="ward")
    # idx_Z_X = dendrogram(Z_X, no_plot=True)['leaves']

    # Z_Y = linkage(t_sim_mat, method="ward")
    # idx_Z_Y = dendrogram(Z_Y, no_plot=True)['leaves']

    # sort
    _diff = diff[idx_Z][:, idx_Z]
    _s_sim_mat = s_sim_mat[idx_Z][:, idx_Z]
    _t_sim_mat = t_sim_mat[idx_Z][:, idx_Z]
        
    # plot with same size of figures
    t_sim_mat = t_sim_mat.detach().cpu().numpy()
    _t_sim_mat = _t_sim_mat.detach().cpu().numpy()
    s_sim_mat = s_sim_mat.detach().cpu().numpy()
    _s_sim_mat = _s_sim_mat.detach().cpu().numpy()
    diff = diff.detach().cpu().numpy()
    _diff = _diff.detach().cpu().numpy()

    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(nrows=2, ncols=3)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(t_sim_mat, cmap="jet")
    ax0.set_title("Teacher original")
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(s_sim_mat, cmap="jet")
    ax1.set_title("Student original")
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(diff, cmap="jet")
    ax2.set_title("Difference")
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(_t_sim_mat, cmap="jet")
    ax3.set_title("Sorted Teacher")
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(_s_sim_mat, cmap="jet")
    ax4.set_title("Sorted Student")
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.imshow(_diff, cmap="jet")
    ax5.set_title("Dentrogram of Difference")

    fig.colorbar(mappable=plt.cm.ScalarMappable(cmap="jet"), ax=[ax0, ax1, ax2, ax3, ax4, ax5], shrink=1.0)
    fig.savefig(os.path.join(save_dir, "diff_graph.png"))

    
    # 히스토그램을 사용한 유사도 분포 비교
    plt.figure(figsize=(6, 6))
    plt.hist(s_sim_mat.flatten(), bins=30, alpha=0.5, label='Student')
    plt.hist(t_sim_mat.flatten(), bins=30, alpha=0.5, label='Teacher')
    plt.title('Similarity Distribution')
    plt.xlabel('Similarity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(save_dir, "histogram.png"))

    # save matrix
    np.save(os.path.join(save_dir, "t_sim_mat.npy"), t_sim_mat)
    np.save(os.path.join(save_dir, "s_sim_mat.npy"), s_sim_mat)
    np.save(os.path.join(save_dir, "diff.npy"), diff)
    return idx_Z







