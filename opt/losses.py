import torch
import torch.nn as nn
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        ~ nn.CrossEntropyLoss()
        """
        loss = -torch.sum(teacher_output*F.log_softmax(student_output), dim=-1)
        return loss.mean()

class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_output, teacher_output):
        """
        L2 between softmax outputs of the teacher and student networks.
        ~ nn.MSELoss()
        """
        diff = student_output-teacher_output
        loss = torch.mean(torch.pow(diff, 2))
        return loss

# class rel_based_KDLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.L2Loss = L2Loss()
    
#     # relation-based KD loss
#     # 1. instance-level final vecs
#     def forward(self, s_repr, t_repr):
#         """
#         ::s_repr: (1, D)
#         ::t_repr: (1, D)
#         """
#         # B*B similarity matrix on s_repr
#         s_repr = s_repr / torch.norm(s_repr, dim=1, keepdim=True)
#         s_sim_mat = torch.mm(s_repr, s_repr.transpose(0, 1))
#         s_sim_mat = s_sim_mat / torch.max(s_sim_mat)
        
#         # B*B similarity matrix on t_repr
#         t_repr = t_repr / torch.norm(t_repr, dim=1, keepdim=True)
#         t_sim_mat = torch.mm(t_repr, t_repr.transpose(0, 1))
#         t_sim_mat = t_sim_mat / torch.max(t_sim_mat)
        
#         # # measure the difference between s_sim_mat and t_sim_mat
#         # mm = torch.mm(s_sim_mat, t_sim_mat)
#         # return torch.sum(mm)
#         return self.L2Loss(s_sim_mat, t_sim_mat)

"""
https://github.com/fiveai/LAME/tree/master
"""

class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())

class rbf_affinity(AffinityMatrix):
    """
    ::sigma : gaussian kernel bandwidth

    N개의 데이터 포인트에 대해 RBF similarity를 계산한다.
    knn 값이 주어지면 knn개의 이웃만을 고려하여 affinity를 계산한다.

    """
    def __init__(self, sigma: float = 1, knn: int = None):
        
        self.sigma = sigma
        self.k = knn

    def __call__(self, X):

        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        if self.k is None:
            self.k = N
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2)) # RBF similarity
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf

class rbf_talor_affinity(AffinityMatrix):
    """
    ::sigma : gaussian kernel bandwidth

    N개의 데이터 포인트에 대해 RBF similarity를 계산한다.

    """
    def __init__(self, delta=1, P=1):
        
        self.delta = delta
        self.P = P

    def __call__(self, X):
        device = X.device 
        z_squared = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)**2  # [N, N]
        
        approx = torch.ones(z_squared.size(), device=device)
        term = torch.tensor(1.0, device=device)

        for p in range(1, self.P+1):
            term = -term * z_squared / (p*(2*self.delta**2))
            approx += term

        return approx

class kNN_affinity(AffinityMatrix):
    """
    N개의 데이터 포인트에 대해 RBF similarity를 계산한다.
    knn 값이 주어지면 knn개의 이웃만을 고려하여 affinity를 계산한다.
    """
    def __init__(self, knn: int):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)  # [N, N]
        if self.knn is None:
            self.knn = N
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, dim=-1, largest=False).indices[:, 1:]  # [N, knn]
        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)
        return W

class MetricLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.L2Loss = L2Loss()
    
    def forward(self, X, sim_func=None):
        if sim_func is None:
            sim_func = torch.mm

        X = X / torch.norm(X, dim=1, keepdim=True)
        sim_mat = sim_func(X, X.transpose(0, 1))
        sim_mat = sim_mat / torch.max(sim_mat)

        identity = torch.eye(X.size(0))
        return self.L2Loss(sim_mat, identity)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_output, teacher_output):
        """
        L2 between outputs of the teacher and student nets.
        ~ nn.MSELoss()
        """
        diff = student_output - teacher_output
        loss = torch.mean(torch.pow(diff, 2))
        return loss
    
# relation-based KD loss
# 1. instance-level final vecs
class rel_based_KDLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.L2Loss = L2Loss()
    
    def forward(self, s_repr, t_repr, sim_func=None):
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

        return self.L2Loss(s_sim_mat, t_sim_mat)



if __name__ == "__main__":
    # Test case with 32 batchs 256-dim of X and Y
    X = torch.randn(5, 2)
    Y = torch.randn(5, 2)
    kd_loss = rel_based_KDLoss()
    knn_sim=kNN_affinity(knn=5)
    rbf_sim = rbf_affinity(sigma=1, knn=5)
    rel_based_kdLoss = kd_loss(X, Y, sim_func=None)
    print("rel_based_KDLoss", rel_based_kdLoss)

    