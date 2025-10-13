import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.ao.nn.quantized import BatchNorm2d


class SAEML(nn.Module):
    def __init__(self, num_views, dims, num_classes):
        super(SAEML, self).__init__()
        self.num_views = num_views
        self.num_classes = num_classes
        self.EvidenceCollectors = nn.ModuleList(
            [EvidenceCollector(dims[i], self.num_classes) for i in range(self.num_views)])

    def forward(self, X):
        evidences = {}
        for v in range(self.num_views):
            evidences[v] = self.EvidenceCollectors[v](X[v])

        alpha = {v: evidences[v] + 1 for v in range(self.num_views)}
        alpha0 = {v: torch.sum(alpha[v], dim=-1, keepdim=True) for v in range(self.num_views)}
        belief_mass = {v: evidences[v] / alpha0[v] for v in range(self.num_views)}
        u_view = {v: self.num_classes / alpha0[v] for v in range(self.num_views)}

        composite_belief_mass = {}
        for a in range(self.num_views):
            for b in range(a + 1, self.num_views):
                key = (a, b)
                composite_belief_mass[key] = (
                                                     belief_mass[a] * (1 - u_view[a]) * u_view[b] +
                                                     belief_mass[b] * (1 - u_view[b]) * u_view[a]
                                             ) / (u_view[a] + u_view[b] - 2 * u_view[a] * u_view[b])

        a_baserate = {
            v: alpha[v] / torch.sum(torch.stack([alpha[w] for w in range(self.num_views)]), dim=0)
            for v in range(self.num_views)
        }
        composite_baserate_a = {
            (a, b): (
                            a_baserate[a] * (1 - u_view[a]) +
                            a_baserate[b] * (1 - u_view[b])
                    ) / (2 - u_view[a] - u_view[b])
            for a in range(self.num_views) for b in range(a + 1, self.num_views)
        }

        vague_belief_mass = {}
        for v in range(self.num_views):
            relevant_composites = [k for k in composite_belief_mass if v in k]
            vague_belief_mass[v] = sum(
                composite_belief_mass[k] * a_baserate[v] / composite_baserate_a[k]
                for k in relevant_composites
            ) / (self.num_views - 1)

        u_category = (self.num_views - sum(belief_mass.values()) - sum(composite_belief_mass.values())) / (self.num_views) / (self.num_classes)
        focal_u = {
            v: a_baserate[v] * u_category
            for v in range(self.num_views)
        }

        P_projected = {
            v: belief_mass[v] + vague_belief_mass[v] - focal_u[v]
            for v in range(self.num_views)
        }

        evidence_a = (sum(
            P_projected[v] * evidences[v] for v in range(self.num_views)
        ))

        return evidences, evidence_a, u_category


class EvidenceCollector(nn.Module):
    def __init__(self, dims, num_classes):
        super(EvidenceCollector, self).__init__()
        self.num_layers = len(dims)
        self.net = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.net.append(nn.Linear(dims[i], dims[i + 1]))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(0.1))

        #Scene,sources
        self.net.append(nn.Linear(dims[self.num_layers - 1], 128))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(0.1))
        self.net.append(nn.Linear(128, num_classes))
        #PIE
        #self.net.append(nn.Linear(dims[self.num_layers - 1], num_classes))

        self.net.append(nn.Softplus())

    def forward(self, x):
        sn=self.net
        h = self.net[0](x)
        for i in range(1, len(self.net)):
            h = self.net[i](h)
        return h


def get_loss_fisher(evidences, evidence_a, target, epoch_num, u_category,A,B,fisher_c,ulossweight):
    alpha_a = evidence_a + 1
    loss_acc = compute_fmse(alpha_a, target, epoch_num, u_category,A,B,fisher_c,ulossweight)
    for v in evidences:
        alpha = evidences[v] + 1
        loss_acc += compute_fmse(alpha, target, epoch_num, u_category,A,B,fisher_c,ulossweight)
    loss_acc = loss_acc / (len(evidences) + 1)
    return loss_acc

def compute_fmse(evi_alp_, labels_, epoch, u_category,A,B,fisher_c,ulossweight):

    target_concentration = 100
    # Convert labels to one-hot encoding
    labels_1hot_ = torch.zeros(evi_alp_.size(0), evi_alp_.size(1), device=evi_alp_.device).scatter_(1, labels_.view(-1, 1), 1)
    evi_alp0_ = torch.sum(evi_alp_, dim=-1, keepdim=True)
    gamma1_alp = torch.polygamma(1, evi_alp_)
    gamma1_alp0 = torch.polygamma(1, evi_alp0_)

    # Compute gap and losses
    gap = labels_1hot_ - evi_alp_ / evi_alp0_
    loss_mse_ = ((gap.pow(2) * gamma1_alp).sum(-1)).mean() / 3.
    loss_var_ = (evi_alp_ * (evi_alp0_ - evi_alp_) * gamma1_alp / (evi_alp0_ * evi_alp0_ * (evi_alp0_ + 1))).sum(-1).mean() / 3.
    loss_det_fisher_ = -(
        torch.log(gamma1_alp).sum(-1) + torch.log(1.0 - (gamma1_alp0 / gamma1_alp).sum(-1))
    ).mean() / 3.
    loss_kl_ = compute_kl_loss(evi_alp_, target_concentration, labels_) / 3.
    regr = np.minimum(1.0, (epoch + 1) / 10.)
    loss = A * loss_mse_ + B * loss_var_ + fisher_c * loss_det_fisher_ + regr * 0.05 * loss_kl_ + ulossweight * u_category.mean()
    #loss = loss_mse_ + loss_var_ + loss_det_fisher_ + regr * 0.05 * loss_kl_ + u_category.mean()
    return loss

def compute_kl_loss(alphas, target_concentration=1, labels=None, concentration=1.0, epsilon=1e-8):
    if target_concentration < 1.0:
        concentration = target_concentration

    target_alphas = torch.ones_like(alphas) * concentration
    if labels is not None:
        target_alphas += torch.zeros_like(alphas).scatter_(-1, labels.unsqueeze(-1), target_concentration - 1)

    alp0 = torch.sum(alphas, dim=-1, keepdim=True)
    target_alp0 = torch.sum(target_alphas, dim=-1, keepdim=True)

    alp0_term = torch.lgamma(alp0 + epsilon) - torch.lgamma(target_alp0 + epsilon)
    alp0_term = torch.where(torch.isfinite(alp0_term), alp0_term, torch.zeros_like(alp0_term))
    assert torch.all(torch.isfinite(alp0_term)).item()

    alphas_term = torch.sum(torch.lgamma(target_alphas + epsilon) - torch.lgamma(alphas + epsilon)
                            + (alphas - target_alphas) * (torch.digamma(alphas + epsilon) -
                                                          torch.digamma(alp0 + epsilon)), dim=-1, keepdim=True)
    alphas_term = torch.where(torch.isfinite(alphas_term), alphas_term, torch.zeros_like(alphas_term))
    assert torch.all(torch.isfinite(alphas_term)).item()

    loss = (torch.squeeze(alp0_term + alphas_term)).mean()

    return loss