import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import exp

# from utilities.misc import pause, check_var

# from loguru import logger


def dst_variables(logits, T=1.0):
    assert T == 1.0, "do not want temperature for now"
    if T == 1.0:
        belief = F.log_softmax(logits, dim=-1)
    else:
        belief = F.log_softmax(logits / T, dim=-1)
    return belief


# * logits_dict: {0: [0.9, 0.1], to K-class}
# * nonneg_type: 'none'
def form_opinion_general(
    logits_dict,
    batch_size,
    class_num,
    sample_cnt=-1,
    temparature=1.0,
    for_train=True,
    device="cuda",
    check=False,
):
    # ^ DST values for each binary belief function
    beliefs_log = torch.empty(size=(batch_size, class_num, 2)).to(device)

    if type(logits_dict) == dict:
        for i, logits in logits_dict.items():
            belief_log = dst_variables(logits, T=temparature)
            beliefs_log[:, i, :] = belief_log
    elif torch.is_tensor(logits_dict):
        beliefs_log[:, :, 0] = logits_dict
        beliefs_log[:, :, 1] = 1 - logits_dict
        beliefs_log = F.log_softmax(beliefs_log, dim=-1)
    else:
        raise NotImplementedError
    # ^ ignorance
    ignorance_log = torch.sum(beliefs_log[:, :, 1], dim=1, keepdim=True)

    # ^ singleton beliefs
    bel_term_log = beliefs_log[:, :, 0] - beliefs_log[:, :, 1]  # 去掉第 i 类对应的 1-pl_i
    belief_log = ignorance_log + bel_term_log  # 加上所有的 pl_i
    belief = torch.exp(belief_log)
    belief_sum = belief.sum(-1, keepdim=True)

    # ^ total uncertainty
    uncertainty = 1.0 - belief_sum

    # ^ alpha of singleton belief
    S = class_num / (uncertainty + 1e-6)
    alpha = belief * S + 1.0

    # ^ alpha of plausibility
    # beliefs = torch.exp(beliefs_log)
    # alpha_pl = beliefs[:, :, 0] * S + 1.0

    ignorance = torch.exp(ignorance_log)
    plausibility = torch.exp(beliefs_log[:, :, 0])
    # bel_term = torch.exp(bel_term_log)
    # if for_train:
        # ^ For regularization term
    plausibility_max = 1.0 - ignorance
    opinions = {
        "S": S,
        "uncertainty": uncertainty.squeeze(-1),
        "belief": belief,
        "plausibility": plausibility,
        # "alpha_pl": alpha_pl,
        # "belief_ratio": bel_term,
        "plausibility_max": plausibility_max,
        "ignorance": ignorance
    }
    return alpha, opinions


class FlexibleLoss(nn.Module):
    def __init__(
        self,
        class_num,
        use_pl_alpha=False,
        annealing=500,
        base_rate=None,
        loss_type="mse",
        nonneg_type="relu",
        with_kl=False,
        loss_factor=1.0,
        reweight_factor=0.05,
        sample_cnt=-1,
        temparature=1.0,
        device=None,
        lambdap=0.2
    ):
        super(FlexibleLoss, self).__init__()
        self.device = device
        self.class_num = class_num
        self.use_pl_alpha = use_pl_alpha
        self.W = class_num
        self.base_rate = base_rate

        self.loss_type = loss_type
        self.nonneg_type = nonneg_type
        self.with_kl = with_kl
        self.loss_factor = loss_factor
        self.sample_cnt = sample_cnt
        self.temparature = temparature
        # CE
        self.ce_loss = nn.CrossEntropyLoss(reduction="none").to(device=device)
        self.ce_T = 5

        # Regularization MSE
        self.reg_loss = nn.MSELoss(reduction="mean").to(device=device)

        # KL
        self.T = annealing / reweight_factor
        self.epoch_single_print, self.pre_print_epoch = True, -1
        self.lambdap = lambdap

    def edl_nll_loss(self, alpha, labels):
        S = alpha.sum(dim=1, keepdim=True)
        nll_loss = F.nll_loss(
            torch.log(alpha) - torch.log(S),
            labels,
            weight=None,
            reduction="mean",
        )
        return nll_loss

    def edl_mse_loss(self, alpha, yi):
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((yi - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        mse_loss = torch.mean(loglikelihood_err + loglikelihood_var)
        return mse_loss

    def binary_labels(self, label):
        label_dict = {}
        for i in range(self.class_num):
            label_dict[i] = torch.ones_like(label)
            label_dict[i][label == i] = 0
        return label_dict

    def forward(self, logits_dict, labels, epoch_num, check=False, phase="train"):
        """
        logits_dict = {
            class 0: [batchsize, 2],
            class 1: [batchsize, 2],
        }
        """
        labels = labels.to(self.device)

        if type(logits_dict) == dict:
            for k in logits_dict.keys():
                logits_dict[k] = logits_dict[k].to(self.device)
        elif torch.is_tensor(logits_dict):
            logits_dict = logits_dict.to(self.device)
        else:
            raise NotImplementedError

        loss = 0.0

        # ^ Singleton loss
        # if phase == "train":
        alpha, opinions = form_opinion_general(
            logits_dict,
            labels.size(0),
            self.class_num,
            temparature=self.temparature,
            sample_cnt=self.sample_cnt,
            check=check,
        )
        loss_dict = None

        if self.use_pl_alpha:
            alpha = opinions["alpha_pl"]

        # ^ For cold start problem
        ce_loss = torch.tensor((0.0)).to(labels.device)
        bi_label_dict = self.binary_labels(labels)
        beliefs_log = torch.empty(size=(logits_dict.shape[0], self.class_num, 2)).to("cuda")
        beliefs_log[:, :, 0] = logits_dict
        beliefs_log[:, :, 1] = 1 - logits_dict

        if epoch_num < self.ce_T:
            if type(logits_dict) == dict:
                for i in range(self.class_num):
                    ce_loss_term = torch.mean(
                        self.ce_loss(logits_dict[i], bi_label_dict[i])
                    )
                    ce_loss += ce_loss_term
            else:
                for i in range(self.class_num):
                    ce_loss_term = torch.mean(
                        self.ce_loss(beliefs_log[:, i, :], bi_label_dict[i])
                    )
                    ce_loss += ce_loss_term
            ce_loss /= self.class_num
            loss += ce_loss

        # ^ Combined EDL loss
        yi = F.one_hot(labels, num_classes=alpha.shape[1])
        if self.loss_type == "nll":
            edl_loss_1 = self.edl_nll_loss(alpha, labels)
        elif self.loss_type == "mse":
            edl_loss_1 = self.edl_mse_loss(alpha, yi)

        # ^ Regularization
        plausibility_max = opinions["plausibility_max"].detach()
        plausibility = opinions["plausibility"]
        belief_gt = plausibility.gather(-1, labels.unsqueeze(-1))
        edl_loss_2 = self.reg_loss(belief_gt, plausibility_max)
        edl_loss = self.loss_factor * (edl_loss_1 + self.lambdap * edl_loss_2)
        # edl_loss = self.loss_factor * (edl_loss_1)

        # ^ KL loss
        if self.with_kl:
            alpha_tilde = yi + (1 - yi) * alpha
            S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
            
            kl = (
                torch.lgamma(S_tilde)
                - torch.lgamma(torch.tensor(alpha_tilde.shape[1]))
                - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
                + (
                    (alpha_tilde - 1)
                    * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
                ).sum(dim=1, keepdim=True)
            )
            kl_loss = epoch_num / self.T * kl.squeeze(-1)
            # kl_loss = 0 / self.T * kl.squeeze(-1)
        else:
            kl_loss = torch.tensor(0.0)

        loss += edl_loss + kl_loss.mean()

        loss_dict = {
            "ce_loss": ce_loss,
            "edl_loss": edl_loss,
            "edl_loss_1": edl_loss_1,
            "edl_loss_2": edl_loss_2,
            "kl_loss": kl_loss.mean(),
        }
        return loss, loss_dict, opinions


def create_loss(
    class_num,
    annealing_epochs,
    use_pl_alpha=False,
    base_rate=None,
    loss_type="mse",
    nonneg_type="relu",
    with_kl=True,
    loss_factor=1.0,
    reweight_factor=0.05,
    sample_cnt=-1,
    temparature=1.0,
    device='cuda',
    *args,
    lambdap=0.2
):
    print("Using Flexible loss ", temparature)
    criterion = FlexibleLoss(
        class_num,
        use_pl_alpha=use_pl_alpha,
        annealing=annealing_epochs,
        base_rate=base_rate,
        loss_type=loss_type,
        nonneg_type=nonneg_type,
        with_kl=with_kl,
        loss_factor=loss_factor,
        reweight_factor=reweight_factor,
        sample_cnt=sample_cnt,
        temparature=temparature,
        device=device,
        lambdap=lambdap
    )
    return criterion