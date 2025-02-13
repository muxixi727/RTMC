import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loss import create_loss, form_opinion_general


class EMV(nn.Module):
    def __init__(self, classes, dims, num_epochs, loss_type='mse', lambdap=0.2):
        super(EMV, self).__init__()
        self.views = np.size(dims)
        self.classes = classes
        # self.emv = nn.ModuleList([EvidentialModel(self.classes, dims[v], dims[v]) for v in range(self.views)])
        # HMDB，SCENE
        self.emv = nn.ModuleList([EvidentialModel(self.classes, dims[v], 128) for v in range(self.views)])
        self.criterion = create_loss(class_num=self.classes, annealing_epochs=num_epochs, lambdap=lambdap)
        self.loss_type = loss_type
        self.lambdap = lambdap

    def forward(self, x1, labels, epoch, phase):
        pl_all = []
        loss = dict()
        opinion = dict()
        single_view_loss_all, loss_all = 0.0, 0.0
        for v in range(self.views):
            pl = self.emv[v](x1[v])
            pl_all.append(pl)
            # 单视角损失
            loss[v] = self.criterion(pl, labels.type(torch.LongTensor), epoch, phase)
            opinion[v] = loss[v][2]
            single_view_loss_all += loss[v][0]
        # 多视角损失
        alpha, mv_belief, mv_ignorance, mv_confusion, mv_pl = self.multi_view_fusion(opinion)
        mv_loss = self.criterion(mv_pl, labels.type(torch.LongTensor), epoch, phase)
        belief_all = torch.sum(mv_loss[2]['belief'], dim=1).unsqueeze(1)
        mv_confusion = torch.ones(mv_loss[2]['belief'].shape[0], 1).cuda() - belief_all - mv_loss[2]['ignorance']
        mv_loss_all, mv_cfloss = 0.0, 0.0
        # 混淆损失
        cf_loss = -torch.log1p(torch.exp(-mv_confusion))
        # cf_loss = mv_confusion ** 2
        mv_cfloss = self.lambdap * torch.mean(cf_loss)
        # mv_cfloss = 0 * torch.mean(cf_loss)
        mv_loss_all = mv_loss[0]
        loss_all = single_view_loss_all + mv_loss_all + mv_cfloss

        # 整理要输出的内容
        single_view_loss = {v: loss[v][0] for v in range(self.views)}
        single_view_opinion = {v: opinion[v] for v in range(self.views)}

        return loss_all, opinion, mv_belief, \
               {"single_view_loss": single_view_loss, "single_view_opinion": single_view_opinion, "single_view_loss_all": single_view_loss_all,
                "mluti_view_loss": mv_loss_all, "cfloss": mv_cfloss,
                "mv_ignorance": mv_loss[2]["ignorance"], "mv_confusion": mv_confusion}

    def multi_view_fusion(self, opinion):
        view_num = len(opinion)
        belief = opinion[0]["belief"]
        ignorance = opinion[0]["ignorance"]
        plausibility = opinion[0]["plausibility"]
        for view in range(view_num - 1):
            pl_cons = torch.maximum(plausibility, opinion[view + 1]["plausibility"])
            # b_cons = torch.minimum(belief, opinion[view + 1]["belief"])
            # b_res_A = belief - b_cons
            # b_res_B = opinion[view + 1]["belief"] - b_cons
            # b_comp = b_res_A * opinion[view + 1]["ignorance"] + b_res_B * ignorance
            pl_res_A = pl_cons - plausibility
            pl_res_B = pl_cons - opinion[view + 1]["plausibility"]
            pl_comp = pl_res_A * opinion[view + 1]["ignorance"] + pl_res_B * ignorance
            # b_comp = pl_res_A * opinion[view + 1]["ignorance"] + pl_res_B * ignorance
            pl_fused = pl_cons - pl_comp
            alpha, opinions = form_opinion_general(
                pl_fused,
                belief.size(0),
                self.classes,
            )
            belief = opinions["belief"]
            ignorance = opinions["ignorance"]
            plausibility = opinions["plausibility"]
        belief_all = torch.sum(belief, dim=1).unsqueeze(1)
        confusion = torch.ones(belief.shape[0], 1).cuda() - belief_all - ignorance
        return alpha, belief, ignorance, confusion, plausibility

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

def normalize(x):
    """
    :param x: input data
    :return: normalization
    """
    scaler = MinMaxScaler((0, 1))
    norm_x = scaler.fit_transform(x, 0)

    return norm_x


class FeatureExtractor(nn.Module):
    def __init__(self, view_dims, hidden_dim=64):
        super(FeatureExtractor, self).__init__()
        # 使用一个全连接层提取特征
        # self.layer = nn.Sequential(
        #     nn.Linear(view_dims, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU()
        # )
        # SCENE
        self.layer = nn.Sequential(
            nn.Linear(view_dims, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = x.cuda()
        return self.layer(x)


# 用于灵活识别的证据深度学习头部
class EvidentialHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(EvidentialHead, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(feature_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算每个类别的可信度
        plausibilities = self.sigmoid(self.fc(x))
        return plausibilities


# 证据模型，结合特征提取和证据头部
class EvidentialModel(nn.Module):
    def __init__(self, num_classes, dim, hidden_dim=128):
        super(EvidentialModel, self).__init__()
        self.feature_extractor = FeatureExtractor(dim, hidden_dim=hidden_dim)
        self.evidential_head = EvidentialHead(hidden_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        plausibilities = self.evidential_head(features)
        return plausibilities


# 用于灵活识别的损失函数
def evidential(plausibilities, labels, num_classes, lambda_reg=1.0, lambda_kl=1, epoch=1,
                    for_train=True, use_pl_alpha=False):
    one_minus_pl = 1 - plausibilities
    # logits_dict
    # batch_size, K, 2
    beliefs_log = torch.empty(size=(plausibilities.shape[0], num_classes, 2)).to('cuda')
    # beliefs_log = F.log_softmax(logits_dict, dim=-1)
    # combine plausibilities and one_minus_pl into beliefs_log
    beliefs_log[:, :, 0] = plausibilities
    beliefs_log[:, :, 1] = one_minus_pl
    beliefs_log = F.log_softmax(beliefs_log, dim=-1)
    ignorance_log = torch.sum(beliefs_log[:, :, 1], dim=1, keepdim=True)
    bel_term_log = beliefs_log[:, :, 0] - beliefs_log[:, :, 1]
    belief_log = ignorance_log + bel_term_log
    belief = torch.exp(belief_log)
    belief_sum = belief.sum(-1, keepdim=True)
    uncertainty = 1.0 - belief_sum

    # ^ alpha of singleton belief
    S = num_classes / (uncertainty + 1e-6)
    alpha = belief * S + 1.0

    # ^ alpha of plausibility
    beliefs = torch.exp(beliefs_log)
    alpha_pl = beliefs[:, :, 0] * S + 1.0

    ignorance = torch.exp(ignorance_log)
    plausibility = torch.exp(beliefs_log[:, :, 0])
    bel_term = torch.exp(bel_term_log)
    if for_train:
        # ^ For regularization term
        plausibility_max = 1.0 - ignorance
        opinions = {
            "S": S,
            "uncertainty": uncertainty.squeeze(-1),
            "belief": belief,
            "plausibility": plausibility,
            "alpha_pl": alpha_pl,
            "belief_ratio": bel_term,
            "plausibility_max": plausibility_max,
        }
    else:
        # ^ belief for binary confusion ({1, 2} for class 1 and 2)
        confusion_bi = ignorance[..., None].repeat(1, num_classes, num_classes)
        conf_term_1 = beliefs[:, :, 0].unsqueeze(1).repeat(1, num_classes, 1) / (
            beliefs[:, :, 1].unsqueeze(1).repeat(1, num_classes, 1) + 1e-6
        )
        conf_term_2 = beliefs[:, :, 0].unsqueeze(-1).repeat(1, 1, num_classes) / (
            beliefs[:, :, 1].unsqueeze(-1).repeat(1, 1, num_classes) + 1e-6
        )
        confusion_bi = confusion_bi * conf_term_1 * conf_term_2
        confusion_bi = (1.0 - torch.eye(num_classes).to('cuda')) * confusion_bi

        # ^ plausibility for binary confusion ({1, 2}, {1, 2, 3}, {1, 2, 4}... for class 1 and 2)
        confusion_bi_pl = torch.matmul(
            beliefs[:, :, 0].unsqueeze(-1), beliefs[:, :, 0].unsqueeze(1)
        )
        confusion_bi_pl = (1.0 - torch.eye(num_classes).to('cuda')) * confusion_bi_pl

        # ^ total confusion
        confusion = uncertainty - ignorance

        # ^ class-wise confusion
        confusion_class = beliefs[:, :, 0] - belief
        sample_cnt = -1
        # ^ sample alpha for large classes (do not use for cifar)
        if sample_cnt != -1:
            assert sample_cnt + 1 <= num_classes
            sample_w = torch.ones((num_classes, num_classes)).to('cuda')
            sample_w = sample_w * (1.0 - torch.eye(num_classes, num_classes).to('cuda'))

            # [class_num, sample_cnt]
            sample_idx = (
                torch.multinomial(sample_w, sample_cnt)
                .unsqueeze(0)
                .repeat(plausibilities.shape[0], 1, 1)
            )
            sampled_bel_prod = torch.prod(
                beliefs[:, :, 1]
                .unsqueeze(1)
                .repeat(1, num_classes, 1)
                .gather(-1, sample_idx),
                dim=-1,
            )
            belief_samp = beliefs[:, :, 0] * sampled_bel_prod
            alpha = belief_samp * S + 1.0

        # ^ belief mass for dual prediction
        dual_belief = (
            confusion_bi
            + belief.unsqueeze(-1).repeat(1, 1, num_classes)
            + belief.unsqueeze(1).repeat(1, num_classes, 1)
        )
        dual_belief = (1.0 - torch.eye(num_classes).to('cuda')) * dual_belief

        opinions = {
            "S": S,
            "uncertainty": uncertainty.squeeze(-1),
            "ignorance": ignorance.squeeze(-1),
            "confusion": confusion.squeeze(-1),
            "confusion_bi": confusion_bi,
            "confusion_bi_pl": confusion_bi_pl,
            "confusion_cls": confusion_class,
            "belief": belief,
            "plausibility": beliefs[:, :, 0],
            "alpha_pl": alpha_pl,
            "dual_belief": dual_belief,
            "beliefs": beliefs,
            "belief_ratio": bel_term,
        }
    if use_pl_alpha:
        alpha = opinions["alpha_pl"]
    return alpha, opinions