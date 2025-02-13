import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from loss import create_loss
from toydata import get_3d_gaussian_data, project_3d_to_2d, new_3d_gaussian_data, twod_gaussian_data
from dataset import MultiViewData


class EMV(nn.Module):
    def __init__(self, classes, dims):
        super(EMV, self).__init__()
        self.views = np.size(dims)
        self.classes = classes
        self.emv = nn.ModuleList([EvidentialModel(self.classes, dims[v], dims[v]) for v in range(self.views)])

    def forward(self, x1):
        pl_all = []
        h1 = dict()
        loss = []
        for v in range(self.views):
            pl = self.emv[v](x1[v])
            pl_all.append(pl)
            # 单视角损失
            # loss.append(self.compute_loss(ER, s[v]))
        # loss = torch.stack(loss).mean()
        # 多视角 pl 融合：求平均 or 取最大

        # s_fusion = torch.stack(s)
        # s_fusion = torch.sum(s_fusion, dim=0) / self.views
        # loss += self.compute_loss(ER, s_fusion)
        # # loss.append(self.compute_loss(ER, s_fusion))
        # # loss = torch.stack(loss).sum()
        return loss, h1
def normalize(x):
    """
    :param x: input data
    :return: normalization
    """
    scaler = MinMaxScaler((0, 1))
    norm_x = scaler.fit_transform(x, 0)

    return norm_x


# 特征提取网络（使用预训练ResNet-18）
class FeatureExtractor(nn.Module):
    def __init__(self, view_dims, hidden_dim=64):
        super(FeatureExtractor, self).__init__()
        # 使用一个全连接层提取特征
        self.layer = nn.Sequential(
            nn.Linear(view_dims, hidden_dim),
            nn.BatchNorm1d(hidden_dim))

    def forward(self, x):
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
def evidential_loss(plausibilities, labels, num_classes, lambda_reg=1.0, lambda_kl=1, epoch=1,
                    for_train=True, use_pl_alpha=False):
    one_minus_pl = 1 - plausibilities
    # logits_dict
    # batch_size, K, 2
    beliefs_log = torch.empty(size=(batch_size, num_classes, 2)).to('cuda')
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
                .repeat(batch_size, 1, 1)
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


    """
    # 计算单类置信度（singleton beliefs）
    beliefs = torch.zeros(plausibilities.shape[0], num_classes).cuda()
    for i in range(num_classes):
        prod_term = torch.prod(one_minus_pl[:, torch.arange(plausibilities.size(1)) != i], dim=1)
        beliefs[:, i] = plausibilities[:, i] * prod_term

    # 无知（空集的质量分配）
    ignorance = torch.prod(one_minus_pl, dim=1)  # 方法中的公式7

    # 从置信度计算狄利克雷分布的浓度参数（公式10）
    alpha = (beliefs * num_classes) / (1 - beliefs.sum(dim=1, keepdim=True)) + 1

    # 狄利克雷损失（公式9）
    edl_loss = 0.0
    for i in range(num_classes):
        edl_loss += torch.sum(labels[:, i] * (torch.log(alpha.sum(dim=1)) - torch.log(alpha[:, i])))

    # 正则化损失（公式11）
    reg_loss = 0.0
    for i in range(num_classes):
        reg_loss += torch.sum(labels[:, i] * (plausibilities[:, i] - (1 - ignorance)) ** 2)

    # KL散度损失（公式12）
    alpha_tilde = labels + (1 - labels) * alpha
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)

    kl_loss = (
            torch.lgamma(S_tilde)
            - torch.lgamma(torch.tensor(alpha_tilde.shape[1]))
            - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)
            + (
                    (alpha_tilde - 1)
                    * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))
            ).sum(dim=1, keepdim=True)
    )
    # kl_loss = nn.KLDivLoss()(F.log_softmax(alpha), torch.ones_like(alpha))

    # 总损失
    total_loss = edl_loss + lambda_reg * reg_loss + lambda_kl * kl_loss * min(1.0, epoch/10)
    
    return total_loss.mean(), beliefs, ignorance
    """


# 初始化模型
since = time.time()
num_classes = 3
num_epochs = 20

model = EvidentialModel(num_classes=num_classes, dim=2).cuda()
criterion = create_loss(class_num=num_classes,
                        annealing_epochs=num_epochs)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1)
batch_size = 128
num_classes = 3
view = 0

# 生成数据
# dataX, dataY = new_3d_gaussian_data()
dataX, dataY = twod_gaussian_data()

normal_vector = [[1, 1, 1], [1, -1, -1], [-1, -1, 1]]
dataY = torch.from_numpy(dataY)
# projections = []
projections = [torch.FloatTensor(
    dataX.astype(np.float32))]
# for i in range(3):
#     point_on_plane = [0, 0, 0]
#     projections.append(project_3d_to_2d(dataX, normal_vector[i], point_on_plane))
#     projections[i] = torch.FloatTensor(projections[i].astype(np.float32))
# x_dict = {view: feature for view, feature in enumerate(projections)}
x_dict = dict()
x_dict[0] = projections[0]
std = 1

data_set = MultiViewData(x_dict, dataY, view, std)
train_data, test_data = data_set.Split_Training_Test_Dataset(0.8)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
data_loaders = {
    "train": train_loader,
    "val": test_loader,
}
# 训练循环
best_acc = 0.0
for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    for phase in ["train", "val"]:
        if phase == "train":
            print("Training")
            model.train()
        else:
            print("Testing")
            model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        # # 记录每个 epoch 的特征、标签、belief、ignorance
        # if epoch + 1 == num_epochs:
        #     x_all = torch.empty((0, 2)).cuda()
        #     beilef_all = torch.empty((0, num_classes)).cuda()
        #     ignorance_all = torch.empty(0).cuda()
        #     label_all = torch.empty(0).cuda()

        for batch in tqdm(data_loaders[phase]):
            data, y, labels = batch
            y = y.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                plausibilities = model(data)
                one_minus_pl = 1 - plausibilities
                pl = torch.empty(size=(plausibilities.shape[0], num_classes, 2)).to('cuda')
                pl[:, :, 0] = plausibilities
                pl[:, :, 1] = one_minus_pl
                loss, loss_dict, opinion = criterion(pl, labels.type(torch.LongTensor), epoch)
                # print('debug')

                # loss, beliefs, ignorance = evidential_loss(plausibilities, y, num_classes, epoch)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * data.size(0)
                beliefs = opinion['belief']
                running_corrects += torch.sum(beliefs.argmax(dim=1) == labels)

        if scheduler is not None:
            if phase == "train":
                scheduler.step()

        epoch_loss = running_loss / len(data_loaders[phase].dataset)
        epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)
        print("{} loss: {:.4f} acc: {:.4f}".format(phase.capitalize(), epoch_loss, epoch_acc))
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(
        time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))


print(model.evidential_head.fc.weight)
# 画出数据点分布及 belief、confusion、ignorance 分布
model.load_state_dict(best_model_wts)
model.eval()
x = np.linspace(-10, 10, 1000)  # 在0到1之间采样100个点
y = np.linspace(-10, 10, 1000)
xx, yy = np.meshgrid(x, y)

# 将网格点变为模型可处理的输入格式 (10000, 2)
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 将采样点转换为 PyTorch 张量
grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32).cuda()/std
grid_points_tensor = grid_points_tensor/grid_points_tensor.norm(dim=-1, keepdim=True)



# 通过模型进行预测
with torch.no_grad():
    plausibilities = model(grid_points_tensor)
    one_minus_pl = 1 - plausibilities
    # 计算单类置信度（singleton beliefs）
    beliefs = torch.zeros(plausibilities.shape[0], num_classes).cuda()
    for i in range(num_classes):
        prod_term = torch.prod(one_minus_pl[:, torch.arange(plausibilities.size(1)) != i], dim=1)
        beliefs[:, i] = plausibilities[:, i] * prod_term

    # 无知（空集的质量分配）
    ignorance = torch.prod(one_minus_pl, dim=1)  # 方法中的公式7
    confusion = 1 - beliefs.sum(dim=1) - ignorance
# 将预测结果转换回 numpy 格式
beliefs_np = beliefs.cpu().numpy()
ignorance_np = ignorance.cpu().numpy()
confusion_np = confusion.cpu().numpy()

# 将结果重整为 (100, 100) 的网格格式
beliefs_reshaped = beliefs_np.reshape(xx.shape[0], xx.shape[1], -1)  # 对应每个类别的 belief
ignorance_reshaped = ignorance_np.reshape(xx.shape)
confusion_reshaped = confusion_np.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, ignorance_reshaped, cmap=plt.cm.Blues)
plt.colorbar(label="Ignorance")
plt.title("Ignorance Heatmap")
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(projections[view][:, 0], projections[view][:, 1], c=dataY, cmap=ListedColormap(['green', 'purple', 'red']), edgecolor='k', s=20)
plt.show()

for i in range(num_classes):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, beliefs_reshaped[:, :, i], cmap=plt.cm.Blues)
    plt.colorbar(label=f"Belief for class {i}")
    plt.title(f"Belief Heatmap for class {i}")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.scatter(projections[view][:, 0], projections[view][:, 1], c=dataY, cmap=ListedColormap(['green', 'purple', 'red']),
                edgecolor='k', s=20)
    plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, confusion_reshaped, cmap=plt.cm.Blues)
plt.colorbar(label="Confusion")
plt.title("Confusion Heatmap")
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(projections[view][:, 0], projections[view][:, 1], c=dataY, cmap=ListedColormap(['green', 'purple', 'red']), edgecolor='k', s=20)
plt.show()

