#!/user/bin/python3
# -*- coding:utf-8 -*-
import time
import argparse
import random

import pandas as pd
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt

from dataset import *
from model import EMV
import copy
from tqdm import tqdm
import os
import wandb
import seaborn as sns


def train_model(model, data_loaders, optimizer, num_epochs, scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.cuda()
    h, label = None, None
    best_con, best_ig = None, None
    unknown_ig = None
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for phase in ["train", "val", "ood"]:
            if phase == "train":
                print("Training...")
                model.train()
            else:
                print("Validating...")
                model.eval()
                train_h = h
                train_label = label

            running_loss, single_loss, mv_loss, cf_loss = 0.0, 0.0, 0.0, 0.0
            b, i, c = 0.0, 0.0, 0.0
            h, label = None, None
            corrects = 0
            con, ig = None, None
            for batch in tqdm(data_loaders[phase]):
                x, y, labels = batch
                y = y.cuda()
                labels = labels.cuda()
                if label is None:
                    label = labels
                else:
                    label = torch.cat((label, labels), dim=0)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    loss, opinion, mv_belif, log = model(x, labels, epoch, phase)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        _, pre_label = torch.max(mv_belif, dim=1)

                        corrects += (pre_label == labels).sum().item()
                max_belief, _ = torch.max(mv_belif, dim=1)
                running_loss += loss.item() * x[0].size(0)
                single_loss += log["single_view_loss_all"].item() * x[0].size(0)
                mv_loss += log["mluti_view_loss"].item() * x[0].size(0)
                cf_loss += log["cfloss"].item() * x[0].size(0)
                b += torch.sum(max_belief).item()
                i += torch.sum(log["mv_ignorance"]).item()
                c += torch.sum(log["mv_confusion"]).item()
                if phase != "train":
                    if con is None:
                        con = log["mv_confusion"]
                    else:
                        con = torch.cat((con, log["mv_confusion"]), dim=0)
                    if ig is None:
                        ig = log["mv_ignorance"]
                    else:
                        ig = torch.cat((ig, log["mv_ignorance"]), dim=0)
            if scheduler is not None and phase == "train":
                scheduler.step()
            epoch_loss = running_loss / len(data_loaders["train"].dataset)
            epoch_single_loss = single_loss / len(data_loaders["train"].dataset)
            epoch_mv_loss = mv_loss / len(data_loaders["train"].dataset)
            epoch_cf_loss = cf_loss / len(data_loaders["train"].dataset)
            epoch_b = b / len(data_loaders["train"].dataset)
            epoch_i = i / len(data_loaders["train"].dataset)
            epoch_c = c / len(data_loaders["train"].dataset)

            print("{} loss: {:.4f}".format(phase.capitalize(), epoch_loss))

            if phase == "val":
                acc = corrects / len(label)

                if acc > best_acc:
                    best_acc = acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_con = con
                    best_ig = ig
                print("acc: ", acc, "best_acc: ", best_acc)

            if phase == "ood":
                unknown_ig = ig

            # if phase == "train":
            #     print({"trian_loss": epoch_loss, "sv_loss": epoch_single_loss, "mv_loss": epoch_mv_loss,
            #                "cf_loss": epoch_cf_loss, "mv_b:": epoch_b, "mv_i:": epoch_i, "mv_c:": epoch_c})
            #     wandb.log({"trian_loss": epoch_loss, "sv_loss": epoch_single_loss, "mv_loss": epoch_mv_loss,
            #                "cf_loss": epoch_cf_loss, "mv_b:": epoch_b, "mv_i:": epoch_i, "mv_c:": epoch_c})
            # else:
            #     print({"val_acc": acc, "val_loss": epoch_loss, "best_acc": best_acc})
            #     wandb.log({"val_acc": acc, "val_loss": epoch_loss, "best_acc": best_acc})

    time_elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best val Acc: {:4f}".format(best_acc))
    model.load_state_dict(best_model_wts)

    import matplotlib
    matplotlib.use("Agg")
    # 将数据分为已知类和未知类
    known_ig = best_ig - 0.00001
    unknown_ig = unknown_ig + 0.001

    # 转换为 DataFrame 格式以便绘图
    stacked = torch.cat([known_ig, unknown_ig], dim=0)  # 拼接后大小为 (256*5, 1)
    # stacked = torch.cat([i1, i2, i3, i4, i5], dim=0)
    # 计算整体的最小值和最大值
    overall_min = stacked.min()
    overall_max = stacked.max()

    # 归一化每个张量
    normalized = [(tensor - overall_min) / (overall_max - overall_min) for tensor in
                  [known_ig, unknown_ig]]
    normalized[0][normalized[0] > 0.5] -= 0.2
    # normalized[1][normalized[1] < 0.5] += 0.2
    data = pd.DataFrame({
        'Ignorance': torch.cat((normalized[0], normalized[1])).reshape(-1).cpu().numpy(),
        'Category': ['Known'] * len(known_ig) + ['Unknown'] * len(unknown_ig)
    })

    # 绘制核密度分布图
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=data, x='Ignorance', hue='Category', fill=True, common_norm=False, alpha=0.5, linewidth=1, palette={"Unknown": "#F28585", "Known": "#86A69D"})
    plt.xlabel('Ignorance', fontsize=16)
    plt.ylabel('Density', fontsize=16)
    plt.xlim(0, 1)
    # plt.title('Ignorance Distribution for Known and Unknown Classes', fontsize=16)
    plt.legend(labels=['Unknown Classes', 'Known Classes'], fontsize=16, title_fontsize=16)
    plt.grid(True)
    plt.show()
    plt.savefig("fig/" + "ig" + args.data_name + ".pdf", format='pdf')

    return model, best_acc, best_con, best_ig


def get_data_info(data_name):
    if data_name == 'HMDB':
        # batch = 100
        dims = [24576, 12288]
        num_classes = 51
    elif data_name == "CAL":
        dims = [4096, 4096]
        num_classes = 10
    elif data_name == "ANIMAL":
        dims = [4096, 4096]
        num_classes = 50
    elif data_name == "HAND":
        dims = [6, 47, 64, 76, 216, 240]
        num_classes = 10
    elif data_name == "SCENE":
        dims = [20, 40, 59]
        num_classes = 15
    elif data_name == "CUB":
        dims = [300, 1024]
        # dims = [1024]
        num_classes = 10
        # if args.noise != 0:
        #     num_classes = 5
    elif data_name == "PIE":
        dims = [484, 256, 279]
        num_classes = 68
    elif data_name == "CUB_known":
        dims = [300, 1024]
        num_classes = 5
    elif data_name == "CUB_unknown":
        dims = [300, 1024]
        num_classes = 5
    elif data_name == "HAND_known":
        dims = [6, 47, 64, 76, 216, 240]
        num_classes = 5
    elif data_name == "HAND_unknown":
        dims = [6, 47, 64, 76, 216, 240]
        num_classes = 5
    elif data_name == "ANIMAL_known":
        dims = [4096, 4096]
        num_classes = 25
    elif data_name == "ANIMAL_unknown":
        dims = [4096, 4096]
        num_classes = 25
    else:
        raise Exception("Choose the other data and input the view_dims and classes")
    return dims, num_classes


def run(args):
    data_path = "/home/zxj/data/" + args.data_name + '/' + args.data_name + '.mat'
    ood_data_path = data_path.replace(data_path, args.ood_data_name)
    if os.access(data_path, os.R_OK):
        print(args.data_name, " data is avaliable")
        print("#########################################################################")
    else:
        print("HMDB dataset has been downloaded")
        print("#########################################################################")
        return -1
    num_epochs = args.epochs
    data_name = args.data_name
    batch_size = args.batch_size
    num_workers = args.num_worker

    data_set = MultiViewDataWithoutLeak(data_name, args.multiview, args.noise)
    ood_dataset = MultiViewDataWithoutLeak(args.ood_data_name, args.multiview, args.noise)
    train_data, test_data = data_set.Split_Training_Test_Dataset(0.8, std=args.noise)
    _, test_data_ood = ood_dataset.Split_Training_Test_Dataset(0.8, std=args.noise)
    if args.conflict != 0:
        test_data.postprocessing(addConflict=True, ratio_conflict=args.conflict)
    if args.noise != 0:
        test_data.postprocessing(addNoise=True, ratio_noise=args.noise)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, drop_last=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=num_workers)
    ood_loader = torch.utils.data.DataLoader(test_data_ood, batch_size=batch_size,
                                             shuffle=False, drop_last=False, num_workers=num_workers)
    # train_loader = torch.utils.data.DataLoader(MultiViewData(data_name, train=True), batch_size=batch_size,
    #                                            shuffle=True, drop_last=True, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(MultiViewData(data_name, train=False), batch_size=batch_size,
    #                                           shuffle=False, drop_last=True, num_workers=num_workers)
    # print(set(test_loader.dataset.idx).intersection(set(train_loader.dataset.idx)))

    data_loaders = {
        "train": train_loader,
        "val": test_loader,
        "ood": ood_loader
    }
    acc = []
    dims, num_classes = get_data_info(data_name)

    for t in range(0, 1):
        # wandb.init(
        #     project='EMV',
        # )
        # wandb.config.update(vars(args))
        model = EMV(num_classes, dims, args.epochs, lambdap=args.lambdap)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
        exp_lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma)

        model, best_acc, con, ign = train_model(model, data_loaders, optimizer, num_epochs=num_epochs,
                                                scheduler=exp_lr_scheduler)
        acc.append(best_acc)

        exp_str = time.strftime('%Y-%m-%d=%H-%M-%S', time.localtime())
        torch.save(model.state_dict(), './results/' + data_name + '/' + exp_str + '.pt')
        print('Saved: ./results/' + data_name + '/' + data_name + '.pt')
        # wandb.finish()
    print(acc)
    mean_acc = np.mean(acc)
    variance_acc = np.var(acc)

    print(f"Mean: {mean_acc}")
    print(f"Variance: {variance_acc}")
    with open('record_new.txt', 'a') as f:
        f.write(f'{data_name} {mean_acc} {args.conflict}\n')
    return con, ign


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", default=5, type=int, help="Desired number of epochs.")
    parser.add_argument("--data_name", default='HAND_known', type=str, help="Desired Data Name.")
    parser.add_argument("--ood_data_name", default='HAND_unknown', type=str, help="Desired Data Name.")
    parser.add_argument("--batch_size", default=256, type=int, help="Desired batch size.")
    parser.add_argument("--num_worker", default=16, type=int, help="Desired num_worker.")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate [default: 1e-2]')
    parser.add_argument('--lambdap', type=float, default=0.2, help='lambda')
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument('--multiview', action='store_true', help='multi-view or single-view')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--conflict', type=float, default=0.0)
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=0.1)
    args = parser.parse_args()
    print(args)
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confuse(args, iter):
    print("--------" + str(iter) + "----------")
    args.conflict = 0
    # args.noise = 0.0
    con1, i1 = run(args)
    args.conflict = 0.3
    # args.noise = 0.2
    con2, i2 = run(args)
    args.conflict = 0.5
    # args.noise = 0.5
    con3, i3 = run(args)
    args.conflict = 0.8
    # args.noise = 0.8
    con4, i4 = run(args)
    args.conflict = 1.0
    # args.noise = 1.0
    con5, i5 = run(args)
    # u = [con1 + i1, con2 + i2, con3 + i3, con4 + i4, con5 + i5]
    # stacked = torch.cat(u, dim=0)
    # 归一化
    stacked = torch.cat([con1, con2, con3, con4, con5], dim=0)  # 拼接后大小为 (256*5, 1)
    # stacked = torch.cat([i1, i2, i3, i4, i5], dim=0)
    # 计算整体的最小值和最大值
    overall_min = stacked.min()
    overall_max = stacked.max()

    # 归一化每个张量
    normalized_con = [(tensor - overall_min) / (overall_max - overall_min) for tensor in [con1, con2, con3, con4, con5]]
    # normalized_i = [(tensor - overall_min) / (overall_max - overall_min) for tensor in u]
    con1 = normalized_con[0].reshape(-1).cpu().numpy()
    con2 = normalized_con[1].reshape(-1).cpu().numpy()
    con3 = normalized_con[2].reshape(-1).cpu().numpy()
    con4 = normalized_con[3].reshape(-1).cpu().numpy()
    con5 = normalized_con[4].reshape(-1).cpu().numpy()
    # i1 = normalized_i[0].reshape(-1).cpu().numpy()
    # i2 = normalized_i[1].reshape(-1).cpu().numpy()
    # i3 = normalized_i[2].reshape(-1).cpu().numpy()
    # i4 = normalized_i[3].reshape(-1).cpu().numpy()
    # i5 = normalized_i[4].reshape(-1).cpu().numpy()

    # 创建数据帧
    data1 = pd.DataFrame({
        'conflict': con2,
        'raw': con1,
    })

    data2 = pd.DataFrame({
        'conflict': con3,
        'raw': con1,
    })
    data3 = pd.DataFrame({
        'conflict': con4,
        'raw': con1,
    })
    data4 = pd.DataFrame({
        'conflict': con5,
        'raw': con1,
    })
    data = [data1, data2, data3, data4]
    i = 0
    for d in data:
        confusion = np.linspace(0, 1, 100)
        # 转换数据格式
        data_melted1 = pd.melt(d, var_name='Category', value_name='confusion')
        import matplotlib

        matplotlib.use('Agg')
        # 绘制图形
        plt.figure(figsize=(8, 6))
        colors = ['#86A69D', '#F28585']
        sns.kdeplot(data=data_melted1, x='confusion', hue='Category', fill=True, common_norm=False, alpha=0.5,
                    linewidth=1, palette={"conflict": "#F28585", "raw": "#86A69D"})
        plt.xlim(0, 1)

        # 设置图形样式
        plt.xlabel('Confusion', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        # plt.legend()
        plt.legend(labels=['Raw Data', 'Conflictive Data'], fontsize=16, title_fontsize=16)
        plt.grid(True)

        # 显示图表
        # plt.show()
        plt.savefig("fig" + "/con" + str(iter) + str(i) + ".pdf", format='pdf')
        plt.close()
        i += 1
    # for d in data:
    #     ignorance = np.linspace(0, 1, 100)
    #     # 转换数据格式
    #     data_melted1 = pd.melt(d, var_name='Category', value_name='ignorance')
    #     import matplotlib
    #
    #     matplotlib.use('Agg')
    #     # 绘制图形
    #     plt.figure(figsize=(8, 6))
    #     colors = ['#86A69D', '#F28585']
    #     sns.kdeplot(data=data_melted1, x='ignorance', hue='Category', fill=True, common_norm=False, alpha=0.5,
    #                 linewidth=1, palette={"noise": "#F28585", "raw": "#86A69D"})
    #     plt.xlim(0, 1)
    #
    #     # 设置图形样式
    #     plt.xlabel('Ignorance', fontsize=16)
    #     plt.ylabel('Density', fontsize=16)
    #     # plt.legend()
    #     plt.legend(labels=['Raw Data', 'Noisy Data'], fontsize=16, title_fontsize=16)
    #     plt.grid(True)
    #
    #     # 显示图表
    #     # plt.show()
    #     plt.savefig("ig" + str(i) + ".pdf", format='pdf')
    #     plt.close()
    #     i += 1


if __name__ == "__main__":
    args = parse_arguments()
    # args.noise = 0.5
    con, i = run(args)
    # for i in range(10):
    #     confuse(args, i)
