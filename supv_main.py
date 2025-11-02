import os
import time
import random
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np
from configs.opts import parser
from model.main_model import supv_main_model as main_model  # 导入全监督主模型
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset import AVEDataset  # 导入全监督数据集
import torch.nn.functional as F

# =================================  seed config ============================
# 设置随机种子以确保实验可重复性
SEED = 43
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True  # 保证卷积结果确定性
torch.backends.cudnn.benchmark = False  # 关闭benchmark以获得可重复结果

# 加载配置文件
config_path = 'configs/main.json'
with open(config_path) as fp:
    config = json.load(config_path)
print(config)
# =============================================================================

def AVPSLoss(av_simm, soft_label):
    """音频-视觉对相似度损失函数，用于全监督设置
    参考论文中的公式(8,9)
    
    参数:
        av_simm: 音频-视觉相似度得分 [batch_size, 10]
        soft_label: 软标签
    """
    relu_av_simm = F.relu(av_simm)  # 使用ReLU确保非负
    sum_av_simm = torch.sum(relu_av_simm, dim=-1, keepdim=True)  # 对时间维度求和
    avg_av_simm = relu_av_simm / (sum_av_simm + 1e-8)  # 归一化得到概率分布
    loss = nn.MSELoss()(avg_av_simm, soft_label)  # 计算均方误差损失
    return loss


def main():
    # 全局变量声明
    global args, logger, writer, dataset_configs
    # 统计变量：记录最佳准确率和对应的epoch
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    
    # 配置参数处理
    dataset_configs = get_and_save_args(parser)
    parser.set_defaults(**dataset_configs)
    args = parser.parse_args()
    
    # GPU设置
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    '''创建快照目录用于保存代码和模型'''
    if not os.path.exists(args.snapshot_pref):
        os.makedirs(args.snapshot_pref)

    # 如果提供了resume路径，则更新快照目录
    if os.path.isfile(args.resume):
        args.snapshot_pref = os.path.dirname(args.resume)

    # 准备日志记录器
    logger = Prepare_logger(args, eval=args.evaluate)

    # 日志记录
    if not args.evaluate:
        logger.info(f'\nCreating folder: {args.snapshot_pref}')
        logger.info('\nRuntime args\n\n{}\n'.format(json.dumps(vars(args), indent=4)))
    else:
        logger.info(f'\nLog file will be save in a {args.snapshot_pref}/Eval.log.')

    '''数据集准备'''
    # 训练数据加载器
    train_dataloader = DataLoader(
        AVEDataset('./data/', split='train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    # 测试数据加载器
    test_dataloader = DataLoader(
        AVEDataset('./data/', split='test'),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    '''模型设置'''
    mainModel = main_model(config['model'])  # 创建全监督主模型
    mainModel = nn.DataParallel(mainModel).cuda()  # 多GPU并行处理
    learned_parameters = mainModel.parameters()
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)  # Adam优化器
    
    # 学习率调度器
    # scheduler = StepLR(optimizer, step_size=40, gamma=0.2)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)  # 多步长衰减
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()  # 二分类交叉熵损失（带sigmoid）
    criterion_event = nn.CrossEntropyLoss().cuda()  # 多分类交叉熵损失

    '''从检查点恢复训练'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError

    '''仅进行评估模式'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard和代码备份'''
    writer = SummaryWriter(args.snapshot_pref)
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")
    recorder.writeopt(args)

    '''训练和测试循环'''
    for epoch in range(args.n_epoch):
        # 训练一个epoch
        loss = train_epoch(mainModel, train_dataloader, criterion, criterion_event, optimizer, epoch)

        # 定期验证或最后一个epoch进行验证
        if ((epoch + 1) % args.eval_freq == 0) or (epoch == args.n_epoch - 1):
            acc = validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch)
            # 更新最佳准确率
            if acc > best_accuracy:
                best_accuracy = acc
                best_accuracy_epoch = epoch
                # 保存最佳模型检查点
                save_checkpoint(
                    mainModel.state_dict(),
                    top1=best_accuracy,
                    task='Supervised',  # 全监督任务
                    epoch=epoch + 1,
                )
            print("-----------------------------")
            print("best acc and epoch:", best_accuracy, best_accuracy_epoch)
            print("-----------------------------")
        scheduler.step()  # 更新学习率


def train_epoch(model, train_dataloader, criterion, criterion_event, optimizer, epoch):
    # 初始化计量器
    batch_time = AverageMeter()  # 批次时间
    data_time = AverageMeter()   # 数据加载时间
    losses = AverageMeter()      # 损失值
    train_acc = AverageMeter()   # 训练准确率
    end_time = time.time()

    model.train()  # 设置模型为训练模式
    # 注意：这里将模型设置为双精度，因为提取的特征是双精度类型
    # 这也会使模型大小翻倍
    model.double()
    optimizer.zero_grad()  # 清空梯度

    # 遍历训练数据
    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        
        '''前向传播'''
        visual_feature, audio_feature, labels = batch_data
        # 如果模型使用单精度，需要转换特征类型
        # visual_feature = visual_feature.float()
        # audio_feature = audio_feature.float()
        labels = labels.double().cuda()
        
        # 模型前向计算 - 全监督模型输出更多组件
        is_event_scores, event_scores, audio_visual_gate, av_score = model(visual_feature, audio_feature)
        
        # 调整输出形状
        is_event_scores = is_event_scores.transpose(1, 0).squeeze().contiguous()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze().contiguous()

        '''损失计算'''
        # 处理标签：去掉背景类别，只保留前景类别
        labels_foreground = labels[:, :, :-1]  # [32, 10, 28]
        labels_BCE, labels_evn = labels_foreground.max(-1)  # 获取事件存在性和事件类型
        
        # 获取主要事件标签
        labels_event, _ = labels_evn.max(-1)
        
        # 计算各个损失组件
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())  # 事件存在性损失
        label_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())  # 音频视觉门控损失
        loss_cas = criterion_event(av_score, labels_event.cuda())  # 分类激活序列损失
        loss_event_class = criterion_event(event_scores, labels_event.cuda())  # 事件分类损失
        
        # 总损失 = 事件存在性损失 + 门控损失 + 事件分类损失 + CAS损失
        loss = loss_is_event + label_is_gate + loss_event_class + loss_cas
        loss.backward()  # 反向传播

        '''计算准确率'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        train_acc.update(acc.item(), visual_feature.size(0) * 10)

        '''梯度裁剪'''
        if args.clip_gradient is not None:
            total_norm = clip_grad_norm_(model.parameters(), args.clip_gradient)

        '''参数更新'''
        optimizer.step()
        optimizer.zero_grad()

        # 更新统计信息
        losses.update(loss.item(), visual_feature.size(0) * 10)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        '''Tensorboard记录迭代损失'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

        '''在终端打印日志'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Train Epoch: [{epoch}][{n_iter}/{len(train_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {train_acc.val:.3f} ({train_acc.avg: .3f})'
            )

    '''Tensorboard记录epoch平均损失'''
    writer.add_scalar('Train_epoch_data/epoch_loss', losses.avg, epoch)

    return losses.avg


@torch.no_grad()  # 禁用梯度计算，节省内存
def validate_epoch(model, test_dataloader, criterion, criterion_event, epoch, eval_only=False):
    # 初始化计量器
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    end_time = time.time()

    model.eval()  # 设置模型为评估模式
    model.double()

    # 遍历测试数据
    for n_iter, batch_data in enumerate(test_dataloader):
        data_time.update(time.time() - end_time)

        '''前向传播'''
        visual_feature, audio_feature, labels = batch_data
        labels = labels.double().cuda()
        bs = visual_feature.size(0)  # 批次大小
        
        # 模型前向计算
        is_event_scores, event_scores, audio_visual_gate, _ = model(visual_feature, audio_feature)
        is_event_scores = is_event_scores.transpose(1, 0).squeeze()
        audio_visual_gate = audio_visual_gate.transpose(1, 0).squeeze()

        '''损失计算'''
        labels_foreground = labels[:, :, :-1]
        labels_BCE, labels_evn = labels_foreground.max(-1)
        labels_event, _ = labels_evn.max(-1)
        loss_is_event = criterion(is_event_scores, labels_BCE.cuda())
        loss_is_gate = criterion(audio_visual_gate, labels_BCE.cuda())
        loss_event_class = criterion_event(event_scores, labels_event.cuda())
        loss = loss_is_event + loss_event_class + loss_is_gate

        '''计算准确率'''
        acc = compute_accuracy_supervised(is_event_scores, event_scores, labels)
        accuracy.update(acc.item(), bs * 10)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        losses.update(loss.item(), bs * 10)

        '''在终端打印日志'''
        if n_iter % args.print_freq == 0:
            logger.info(
                f'Test Epoch [{epoch}][{n_iter}/{len(test_dataloader)}]\t'
                # f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'Prec@1 {accuracy.val:.3f} ({accuracy.avg:.3f})'
            )

    '''Tensorboard记录验证损失和准确率'''
    if not eval_only:
        writer.add_scalar('Val_epoch_data/epoch_loss', losses.avg, epoch)
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    # 记录验证结果
    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )
    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    """计算全监督学习的准确率
    
    参数:
        is_event_scores: 事件存在性得分 [batch_size, 10]
        event_scores: 事件分类得分 [batch_size, 29]
        labels: 真实标签 [batch_size, 10, 29]
    """
    # 获取目标标签（每个时间步的类别）
    _, targets = labels.max(-1)
    
    # 处理事件存在性得分：sigmoid激活并二值化
    is_event_scores = is_event_scores.sigmoid()
    scores_pos_ind = is_event_scores > 0.5  # 事件存在性判断
    scores_mask = scores_pos_ind == 0  # 事件不存在的掩码
    
    # 事件分类
    _, event_class = event_scores.max(-1)  # 获取每个样本的主要事件类别
    
    # 构建预测结果
    pred = scores_pos_ind.long()  # 转换为整数类型
    pred *= event_class[:, None]  # 将事件类别应用到存在事件的位置
    
    # 为不存在事件的位置设置背景类别（28）
    pred[scores_mask] = 28  # 28表示背景类别
    
    # 计算准确率
    correct = pred.eq(targets)  # 比较预测和真实标签
    correct_num = correct.sum().double()  # 计算正确预测数量
    acc = correct_num * (100. / correct.numel())  # 计算准确率百分比

    return acc


def save_checkpoint(state_dict, top1, task, epoch):
    """保存模型检查点"""
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()