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
from model.main_model import weak_main_model as main_model
from utils import AverageMeter, Prepare_logger, get_and_save_args
from utils.Recorder import Recorder
from dataset.AVE_dataset_weak import AVEDataset

# =================================  seed config ============================
# 设置随机种子以确保实验可重复性
SEED = 666
random.seed(SEED)
np.random.seed(seed=SEED)
torch.manual_seed(seed=SEED)
torch.cuda.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True  # 保证卷积结果确定性
torch.backends.cudnn.benchmark = False  # 关闭benchmark以获得可重复结果

# =============================================================================
# 加载配置文件
config_path = 'configs/weak.json'
with open(config_path) as fp:
    config = json.load(config_path)
print(config)

def main():
    # 全局变量声明
    global args, logger, writer, dataset_configs
    # 统计变量：记录最佳准确率和对应的epoch
    global best_accuracy, best_accuracy_epoch
    best_accuracy, best_accuracy_epoch = 0, 0
    
    # 配置参数处理
    dataset_configs = get_and_save_args(parser)  # 获取并保存参数
    parser.set_defaults(**dataset_configs)  # 设置默认参数
    args = parser.parse_args()  # 解析命令行参数
    
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
        logger.info(f'\nLog file will be save in {args.snapshot_pref}/Eval.log.')

    '''数据集准备'''
    # 训练数据加载器
    train_dataloader = DataLoader(
        AVEDataset('./data/', split='train'),  # 训练集
        batch_size=args.batch_size,
        shuffle=True,  # 训练时打乱数据
        num_workers=8,  # 多进程加载数据
        pin_memory=True  # 锁页内存，加速GPU传输
    )

    # 测试数据加载器
    test_dataloader = DataLoader(
        AVEDataset('./data/', split='test'),  # 测试集
        batch_size=args.test_batch_size,
        shuffle=False,  # 测试时不打乱数据
        num_workers=8,
        pin_memory=True
    )

    '''模型设置'''
    mainModel = main_model(config["model"])  # 创建主模型
    mainModel = nn.DataParallel(mainModel).cuda()  # 多GPU并行处理
    learned_parameters = mainModel.parameters()  # 获取模型参数
    optimizer = torch.optim.Adam(learned_parameters, lr=args.lr)  # Adam优化器
    
    # 学习率调度器
    # scheduler = CosineAnnealingLR(optimizer, T_max=40)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.5)  # 多步长衰减
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss().cuda()  # 二分类交叉熵损失
    # criterion_event = nn.CrossEntropyLoss().cuda()
    criterion_event = nn.MultiLabelSoftMarginLoss().cuda()  # 多标签分类损失

    '''从检查点恢复训练'''
    if os.path.isfile(args.resume):
        logger.info(f"\nLoading Checkpoint: {args.resume}\n")
        mainModel.load_state_dict(torch.load(args.resume))  # 加载模型权重
    elif args.resume != "" and (not os.path.isfile(args.resume)):
        raise FileNotFoundError  # 如果指定了resume但文件不存在则报错

    '''仅进行评估模式'''
    if args.evaluate:
        logger.info(f"\nStart Evaluation..")
        validate_epoch(mainModel, test_dataloader, criterion, criterion_event, epoch=0, eval_only=True)
        return

    '''Tensorboard和代码备份'''
    writer = SummaryWriter(args.snapshot_pref)  # 创建Tensorboard写入器
    recorder = Recorder(args.snapshot_pref, ignore_folder="Exps/")  # 创建记录器
    recorder.writeopt(args)  # 记录参数

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
                    task='WeaklySupervised',
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
    model.double()  # 使用双精度
    optimizer.zero_grad()  # 清空梯度

    # 遍历训练数据
    for n_iter, batch_data in enumerate(train_dataloader):
        data_time.update(time.time() - end_time)
        
        '''前向传播'''
        visual_feature, audio_feature, labels = batch_data
        labels = labels.double().cuda()  # 标签转移到GPU
        
        # 模型前向计算
        is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)

        # 计算损失
        loss_event_class = criterion_event(event_scores, labels)
        loss = loss_event_class
        loss.backward()  # 反向传播

        '''计算准确率'''
        acc = torch.tensor([0])  # 这里准确率计算被简化了
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

        '''Tensorboard记录'''
        writer.add_scalar('Train_data/loss', losses.val, epoch * len(train_dataloader) + n_iter + 1)

    # 记录epoch平均损失
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
        labels = labels.cuda()
        bs = visual_feature.size(0)  # 批次大小
        
        # 模型前向计算
        is_event_scores, raw_logits, event_scores = model(visual_feature, audio_feature)
        
        # 计算准确率
        acc = compute_accuracy_supervised(is_event_scores, raw_logits, labels)
        accuracy.update(acc.item(), bs)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

    # Tensorboard记录验证准确率
    if not eval_only:
        writer.add_scalar('Val_epoch/Accuracy', accuracy.avg, epoch)

    # 记录验证结果
    logger.info(
        f'**************************************************************************\t'
        f"\tEvaluation results (acc): {accuracy.avg:.4f}%."
    )

    return accuracy.avg


def compute_accuracy_supervised(is_event_scores, event_scores, labels):
    """计算监督学习的准确率"""
    # 获取目标标签
    _, targets = labels.max(-1)
    
    # 处理事件得分
    is_event_scores = is_event_scores.sigmoid()  # sigmoid激活
    scores_pos_ind = is_event_scores > 0.5  # 二值化
    scores_mask = scores_pos_ind == 0
    
    # 事件分类
    _, event_class = event_scores.max(-1)
    
    # 构建预测结果
    pred = scores_pos_ind.long()
    pred *= event_class[:, None]
    pred[scores_mask] = 28  # 28表示背景类别
    
    # 计算准确率
    correct = pred.eq(targets)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def compute_accuracy_weak(event_scores, labels):
    """计算弱监督学习的准确率"""
    # 获取预测结果
    _, pred = event_scores.max(-1)
    pred = pred.transpose(1, 0).contiguous()
    
    # 获取目标标签
    _, target = labels.max(-1)
    
    # 计算准确率
    correct = pred.eq(target)
    correct_num = correct.sum().double()
    acc = correct_num * (100. / correct.numel())

    return acc


def save_checkpoint(state_dict, top1, task, epoch):
    """保存模型检查点"""
    model_name = f'{args.snapshot_pref}/model_epoch_{epoch}_top1_{top1:.3f}_task_{task}_best_model.pth.tar'
    torch.save(state_dict, model_name)


if __name__ == '__main__':
    main()