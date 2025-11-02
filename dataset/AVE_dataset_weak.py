import os
import h5py  # 用于读写HDF5格式文件（适合存储大量多维数组）
import torch
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载核心类


class AVEDataset(Dataset):
    """
    视听（Audio-Visual）数据集类，继承自PyTorch的Dataset
    支持加载带标签的干净样本和噪声样本（仅背景），适用于训练和测试阶段
    """

    def __init__(self, data_root, split='train'):
        """
        初始化数据集，设置数据路径和基本参数

        参数:
            data_root: 数据根目录，包含所有HDF5格式的数据文件
            split: 数据集划分类型，可选'train'（训练集）或'test'（测试集）
        """
        super(AVEDataset, self).__init__()  # 调用父类构造函数
        self.split = split  # 记录数据集划分类型

        # 定义各类数据的文件路径
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')  # 干净样本的视觉特征
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')  # 干净样本的音频特征
        self.noisy_visual_feature_path = os.path.join(data_root, 'visual_feature_noisy.h5')  # 噪声样本（仅背景）的视觉特征
        self.noisy_audio_feature_path = os.path.join(data_root, 'audio_feature_noisy.h5')  # 噪声样本（仅背景）的音频特征

        # 标签文件路径
        self.labels_path = os.path.join(data_root, 'labels.h5')  # 原始标签（用于测试）
        self.dir_labels_path = os.path.join(data_root, 'mil_labels.h5')  # 视频级标签（MIL：多实例学习）
        self.dir_labels_bg_path = os.path.join(data_root, 'labels_noisy.h5')  # 噪声样本（仅背景）的标签

        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')  # 当前划分的样本顺序文件（记录样本索引）
        self.h5_isOpen = False  # 标记HDF5文件是否已打开（延迟加载用）

    def __getitem__(self, index):
        """
        根据索引获取单个样本（特征+标签）

        参数:
            index: 样本索引（在当前数据集中的相对位置）

        返回:
            visual_feat: 视觉特征
            audio_feat: 音频特征
            label: 样本标签
        """
        # 延迟打开HDF5文件：首次访问数据时才打开，减少初始化时的资源占用
        if not self.h5_isOpen:
            # 打开样本顺序文件（记录当前划分的样本索引）
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            # 打开干净样本的特征文件
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            # 打开干净样本的视频级标签
            self.clean_labels = h5py.File(self.dir_labels_path, 'r')['avadataset']

            # 训练集需要加载噪声样本（仅背景）的数据
            if self.split == 'train':
                self.negative_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']  # 噪声样本标签
                self.negative_visual_feature = h5py.File(self.noisy_visual_feature_path, 'r')['avadataset']  # 噪声样本视觉特征
                self.negative_audio_feature = h5py.File(self.noisy_audio_feature_path, 'r')['avadataset']  # 噪声样本音频特征

            # 测试集需要加载原始细粒度标签
            if self.split == 'test':
                self.labels = h5py.File(self.labels_path, 'r')['avadataset']  # 测试用原始标签

            self.h5_isOpen = True  # 标记文件已打开

        # 确定干净样本的数量（等于样本顺序文件的长度）
        clean_length = len(self.sample_order)

        # 索引超过干净样本数量时，返回噪声样本（仅训练集有效）
        if index >= clean_length:
            valid_index = index - clean_length  # 计算噪声样本在自身列表中的索引
            visual_feat = self.negative_visual_feature[valid_index]
            audio_feat = self.negative_audio_feature[valid_index]
            label = self.negative_labels[valid_index]
        else:
            # 索引在干净样本范围内：返回干净样本
            sample_index = self.sample_order[index]  # 获取原始数据中的索引
            visual_feat = self.visual_feature[sample_index]  # 提取视觉特征
            audio_feat = self.audio_feature[sample_index]  # 提取音频特征

            # 训练集用视频级标签，测试集用原始细粒度标签
            if self.split == 'train':
                label = self.clean_labels[sample_index]
            else:
                label = self.labels[sample_index]

        return visual_feat, audio_feat, label

    def __len__(self):
        """
        获取当前数据集的样本总数

        返回:
            length: 样本数量（训练集=干净样本数+噪声样本数；测试集=干净样本数）
        """
        if self.split == 'train':
            # 训练集长度 = 干净样本数 + 噪声样本数
            sample_order = h5py.File(self.sample_order_path, 'r')['order']  # 干净样本索引列表
            noisy_labels = h5py.File(self.dir_labels_bg_path, 'r')['avadataset']  # 噪声样本标签（用于获取数量）
            length = len(sample_order) + len(noisy_labels)
        elif self.split == 'test':
            # 测试集长度 = 干净样本数
            sample_order = h5py.File(self.sample_order_path, 'r')['order']
            length = len(sample_order)
        else:
            # 未实现其他划分（如验证集）
            raise NotImplementedError(f"未支持的数据集划分：{self.split}")

        return length