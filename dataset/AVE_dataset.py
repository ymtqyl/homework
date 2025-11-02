import os
import h5py  # 用于处理HDF5格式文件的库，适合存储大量数值数据
import torch
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据加载工具


class AVEDataset(Dataset):
    """
    自定义数据集类，用于加载视听（Audio-Visual）数据
    继承自PyTorch的Dataset类，需实现__getitem__和__len__方法
    """

    def __init__(self, data_root, split='train'):
        """
        初始化数据集

        参数:
            data_root: 数据根目录路径，包含所有HDF5格式的数据文件
            split: 数据集划分类型，可选'train'（训练集）、'val'（验证集）或'test'（测试集）
        """
        super(AVEDataset, self).__init__()  # 调用父类构造函数
        self.split = split  # 保存数据集划分类型

        # 构建各类数据的文件路径
        self.visual_feature_path = os.path.join(data_root, 'visual_feature.h5')  # 视觉特征文件路径
        self.audio_feature_path = os.path.join(data_root, 'audio_feature.h5')  # 音频特征文件路径
        self.labels_path = os.path.join(data_root, 'labels.h5')  # 标签文件路径
        self.sample_order_path = os.path.join(data_root, f'{split}_order.h5')  # 当前划分的样本顺序文件路径

        self.h5_isOpen = False  # 标记HDF5文件是否已打开，初始为未打开

    def __getitem__(self, index):
        """
        根据索引获取单个样本数据

        参数:
            index: 样本索引（在当前划分中的相对索引）

        返回:
            visual_feat: 视觉特征
            audio_feat: 音频特征
            label: 样本标签
        """
        # 首次访问数据时打开所有HDF5文件（延迟打开，节省资源）
        if not self.h5_isOpen:
            # 打开视觉特征文件，读取'avadataset'数据集
            self.visual_feature = h5py.File(self.visual_feature_path, 'r')['avadataset']
            # 打开音频特征文件，读取'avadataset'数据集
            self.audio_feature = h5py.File(self.audio_feature_path, 'r')['avadataset']
            # 打开标签文件，读取'avadataset'数据集
            self.labels = h5py.File(self.labels_path, 'r')['avadataset']
            # 打开样本顺序文件，读取'order'数据集（记录当前划分的样本在总数据中的索引）
            self.sample_order = h5py.File(self.sample_order_path, 'r')['order']
            self.h5_isOpen = True  # 标记文件已打开

        # 获取当前索引对应的原始数据索引（因为样本顺序是按划分定义的）
        sample_index = self.sample_order[index]

        # 根据原始索引提取对应特征和标签
        visual_feat = self.visual_feature[sample_index]
        audio_feat = self.audio_feature[sample_index]
        label = self.labels[sample_index]

        return visual_feat, audio_feat, label

    def __len__(self):
        """
        获取当前划分的样本总数

        返回:
            sample_num: 样本数量
        """
        # 打开样本顺序文件，获取'order'数据集的长度（即样本数量）
        f = h5py.File(self.sample_order_path, 'r')
        sample_num = len(f['order'])
        f.close()  # 读取后关闭文件，避免资源占用
        return sample_num