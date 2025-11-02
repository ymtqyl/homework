import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention  # Transformer多头注意力模块
from torch.nn import ModuleList  # 用于存储多个子模块的列表
from torch.nn.init import xavier_uniform_  # 初始化方法（未直接使用）
from torch.nn import Dropout  # dropout层，防止过拟合
from torch.nn import Linear  # 全连接层
from torch.nn import LayerNorm  # 层归一化，稳定训练


class Encoder(Module):
    r"""Transformer编码器，由N个编码器层堆叠而成

    Args:
        encoder_layer: EncoderLayer类的实例（必需），定义单个编码器层结构
        num_layers: 编码器中层的数量（必需）
        norm: 层归一化组件（可选），用于输出的最终归一化
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        # 复制N个编码器层组成层列表（深拷贝确保层独立）
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers  # 记录层数
        self.norm = norm  # 存储归一化层

    def forward(self, src):
        r"""输入依次通过所有编码器层，输出最终特征

        Args:
            src: 输入特征，形状通常为 [序列长度, 批次大小, 特征维度]
        Returns:
            output: 编码后的特征，与输入形状一致
        """
        output = src

        # 逐层传递输入
        for i in range(self.num_layers):
            output = self.layers[i](output)

        # 若指定归一化层，对最终输出归一化
        if self.norm:
            output = self.norm(output)

        return output


class Decoder(Module):
    r"""Transformer解码器，由N个解码器层堆叠而成

    Args:
        decoder_layer: DecoderLayer类的实例（必需），定义单个解码器层结构
        num_layers: 解码器中层的数量（必需）
        norm: 层归一化组件（可选），用于输出的最终归一化
    """

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        # 复制N个解码器层组成层列表
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers  # 记录层数
        self.norm = norm  # 存储归一化层

    def forward(self, tgt, memory):
        r"""输入依次通过所有解码器层，结合编码器输出（memory）进行解码

        Args:
            tgt: 目标序列特征，形状 [目标序列长度, 批次大小, 特征维度]
            memory: 编码器输出特征，形状 [源序列长度, 批次大小, 特征维度]
        Returns:
            output: 解码后的特征，与tgt形状一致
        """
        output = tgt

        # 逐层传递输入
        for i in range(self.num_layers):
            output = self.layers[i](output, memory)

        # 若指定归一化层，对最终输出归一化
        if self.norm:
            output = self.norm(output)

        return output


class EncoderLayer(Module):
    r"""Transformer编码器层（借鉴自CMRAN），包含自注意力和前馈网络

    Args:
        d_model: 输入特征维度（必需）
        nhead: 多头注意力的头数（必需）
        dim_feedforward: 前馈网络中间层维度（默认1024）
        dropout: dropout概率（默认0.1）
        activation: 中间层激活函数（relu/gelu，默认relu）
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        # 自注意力模块（query/key/value均为输入自身）
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络：Linear -> Activation -> Dropout -> Linear
        self.linear1 = Linear(d_model, dim_feedforward)  # 升维
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)  # 降维

        # 层归一化（Pre-LN结构，归一化后再进入子层）
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)  # 自注意力输出dropout
        self.dropout2 = Dropout(dropout)  # 前馈网络输出dropout

        # 获取激活函数
        self.activation = _get_activation_fn(activation)

    def forward(self, src):
        r"""编码器层前向传播：自注意力 -> 残差连接+归一化 -> 前馈网络 -> 残差连接+归一化

        Args:
            src: 输入特征 [序列长度, 批次大小, d_model]
        Returns:
            src: 输出特征 [序列长度, 批次大小, d_model]
        """
        # 自注意力子层
        src2 = self.self_attn(src, src, src)[0]  # 自注意力输出（忽略注意力权重）
        src = src + self.dropout1(src2)  # 残差连接+dropout
        src = self.norm1(src)  # 层归一化

        # 前馈网络子层
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 前馈网络输出
        src = src + self.dropout2(src2)  # 残差连接+dropout
        src = self.norm2(src)  # 层归一化

        return src


class DecoderLayer(Module):
    r"""Transformer解码器层（借鉴自CMRAN），包含多头注意力（交叉注意力）和前馈网络

    Args:
        d_model: 输入特征维度（必需）
        nhead: 多头注意力的头数（必需）
        dim_feedforward: 前馈网络中间层维度（默认1024）
        dropout: dropout概率（默认0.1）
        activation: 中间层激活函数（relu/gelu，默认relu）
    """

    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        # 自注意力模块（目标序列内部注意力）
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # 多头交叉注意力模块（目标序列与编码器输出的注意力）
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        # 前馈网络（与编码器层结构一致）
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        # 层归一化（Pre-LN结构）
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)  # 交叉注意力输出dropout
        self.dropout2 = Dropout(dropout)  # 前馈网络输出dropout

        # 获取激活函数
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory):
        r"""解码器层前向传播：交叉注意力 -> 残差连接+归一化 -> 前馈网络 -> 残差连接+归一化

        Args:
            tgt: 目标序列特征 [目标长度, 批次大小, d_model]
            memory: 编码器输出特征 [源长度, 批次大小, d_model]
        Returns:
            tgt: 输出特征 [目标长度, 批次大小, d_model]
        """
        # 融合memory和tgt作为交叉注意力的key/value（自定义修改，增强交互）
        memory = torch.cat([memory, tgt], dim=0)

        # 交叉注意力子层（tgt为query，memory为key/value）
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout1(tgt2)  # 残差连接+dropout
        tgt = self.norm1(tgt)  # 层归一化

        # 前馈网络子层
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)  # 残差连接+dropout
        tgt = self.norm2(tgt)  # 层归一化

        return tgt


def _get_clones(module, N):
    r"""复制N个相同的模块，组成ModuleList（支持PyTorch模型管理）

    Args:
        module: 待复制的模块
        N: 复制次数
    Returns:
        ModuleList: 包含N个深拷贝模块的列表
    """
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    r"""根据字符串获取对应的激活函数

    Args:
        activation: 激活函数名称（relu/gelu）
    Returns:
        激活函数（torch.nn.functional中的函数）
    """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


class New_Audio_Guided_Attention(nn.Module):
    r"""音频引导注意力模块（论文改进版，更高效）
    功能：结合音频特征引导视频特征的通道注意力和空间注意力，同时融入视频自注意力
    """

    def __init__(self, beta):
        super(New_Audio_Guided_Attention, self).__init__()

        self.beta = beta  # 自注意力特征的权重系数
        self.relu = nn.ReLU()  # ReLU激活函数
        self.video_input_dim = 512  # 视频输入特征维度（对应CNN输出的通道数）
        self.audio_input_dim = 128  # 音频输入特征维度

        self.hidden_dim = 256  # 注意力模块隐藏层维度

        # 1. 通道注意力（Channel Attention）相关层
        self.affine_video_1 = nn.Linear(self.video_input_dim, self.video_input_dim)  # 视频特征线性变换
        self.affine_audio_1 = nn.Linear(self.audio_input_dim, self.video_input_dim)  # 音频特征线性变换（对齐视频维度）
        self.affine_bottleneck = nn.Linear(self.video_input_dim, self.hidden_dim)  # 瓶颈层（降维）
        self.affine_v_c_att = nn.Linear(self.hidden_dim, self.video_input_dim)  # 通道注意力权重输出层

        # 2. 空间注意力（Spatial Attention）相关层
        self.affine_video_2 = nn.Linear(self.video_input_dim, self.hidden_dim)  # 视频特征线性变换
        self.affine_audio_2 = nn.Linear(self.audio_input_dim, self.hidden_dim)  # 音频特征线性变换（对齐隐藏层维度）
        self.affine_v_s_att = nn.Linear(self.hidden_dim, 1)  # 空间注意力权重输出层（单通道）

        # 3. 视频自注意力（Self Attention）相关层
        self.latent_dim = 4  # 自注意力 latent 维度系数（用于降维）
        self.video_query = nn.Linear(self.video_input_dim, self.video_input_dim // self.latent_dim)  # Query投影
        self.video_key = nn.Linear(self.video_input_dim, self.video_input_dim // self.latent_dim)  # Key投影
        self.video_value = nn.Linear(self.video_input_dim, self.video_input_dim)  # Value投影

        # 4. 视频自空间注意力（辅助）相关层
        self.affine_video_ave = nn.Linear(self.video_input_dim, self.hidden_dim)  # 视频平均特征变换
        self.affine_video_3 = nn.Linear(self.video_input_dim, self.hidden_dim)  # 视频特征变换
        self.ave_bottleneck = nn.Linear(512, 256)  # 瓶颈层（未使用，预留）
        self.ave_v_att = nn.Linear(self.hidden_dim, 1)  # 自空间注意力权重输出层

        # 其他组件
        self.tanh = nn.Tanh()  # Tanh激活函数
        self.softmax = nn.Softmax(dim=-1)  # Softmax（计算注意力权重）
        self.dropout = nn.Dropout(0.2)  # Dropout（防止过拟合）
        self.norm = nn.LayerNorm(self.video_input_dim)  # 层归一化（稳定训练）

    def forward(self, video, audio):
        r"""前向传播：融合视频自注意力、音频引导的通道注意力和空间注意力

        Args:
            video: 视频特征 [batch, 10, 7, 7, 512]（batch=批次，10=时间步，7x7=空间尺寸，512=通道数）
            audio: 音频特征 [batch, 10, 128]（batch=批次，10=时间步，128=特征维度）
        Returns:
            c_s_att_visual_feat: 融合后视频特征 [batch, 10, 512]（时间步级别的全局特征）
        """
        # 音频特征维度调整（暂未用到，仅变换维度）
        audio = audio.transpose(1, 0)
        # 解析视频和音频特征维度
        batch, t_size, h, w, v_dim = video.size()  # batch=批次，t_size=时间步，h/w=空间高/宽，v_dim=视频通道数
        a_dim = audio.size(-1)  # 音频特征维度
        # 维度reshape：适配后续注意力计算
        audio_feature = audio.reshape(batch * t_size, a_dim)  # [batch*t_size, 128]（展平时间步）
        visual_feature = video.reshape(batch, t_size, -1, v_dim)  # [batch, t_size, 49, 512]（7x7=49，展平空间维度）
        raw_visual_feature = visual_feature  # 保存原始视频特征（用于残差连接）

        # ============================== 1. 视频自注意力（Video Self Attention） ==============================
        # 投影生成Query、Key、Value
        video_query_feature = self.video_query(visual_feature).reshape(batch * t_size, h * w,
                                                                       -1)  # [batch*t_size, 49, 128]（512/4=128）
        video_key_feature = self.video_key(visual_feature).reshape(batch * t_size, h * w, -1).permute(0, 2,
                                                                                                      1)  # [batch*t_size, 128, 49]（转置为Key维度）
        # 计算注意力能量（Query·Key^T）
        energy = torch.bmm(video_query_feature, video_key_feature)  # [batch*t_size, 49, 49]（空间位置间的相关性）
        attention = self.softmax(energy)  # 注意力权重 [batch*t_size, 49, 49]
        # 计算自注意力输出（注意力权重·Value）
        video_value_feature = self.video_value(visual_feature).reshape(batch * t_size, h * w,
                                                                       -1)  # [batch*t_size, 49, 512]
        output = torch.matmul(attention, video_value_feature)  # [batch*t_size, 49, 512]
        # 残差连接+dropout+归一化
        output = self.norm(visual_feature.reshape(batch * t_size, h * w, -1) + self.dropout(output))
        visual_feature = output  # 更新视频特征为自注意力输出

        # ============================== 2. 视频自空间注意力（Video Self Spatial Attention） ==============================
        # 计算视频特征空间平均（全局池化）
        video_average = visual_feature.sum(dim=1) / (h * w)  # [batch*t_size, 512]（每个时间步的空间平均特征）
        video_average = video_average.reshape(batch * t_size, v_dim)
        video_average = self.relu(self.affine_video_ave(video_average)).unsqueeze(-2)  # [batch*t_size, 1, 256]（激活+增维）
        # 视频特征投影为查询向量
        self_video_att_feat = visual_feature.reshape(batch * t_size, -1, v_dim)  # [batch*t_size, 49, 512]
        self_video_att_query = self.relu(self.affine_video_3(self_video_att_feat))  # [batch*t_size, 4