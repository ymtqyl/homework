"""
main_model.py

本文件定义了用于音视频事件定位的两个核心模型架构：
1.  weak_main_model: 用于弱监督学习（仅有视频级别的标签）。
2.  supv_main_model: 用于全监督学习（有详细的时间戳标签）。

模型融合了 RNN、Transformer（Encoder/Decoder）和多种注意力机制
来实现音视频的跨模态信息融合与事件定位。
"""

import torch
from torch import nn
import torch.nn.functional as F
# 导入本地 .models 模块中的自定义注意力机制
from .models import New_Audio_Guided_Attention
# 导入本地 .models 模块中的 Transformer 组件
from .models import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention
# 导入自定义的 Dual_lstm 模块
from .Dual_lstm import Dual_lstm


class RNNEncoder(nn.Module):
    """
    一个封装了音频和视频 LSTMs 的 RNN 编码器。
    用于对音频和视频特征序列进行初步的时序编码。
    """
    def __init__(self, audio_dim, video_dim, d_model, num_layers):
        super(RNNEncoder, self).__init__()

        self.d_model = d_model
        # 音频 LSTM：双向，输出维度为 d_model (d_model/2 * 2)
        self.audio_rnn = nn.LSTM(audio_dim, int(d_model / 2), num_layers=num_layers, batch_first=True,
                                 bidirectional=True, dropout=0.2)
        # 视频 LSTM：双向，输出维度为 d_model * 2
        # 注意：这里的输出维度是 d_model * 2，而音频是 d_model
        self.visual_rnn = nn.LSTM(video_dim, d_model, num_layers=num_layers, batch_first=True, bidirectional=True,
                                  dropout=0.2)

    def forward(self, audio_feature, visual_feature):
        """
        前向传播。
        Inputs:
            - audio_feature: (batch, seq_len, audio_dim)
            - visual_feature: (batch, seq_len, video_dim)
        Outputs:
            - audio_output: (batch, seq_len, d_model)
            - video_output: (batch, seq_len, d_model * 2)
        """
        audio_output, _ = self.audio_rnn(audio_feature)
        video_output, _ = self.visual_rnn(visual_feature)
        return audio_output, video_output


class InternalTemporalRelationModule(nn.Module):
    """
    模态内部时序关系模块（基于 Transformer Encoder）。
    用于捕捉单个模态（如视频或音频）内部的时间依赖关系。
    """
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(InternalTemporalRelationModule, self).__init__()
        # 定义一个 Transformer 编码器层
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        # 堆叠多个编码器层（这里 num_layers=2）
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        # 线性层，用于将输入特征投影到 d_model 维度
        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here? (代码中的原始注释)

    def forward(self, feature):
        """
        Inputs:
            - feature: (seq_len, batch, input_dim)
        Outputs:
            - feature: (seq_len, batch, d_model)
        """
        # 1. 线性投影到 d_model 维度
        feature = self.affine_matrix(feature)
        # 2. 通过 Transformer Encoder 捕捉时序关系
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    """
    跨模态关系注意力模块（基于 Transformer Decoder）。
    用于融合两种模态的信息。
    它使用一种模态作为 Query，另一种模态作为 Key 和 Value (Memory)。
    """
    def __init__(self, input_dim, d_model, feedforward_dim):
        super(CrossModalRelationAttModule, self).__init__()

        # 定义一个 Transformer 解码器层
        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4, dim_feedforward=feedforward_dim)
        # 堆叠解码器层（这里 num_layers=1）
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        # 线性层，用于将 Query 特征投影到 d_model 维度
        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        """
        Inputs:
            - query_feature: (seq_len_q, batch, input_dim) - 作为 Query
            - memory_feature: (seq_len_kv, batch, d_model) - 作为 Key 和 Value
        Outputs:
            - output: (seq_len_q, batch, d_model)
        """
        # 1. 线性投影 Query 特征
        query_feature = self.affine_matrix(query_feature)
        # 2. 通过 Transformer Decoder 进行跨模态注意力
        output = self.decoder(query_feature, memory_feature)

        return output


class CAS_Module(nn.Module):
    """
    类别激活序列（Class Activation Sequence）模块。
    使用一个 1x1 卷积（等效于时序上的全连接层）来预测每个时间步的类别分数。
    """
    def __init__(self, d_model, num_class=28):
        super(CAS_Module, self).__init__()
        self.d_model = d_model
        self.num_class = num_class
        self.dropout = nn.Dropout(0.2)

        # 使用 1D 卷积（kernel_size=1）作为分类器
        # In: (batch, d_model, seq_len) -> Out: (batch, num_class+1, seq_len)
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=self.num_class+1, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, content):
        """
        Inputs:
            - content: (batch, seq_len, d_model)
        Outputs:
            - out: (batch, seq_len, num_class+1)
        """
        # 交换维度以适应 Conv1d: (batch, seq_len, d_model) -> (batch, d_model, seq_len)
        content = content.permute(0, 2, 1)

        # 通过 1x1 卷积进行分类
        out = self.classifier(content)
        
        # 换回原始维度: (batch, num_class+1, seq_len) -> (batch, seq_len, num_class+1)
        out = out.permute(0, 2, 1)
        return out


class SupvLocalizeModule(nn.Module):
    """
    全监督定位模块。
    用于在全监督设置下预测事件的边界（开始/结束）和类别。
    """
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        # 线性分类器，预测每个时间步是否为事件（1维输出）
        self.classifier = nn.Linear(d_model, 1)  # start and end
        # 线性分类器，预测整个序列的事件类别（28维输出）
        self.event_classifier = nn.Linear(d_model, 28)

    def forward(self, fused_content):
        """
        Inputs:
            - fused_content: (seq_len, batch, d_model)
        Outputs:
            - logits: (seq_len, batch, 1) - 每个时间步的事件激活
            - class_scores: (batch, 28) - 整个序列的类别分数
        """
        # --- 序列级类别预测 ---
        # 1. 转置: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        # 2. 在时间维度上取最大值 (Max Pooling)，得到 (batch, d_model)
        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        
        # 3. 预测整个序列的类别
        class_logits = self.event_classifier(max_fused_content)
        class_scores = class_logits # (batch, 28)

        # --- 帧级事件预测 ---
        # 预测每个时间步是否属于一个事件
        logits = self.classifier(fused_content) # (seq_len, batch, 1)

        return logits, class_scores


class WeaklyLocalizationModule(nn.Module):

        # 弱监督定位模块（多示例学习 MIL）。

    def __init__(self, input_dim):
        super(WeaklyLocalizationModule, self).__init__()

        self.hidden_dim = input_dim
        # 帧级分类器（是否为事件）
        self.classifier = nn.Linear(self.hidden_dim, 1)
        # 序列级分类器（事件类别）
        self.event_classifier = nn.Linear(self.hidden_dim, 29)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content):
        # (seq_len, batch, dim) -> (batch, seq_len, dim)
        fused_content = fused_content.transpose(0, 1)
        # 序列级特征（Max Pooling）
        max_fused_content, _ = fused_content.max(1)
        
        # 1. 帧级置信度分数 (batch, seq_len, 1)
        is_event_scores = self.classifier(fused_content)
        
        # 2. 序列级原始类别 logits (batch, 29) -> (batch, 1, 29)
        raw_logits = self.event_classifier(max_fused_content)[:, None, :]
        
        # 3. 融合：P(class | frame_t) = P(is_event | frame_t) * P(class | video)
        fused_logits = is_event_scores.sigmoid() * raw_logits # (batch, seq_len, 29)
        
        # 4. MIL：在时间维度上取最大值，得到视频级别的类别预测
        logits, _ = torch.max(fused_logits, dim=1)
        event_scores = self.softmax(logits)

        return is_event_scores.squeeze(), raw_logits.squeeze(), event_scores

class AudioVideoInter(nn.Module):
    """
    音视频交互模块（基于 MultiheadAttention）。
    使用一个融合特征（逐点相乘）作为 Query，
    并使用原始的音频和视频特征（拼接后）作为 Key 和 Value。
    """
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        # 多头注意力层
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, video_feat, audio_feat):
        """
        Inputs:
            - video_feat: (seq_len, batch, d_model)
            - audio_feat: (seq_len, batch, d_model)
        Outputs:
            - output: (seq_len, batch, d_model)
        """
        # 1. 逐点相乘，生成融合特征作为 Query
        global_feat = video_feat * audio_feat
        
        # 2. 拼接音频和视频特征，作为 Memory (Key 和 Value)
        # memory shape: (seq_len * 2, batch, d_model)
        memory = torch.cat([audio_feat, video_feat], dim=0)
        
        # 3. 多头注意力：Query=global_feat, Key=memory, Value=memory
        mid_out = self.video_multihead(global_feat, memory, memory)[0]
        
        # 4. 残差连接和层归一化
        output = self.norm1(global_feat + self.dropout(mid_out))

        return output


class weak_main_model(nn.Module):
    """
    弱监督学习的主模型。
    """
    def __init__(self, config):
        super(weak_main_model, self).__init__()
        # --- 1. 加载配置参数 ---
        self.config = config
        self.beta = self.config["beta"]
        self.alpha = self.config["alpha"]
        self.gamma = self.config["gamma"]
        
        # --- 2. 模块初始化 ---
        # 2.1 音频引导的空间-通道注意力
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        
        self.video_input_dim = self.config["video_inputdim"]
        self.video_fc_dim = self.config["video_inputdim"]
        self.d_model = self.config["d_model"]
        self.audio_input_dim = self.config["audio_inputdim"]
        
        # 2.2 视频特征的初始全连接层 (用于维度适配)
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # 2.3 模态内部时序编码器 (Transformer Encoder)
        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=2048)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=2048)
        
        # 2.4 跨模态注意力解码器 (Transformer Decoder)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_fc_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.audio_input_dim, d_model=self.d_model, feedforward_dim=1024)
        
        # 2.5 音视频交互模块
        self.AVInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=2, head_dropout=0.2)
        
        # 2.6 门控单元 (用于生成帧级事件激活)
        self.audio_gated = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())
        self.video_gated = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())
        
        # 2.7 最终的 CAS 模块和分类器
        self.CAS_model = CAS_Module(d_model=self.d_model, num_class=28)
        self.classifier = nn.Linear(self.d_model, 1) # 帧级分类器 (未使用)
        self.softmax = nn.Softmax(dim=-1)
        
        # 2.8 辅助的模态特定分类器 (用于生成辅助门控)
        self.audio_cas = nn.Linear(self.d_model, 29)
        self.video_cas = nn.Linear(self.d_model, 29)

    def forward(self, visual_feature, audio_feature):
        """
        弱监督模型前向传播
        Inputs:
            - visual_feature: (batch, seq_len, video_input_dim)
            - audio_feature: (batch, seq_len, audio_input_dim)
        """
        
        # --- 1. 特征预处理 ---
        # (batch, seq_len, dim) -> (seq_len, batch, dim) 以适应 Transformer
        audio_feature = audio_feature.transpose(1, 0).contiguous()
        
        # 视频特征通过 FC 层
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # --- 2. 音频引导的空间-通道注意力 ---
        # visual_feature: (batch, seq_len, dim)
        # audio_feature: (seq_len, batch, dim) -> (batch, seq_len, dim)
        # 注意：这里 audio_feature 需要转置回来以匹配注意力模块的输入
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature.transpose(1, 0).contiguous())
        
        # (batch, seq_len, dim) -> (seq_len, batch, dim)
        visual_feature = visual_feature.transpose(1, 0).contiguous()

        # --- 3. 跨模态 Transformer 编码 ---
        
        # --- (3a) 音频作为 Query，视频作为 Memory ---
        # 视频特征通过 Encoder 编码，作为 Key 和 Value
        video_key_value_feature = self.video_encoder(visual_feature)
        # 音频特征作为 Query，查询视频的 Key/Value
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # --- (3b) 视频作为 Query，音频作为 Memory ---
        # 音频特征通过 Encoder 编码，作为 Key 和 Value
        audio_key_value_feature = self.audio_encoder(audio_feature)
        # 视频特征作为 Query，查询音频的 Key/Value
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        # --- 4. 门控机制 (用于帧级激活) ---
        # 基于音频编码特征生成门控信号
        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature) # (注意：这里也用了 audio_key_value_feature)

        # 平均门控信号
        av_gate = (audio_gate + video_gate) / 2
        av_gate = av_gate.permute(1, 0, 2) # (seq_len, batch, 1) -> (batch, seq_len, 1)

        # --- 5. 门控融合 (特征增强) ---
        # 使用门控信号和 alpha 参数来调整（增强）查询输出
        video_query_output = (1 - self.alpha)*video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = (1 - self.alpha)*audio_query_output + video_gate * audio_query_output * self.alpha

        # --- 6. 辅助分类器（用于生成辅助门控） ---
        video_cas = self.video_cas(video_query_output).permute(1, 0, 2) # (batch, seq_len, 29)
        audio_cas = self.audio_cas(audio_query_output).permute(1, 0, 2) # (batch, seq_len, 29)
        
        # 得到模态特定的类别门控
        video_cas_gate = video_cas.sigmoid()
        audio_cas_gate = audio_cas.sigmoid()
        

        # --- 7. 最终跨模态交互 ---
        # 将门控融合后的特征送入交互模块
        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        
        # --- 8. 弱监督定位 (MIL) 逻辑 ---
        
        # 融合两个交互模块的输出
        fused_content = (video_query_output + audio_query_output) / 2
        # (seq_len, batch, dim) -> (batch, seq_len, dim)
        fused_content = fused_content.transpose(0, 1)
        
        # (a) 帧级分类 (CAS)
        cas_score = self.CAS_model(fused_content) # (batch, seq_len, 29)
        
        # (b) 使用辅助分类器的门控来加权 CAS 分数
        cas_score = self.gamma*video_cas_gate*cas_score + self.gamma*audio_cas_gate*cas_score
        
        # (c) Top-K Pooling: 在时间维度上选取 K=4 个最高分
        sorted_scores, _ = cas_score.sort(descending=True, dim=1)
        topk_scores = sorted_scores[:, :4, :] # (batch, 4, 29)
        
        # (d) 得到序列级的原始 logits (平均 Top-K)
        raw_logits = torch.mean(topk_scores, dim=1)[:, None, :] # (batch, 1, 29)
        
        # (e) 融合：P(class | frame_t) = P(is_event | frame_t) * P(class | video)
        #     这里 P(is_event | frame_t) 用 av_gate (模态门控) 代替
        fused_logits = av_gate * raw_logits # (batch, seq_len, 29)
        
        # (f) MIL：在时间维度上取最大值，得到视频级别的类别预测
        logits, _ = torch.max(fused_logits, dim=1) # (batch, 29)
        
        # (g) Softmax 得到最终概率
        event_scores = self.softmax(logits)

        # 返回：帧级事件门控, 序列级原始logits, 序列级最终概率
        return av_gate.squeeze(), raw_logits.squeeze(), event_scores


class supv_main_model(nn.Module):
    """
    全监督学习的主模型。
    """
    def __init__(self, config):
        super(supv_main_model, self).__init__()
        # --- 1. 加载配置参数 ---
        self.config = config
        self.beta = self.config["beta"]
        
        # --- 2. 模块初始化 ---
        # 2.1 音频引导的空间-通道注意力
        self.spatial_channel_att = New_Audio_Guided_Attention(self.beta).cuda()
        self.video_input_dim = self.config['video_inputdim']
        self.audio_input_dim = self.config['audio_inputdim']

        self.video_fc_dim = 512
        self.d_model = self.config['d_model']

        # 2.2 视频特征 FC 层
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        # 2.3 模态内部编码器 (Transformer Encoder)
        self.video_encoder = InternalTemporalRelationModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        
        # 2.4 跨模态解码器 (Transformer Decoder)
        self.video_decoder = CrossModalRelationAttModule(input_dim=self.video_input_dim, d_model=self.d_model, feedforward_dim=1024)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=self.d_model, d_model=self.d_model, feedforward_dim=1024)
        
        # 2.5 初始的 RNN 编码层 (与弱监督模型不同)
        self.audio_visual_rnn_layer = RNNEncoder(audio_dim=self.audio_input_dim, video_dim=self.video_input_dim, d_model=self.d_model, num_layers=1)

        # 2.6 门控单元
        self.audio_gated = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())
        self.video_gated = nn.Sequential(nn.Linear(self.d_model, 1), nn.Sigmoid())
        
        # 2.7 音视频交互模块
        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        self.VAInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.2)
        
        # 2.8 全监督定位模块
        self.localize_module = SupvLocalizeModule(self.d_model)
        
        # 2.9 辅助的模态特定分类器
        self.video_norm = nn.LayerNorm(self.d_model) # (LayerNorm 未在 forward 中使用)
        self.audio_norm = nn.LayerNorm(self.d_model) # (LayerNorm 未在 forward 中使用)
        self.audio_cas = nn.Linear(self.d_model, 28)
        self.video_cas = nn.Linear(self.d_model, 28)
        
        # 2.10 超参数
        self.alpha = self.config['alpha']
        self.gamma = self.config['gamma']


    def forward(self, visual_feature, audio_feature):
        """
        全监督模型前向传播
        Inputs:
            - visual_feature: (batch, seq_len, video_input_dim)
            - audio_feature: (batch, seq_len, audio_input_dim)
        """
        
        # --- 1. 特征预处理 ---
        audio_rnn_input = audio_feature # 原始特征
        # (batch, seq_len, dim) -> (seq_len, batch, dim)
        audio_feature_trans = audio_feature.transpose(1, 0).contiguous()
        
        visual_feature = self.v_fc(visual_feature)
        visual_feature = self.dropout(self.relu(visual_feature))

        # --- 2. 音频引导的空间-通道注意力 ---
        # visual_feature: (batch, seq_len, dim)
        # audio_feature_trans: (seq_len, batch, dim) -> (batch, seq_len, dim)
        visual_feature = self.spatial_channel_att(visual_feature, audio_feature_trans.transpose(1,0).contiguous())
        visual_rnn_input = visual_feature # (batch, seq_len, dim)

        # --- 3. 初始 RNN 编码 ---
        # (与弱监督模型不同，这里首先使用 RNN 编码)
        audio_rnn_output1, visual_rnn_output1 = self.audio_visual_rnn_layer(audio_rnn_input, visual_rnn_input)
        
        # (batch, seq_len, dim) -> (seq_len, batch, dim)
        audio_encoder_input1 = audio_rnn_output1.transpose(1, 0).contiguous()
        # 注意：visual_rnn_output1 的维度是 d_model*2，而 video_encoder 的 input_dim 是 video_input_dim
        visual_encoder_input1 = visual_rnn_output1.transpose(1, 0).contiguous()

        # --- 4. 跨模态 Transformer 编码 ---
        
        # --- (4a) 音频作为 Query，视频作为 Memory ---
        video_key_value_feature = self.video_encoder(visual_encoder_input1)
        audio_query_output = self.audio_decoder(audio_encoder_input1, video_key_value_feature)

        # --- (4b) 视频作为 Query，音频作为 Memory ---
        audio_key_value_feature = self.audio_encoder(audio_encoder_input1)
        video_query_output = self.video_decoder(visual_encoder_input1, audio_key_value_feature)

        # --- 5. 门控机制 ---
        audio_gate = self.audio_gated(audio_key_value_feature)
        video_gate = self.video_gated(video_key_value_feature)
        # 逐点相乘
        audio_visual_gate = audio_gate * video_gate

        # --- 6. 门控融合 (特征增强) ---
        video_query_output = video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = audio_query_output + video_gate * audio_query_output * self.alpha

        # --- 7. 辅助分类器（用于生成辅助分数） ---
        video_cas = self.video_cas(video_query_output).permute(1, 0, 2) # (batch, seq_len, 28)
        audio_cas = self.audio_cas(audio_query_output).permute(1, 0, 2)
        
        # Top-K Pooling
        sorted_scores_video, _ = video_cas.sort(descending=True, dim=1)
        topk_scores_video = sorted_scores_video[:, :4, :]
        score_video = torch.mean(topk_scores_video, dim=1) # (batch, 28)
        
        sorted_scores_audio, _ = audio_cas.sort(descending=True, dim=1)
        topk_scores_audio = sorted_scores_audio[:, :4, :]
        score_audio = torch.mean(topk_scores_audio, dim=1) # (batch, 28)

        # 辅助分数 (平均)
        av_score = (score_video + score_audio) / 2 # (batch, 28)

        # --- 8. 最终跨模态交互 ---
        video_query_output = self.AVInter(video_query_output, audio_query_output)
        audio_query_output = self.VAInter(audio_query_output, video_query_output)
        
        # --- 9. 全监督定位 ---
        # 融合两个交互模块的输出
        fused_output = (video_query_output + audio_query_output) / 2
        
        # 通过定位模块得到帧级和序列级的分数
        is_event_scores, event_scores = self.localize_module(fused_output)
        
        # --- 10. 最终分数融合 ---
        # 使用辅助分数 (av_score) 和 gamma 参数来调整（增强）最终的序列级分数
        event_scores = event_scores + self.gamma*av_score
        
        # 返回：帧级事件分数, 序列级类别分数, 模态门控, 辅助分数
        return is_event_scores, event_scores, audio_visual_gate, av_score