"""
Dual_lstm.py

定义了一个自定义的 LSTM 变体，称为 "Dual LSTM"。
- Dual_lstm_cell: 定义了 LSTM 单元的计算逻辑，
  它在一次 forward 调用中同时更新音频和视频的 (hidden, cell) 状态。
- Dual_lstm: 包装了 Dual_lstm_cell，使其能像 nn.LSTM 一样处理整个序列。
"""

import torch
import copy
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
import math
from torch.autograd import Variable


class Dual_lstm_cell(nn.Module):
    """
    双通道 LSTM 单元 (LSTM Cell)。
    
    这个单元在一次 `forward` pass 中计算视觉和音频两个流的 LSTM 更新。
    注意：在当前实现中，`visual_gates` 仅依赖于视觉输入，
    `audio_gates` 仅依赖于音频输入（跨模态的项被注释掉了）。
    因此，它在功能上等价于两个独立的 LSTMCell。
    """
    def __init__(self, visual_input_dim, audio_input_dim, hidden_dim, alph=0.5, bias=True):
        """
        初始化 Dual_lstm_cell。
        
        Args:
            visual_input_dim (int): 视觉输入特征的维度。
            audio_input_dim (int): 音频输入特征的维度。
            hidden_dim (int): 隐藏状态和单元状态的维度。
            alph (float, optional): 用于跨模态融合的 alpha 权重。
            bias (bool, optional): 线性层是否使用偏置。
        """
        super(Dual_lstm_cell, self).__init__()

        self.visual_input_dim = visual_input_dim
        self.audio_input_dim  = audio_input_dim
        self.hidden_dim = hidden_dim
        self.alph = alph 
        
        # --- 视觉流的线性层 ---
        # 4 * hidden_dim 对应于输入门(i)、遗忘门(f)、单元门(c)、输出门(o)
        self.vs_linear = nn.Linear(self.visual_input_dim, 4 * self.hidden_dim, bias=bias) # (visual_state)
        self.vh_linear = nn.Linear(self.hidden_dim, 4* self.hidden_dim, bias=bias)       # (visual_hidden)
        
        # --- 音频流的线性层 (第一组) ---
        self.as_linear = nn.Linear(self.audio_input_dim, 4 * self.hidden_dim, bias=bias) # (audio_state)
        self.ah_linear = nn.Linear(self.hidden_dim, 4 * self.hidden_dim, bias=bias)       # (audio_hidden)

        # --- 音频流和跨模态的线性层 (第二组) ---
        # (在 forward 中，音频流实际使用的是这组)
        self.as_linear2 = nn.Linear(self.audio_input_dim, 4*self.hidden_dim, bias=bias)
        self.ah_linear2 = nn.Linear(self.hidden_dim, 4*self.hidden_dim, bias=bias)
        self.vs_linear2 = nn.Linear(self.visual_input_dim, 4*self.hidden_dim, bias=bias)
        self.vh_linear2 = nn.Linear(self.hidden_dim, 4*self.hidden_dim, bias=bias)
        
        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数（均匀分布）"""
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, visual_state, visual_hidden, visual_cell, audio_state, audio_hidden, audio_cell):
        """
        执行单个时间步的 LSTM 计算。
        
        Inputs: (t 时刻的输入)
            - visual_state, audio_state: (batch, input_dim) (当前时间步的输入 x_t)
            - visual_hidden, audio_hidden: (batch, hidden_dim) (t-1 时刻的隐藏状态 h_t-1)
            - visual_cell, audio_cell: (batch, hidden_dim) (t-1 时刻的单元状态 c_t-1)
        Outputs: (t 时刻的状态)
            - visual_output, audio_output: (batch, hidden_dim) (t 时刻的隐藏状态 h_t)
            - visual_cell, audio_cell: (batch, hidden_dim) (t 时刻的单元状态 c_t)
        """
        
        # --- 1. 计算视觉流的四个门（i, f, c, o） ---
        # W_v * x_v_t + U_v * h_v_t-1
        visual_gates = self.vs_linear(visual_state) + self.vh_linear(visual_hidden)
            # (注释掉的跨模态项)
            #self.alph*self.as_linear(audio_state) + self.alph*self.ah_linear(audio_hidden)

        # --- 2. 计算音频流的四个门（i, f, c, o） ---
        # (注意：这里使用了 as_linear2 和 ah_linear2)
        # W_a * x_a_t + U_a * h_a_t-1
        audio_gates = self.as_linear2(audio_state) + self.ah_linear2(audio_hidden)
            # (注释掉的跨模态项)
            #self.alph*self.vs_linear2(visual_state) + self.alph*self.vh_linear2(visual_hidden)

        # --- 3. 将四个门的计算结果分开 ---
        # (batch, 4*hidden_dim) -> 4 * (batch, hidden_dim)
        visual_i_gate, visual_f_gate, visual_c_gate, visual_o_gate = visual_gates.chunk(4,1)
        audio_i_gate, audio_f_gate, audio_c_gate, audio_o_gate = audio_gates.chunk(4,1)

        # --- 4. 视觉流 LSTM 计算 ---
        visual_i_gate = F.sigmoid(visual_i_gate) # 输入门 (i_t)
        visual_f_gate = F.sigmoid(visual_f_gate) # 遗忘门 (f_t)
        visual_c_gate = F.tanh(visual_c_gate)    # 候选单元 (c_tilde_t)
        visual_o_gate = F.sigmoid(visual_o_gate) # 输出门 (o_t)

        # 单元状态更新 (c_t = f_t * c_t-1 + i_t * c_tilde_t)
        visual_cell = visual_f_gate * visual_cell + visual_i_gate * visual_c_gate
        # 隐藏状态（输出） (h_t = o_t * tanh(c_t))
        visual_output = visual_o_gate * torch.tanh(visual_cell)

        # --- 5. 音频流 LSTM 计算 ---
        audio_i_gate = F.sigmoid(audio_i_gate) # 输入门
        audio_f_gate = F.sigmoid(audio_f_gate) # 遗忘门
        audio_c_gate = F.tanh(audio_c_gate)    # 候选单元
        audio_o_gate = F.sigmoid(audio_o_gate) # 输出门

        # 单元状态更新
        audio_cell = audio_f_gate * audio_cell + audio_i_gate * audio_c_gate
        # 隐藏状态（输出）
        audio_output = audio_o_gate * torch.tanh(audio_cell)

        return visual_output, visual_cell, audio_output, audio_cell

class Dual_lstm(nn.Module):
    """
    双通道 LSTM 序列模型。
    
    本模块在时间维度上“展开” `Dual_lstm_cell`，
    以处理完整的输入序列。
    它实现了（单向）LSTM 的功能，因为反向部分被注释掉了。
    """
    def __init__(self):
        """
        初始化 Dual_lstm 序列模型。
        """
        super(Dual_lstm, self).__init__()

        # --- 定义固定的维度（硬编码） ---
        self.video_input_dim = 512
        self.video_fc_dim = 512
        self.d_model = 256 # (d_model 在这里用作 hidden_dim)
        
        # (v_fc 在这个类中未被使用)
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        
        # --- 实例化自定义的 LSTM 单元 ---
        # (正向)
        self.LSTM_cell = Dual_lstm_cell(visual_input_dim=512, audio_input_dim=128, hidden_dim=256)
        # (反向 LSTM 单元被注释掉了)
        #self.LSTM_cell_r = Dual_lstm_cell(visual_input_dim=512, audio_input_dim=128, hidden_dim=256)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)


    def forward(self, audio_feature, visual_feature):
        """
        执行完整的序列前向传播。
        
        Inputs:
            - audio_feature: (batch, seq_len, 128)
            - visual_feature: (batch, seq_len, 512)
        Outputs:
            - audio_output: (batch, seq_len, 256) (正向传播的隐藏状态序列)
            - visual_output: (batch, seq_len, 256) (正向传播的隐藏状态序列)
        """
        audio_rnn_input = audio_feature
        visual_rnn_input = visual_feature

        # --- 1. 初始化隐藏状态 (h_0) 和单元状态 (c_0) ---
        # (包括了反向传播的初始化，虽然反向传播被注释掉了)
        if torch.cuda.is_available():
            visual_hidden = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
            visual_hidden_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda()) # (反向)
        else:
            visual_hidden = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))
            visual_hidden_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            visual_cell = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda())
            visual_cell_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model).cuda()) # (反向)
        else:
            visual_cell = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))
            visual_cell_r = Variable(torch.zeros(visual_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            audio_hidden = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
            audio_hidden_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda()) # (反向)
        else:
            audio_hidden = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))
            audio_hidden_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))

        if torch.cuda.is_available():
            audio_cell = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda())
            audio_cell_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model).cuda()) # (反向)
        else:
            audio_cell = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))
            audio_cell_r = Variable(torch.zeros(audio_rnn_input.size(0), self.d_model))

        # --- 2. 准备存储输出序列 ---
        visual_output = []
        audio_output = []
        visual_output_r = [] # (反向)
        audio_output_r = [] # (反向)
        
        # 获取序列长度 (seq_len)
        length = visual_rnn_input.size(1) 

        # --- 3. 状态类型转换 ---
        # (确保状态张量的数据类型为 double)
        visual_hidden = visual_hidden.double()
        visual_cell = visual_cell.double()
        audio_hidden = audio_hidden.double()
        audio_cell = audio_cell.double()
        visual_hidden_r = visual_hidden_r.double()
        visual_cell_r = visual_cell_r.double()
        audio_hidden_r = audio_hidden_r.double()
        audio_cell_r = audio_cell_r.double()


        # --- 4. 正向 RNN 展开 (循环) ---
        # 循环遍历序列的每个时间步
        for i in range(length):
            # 调用自定义 cell，传入 t=i 时刻的输入 和 t-1 时刻的状态
            visual_hidden, visual_cell, audio_hidden, audio_cell = self.LSTM_cell(
                visual_rnn_input[:,i,:], visual_hidden, visual_cell,
                audio_rnn_input[:,i,:], audio_hidden, audio_cell
            )
            # 存储 t=i 时刻的输出 (即隐藏状态 h_t)
            visual_output.append(visual_hidden)
            audio_output.append(audio_hidden)

        # --- 5. 堆叠输出 ---
        # 将 list 转换为 (batch, seq_len, hidden_dim) 的张量
        visual_output = torch.stack(visual_output,dim=1)
        audio_output = torch.stack(audio_output, dim=1)


        # --- 6. 反向 RNN (已注释掉) ---
        # (这里的代码块用于实现双向 LSTM)
        # for i in range(length):
        #     visual_hidden_r, visual_cell_r, audio_hidden_r, audio_cell_r = self.LSTM_cell_r(visual_rnn_input[:,length-1-i,:], visual_hidden_r,
        #                                                                                     visual_cell_r, audio_rnn_input[:,length-1-i,:],
        #                                                                                     audio_hidden_r, audio_cell_r)
        #     visual_output_r.append(visual_hidden_r)
        #     audio_output_r.append(audio_hidden_r)
        #
        # visual_output_r = torch.stack(visual_output_r, dim=1)
        # visual_output_r = torch.flip(visual_output_r, dims=[1]) # 反转时间维度
        # audio_output_r = torch.stack(audio_output_r, dim=1)
        # audio_output_r = torch.flip(audio_output_r, dims=[1])
        #
        # # 拼接正向和反向的输出
        # visual_output = torch.cat((visual_output, visual_output_r), dim=2)
        # audio_output = torch.cat((audio_output, audio_output_r), dim=2)
        
        # 返回正向传播的输出
        return audio_output, visual_output


# --- 7. (用于测试模型的示例代码) ---
# model = Dual_lstm()
# visual_feature = torch.randn(32, 10,512)
# audio_feature = torch.randn(32, 10, 128)
# model(audio_feature, visual_feature)
#