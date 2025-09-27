"""
M2Rec: Multi-scale Mamba for Efficient Sequential Recommendation

该模型基于Mamba架构，结合大语言模型(LLM)嵌入向量，实现高效的序列推荐。
主要特点：
1. 使用Mamba状态空间模型处理序列数据
2. 融合物品的LLM嵌入向量和传统嵌入
3. 支持多尺度特征融合
4. 高效处理长序列推荐任务

作者: M2Rec项目组
"""

import torch
from torch import nn
# from mamba_ssm import Mamba as Mambao
from mamba.mamba_ssm.modules.mamba_simple import Mamba
# from mamba.mamba_ssm.modules.mamba_simple_v2 import Mamba
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import numpy as np
import pickle
import pandas as pd


def load_data(file):
    """
    加载物品元数据嵌入向量

    Args:
        file (str): pickle文件路径

    Returns:
        dict: 物品ID到嵌入向量的映射字典
    """
    data_load_file = []
    file_1 = open(file, "rb")
    # file_1.seek(0)
    data_load_file = pickle.load(file_1)
    return data_load_file


class Mamba4Rec(SequentialRecommender):
    """
    M2Rec模型主类：多尺度Mamba序列推荐模型

    该类继承自RecBole的SequentialRecommender，实现了基于Mamba架构的序列推荐模型。
    核心创新：融合传统物品嵌入和LLM生成的物品元数据嵌入。

    模型架构：
    1. 物品嵌入层 (传统ID嵌入 + LLM元数据嵌入)
    2. 多层Mamba编码器
    3. 输出预测层

    Args:
        config (dict): 模型配置参数
        dataset: RecBole数据集对象
    """

    def __init__(self, config, dataset):
        """
        初始化M2Rec模型

        Args:
            config (dict): 包含所有超参数的配置字典
            dataset: RecBole格式的数据集对象
        """
        super(Mamba4Rec, self).__init__(config, dataset)

        # ======================== 基础模型参数 ========================
        self.hidden_size = config["hidden_size"]  # 隐藏层维度，默认64
        self.loss_type = config["loss_type"]  # 损失函数类型：'BPR'或'CE'
        self.num_layers = config["num_layers"]  # Mamba层数，默认1
        self.dropout_prob = config["dropout_prob"]  # Dropout概率，默认0.2

        # ======================== Mamba块超参数 ========================
        self.d_state = config["d_state"]  # SSM状态扩展因子，默认32
        self.d_conv = config["d_conv"]  # 局部卷积宽度，默认4
        self.expand = config["expand"]  # 块扩展因子，默认2
        self.data = config["dataset"]  # 数据集名称

        # ======================== LLM嵌入向量加载 ========================
        # 加载预训练的物品元数据嵌入向量 (item_id -> 1024维向量)
        self.llm_vec = load_data('./dataset/{}/item_meta_emb.pkl'.format(self.data))
        # 为padding物品(ID=0)设置零向量
        self.llm_vec[0] = np.array([0.0] * 1024)

        # 将字典转换为DataFrame，便于批量索引
        # shape: [n_items, 1024] - 每行对应一个物品的1024维LLM嵌入
        self.llm_matrix = pd.DataFrame([self.llm_vec[i] for i in range(len(self.llm_vec))])

        # ======================== 嵌入层定义 ========================
        # 传统物品ID嵌入层: [n_items] -> [hidden_size]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        # ======================== 归一化和正则化层 ========================
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)  # 层归一化
        self.dropout = nn.Dropout(self.dropout_prob)  # Dropout正则化
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]  # 最大序列长度

        # ======================== Mamba编码器层 ========================
        # 多层Mamba编码器，每层包含Mamba块+FFN
        self.mamba_layers = nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,  # 模型维度
                d_state=self.d_state,  # SSM状态维度
                d_conv=self.d_conv,  # 卷积核大小
                expand=self.expand,  # 扩展比例
                dropout=self.dropout_prob,  # Dropout概率
                num_layers=self.num_layers,  # 层数（用于残差连接判断）
            ) for _ in range(self.num_layers)
        ])

        # ======================== 损失函数 ========================
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()  # 贝叶斯个性化排序损失
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()  # 交叉熵损失
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # ======================== LLM嵌入处理网络 ========================
        # 用于处理1024维LLM嵌入向量的卷积网络
        # Input: [batch_size, seq_len, 1024] -> Output: [batch_size, seq_len, hidden_size]
        self.l1 = nn.Sequential(
            # 1D卷积：[200, 1024] -> [200, 341] (kernel=4, stride=3)
            nn.Conv1d(200, 200, 4, stride=3),  # 时序卷积
            nn.GELU(),  # GELU激活函数
            # 线性变换：341 -> 64 (hidden_size)
            nn.Linear(341, 64),  # 降维到hidden_size
            nn.GELU()  # GELU激活函数
        )

        # ======================== 融合权重参数 ========================
        # 用于控制传统嵌入和LLM嵌入的融合比例
        self.alpha = nn.Parameter(torch.FloatTensor(1) * 0.5, requires_grad=True)  # 传统嵌入权重
        self.beta = nn.Parameter(torch.FloatTensor(1) * 0.5, requires_grad=True)  # LLM嵌入权重

        # ======================== 权重初始化 ========================
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        权重初始化函数

        采用不同的初始化策略：
        - Linear和Embedding层：正态分布初始化 (mean=0.0, std=0.02)
        - LayerNorm层：bias设为0，weight设为1
        - Linear层bias：设为0

        Args:
            module: 要初始化的网络模块
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        """
        模型前向传播

        核心流程：
        1. 获取传统物品嵌入 [batch_size, seq_len, hidden_size]
        2. 获取LLM物品嵌入并变换 [batch_size, seq_len, hidden_size]
        3. 融合两种嵌入 (可选)
        4. 通过Mamba层进行序列建模
        5. 提取最后一个有效位置的表示作为序列表示

        Args:
            item_seq (torch.Tensor): 物品序列，shape: [batch_size, seq_len]
            item_seq_len (torch.Tensor): 序列长度，shape: [batch_size]

        Returns:
            torch.Tensor: 序列表示向量，shape: [batch_size, hidden_size]
        """

        # ======================== 第1步：获取LLM嵌入向量 ========================
        # 从第一个样本中提取物品ID序列（注意：这里只处理batch中的第一个样本）
        # shape: [seq_len] - 物品ID列表
        item_index = item_seq.clone()[0].cpu().tolist()

        # 根据物品ID从LLM嵌入矩阵中提取对应的嵌入向量
        # shape: [seq_len, 1024] - 每个物品对应1024维LLM嵌入
        item_llm_vec = np.array(self.llm_matrix.iloc[item_index, :])
        item_llm_vec = torch.tensor(item_llm_vec).float().cuda()

        # 通过卷积网络处理LLM嵌入，降维到hidden_size
        # Input shape: [seq_len, 1024] -> Output shape: [seq_len, hidden_size]
        llm_output = self.l1(item_llm_vec)

        # ======================== 第2步：获取传统物品嵌入 ========================
        # 通过Embedding层获取传统物品嵌入
        # Input shape: [batch_size, seq_len] -> Output shape: [batch_size, seq_len, hidden_size]
        item_emb = self.item_embedding(item_seq)

        # ======================== 第3步：嵌入融合（当前被注释掉）========================
        # 理论上的融合公式：input_emb = alpha * item_emb + beta * llm_output
        # 注意：下面的融合代码被注释掉了，当前只使用传统嵌入
        input_emb = self.alpha * item_emb + self.beta * llm_output

        # 当前实际使用：只使用传统嵌入
        # shape: [batch_size, seq_len, hidden_size]
        input_emb = item_emb

        # ======================== 第4步：应用Dropout和LayerNorm ========================
        # shape保持不变: [batch_size, seq_len, hidden_size]
        input_emb = self.dropout(input_emb)
        input_emb = self.LayerNorm(input_emb)

        # ======================== 第5步：多层Mamba编码 ========================
        # 通过多层Mamba编码器进行序列建模
        for i in range(self.num_layers):
            # 每层Mamba的输入输出shape都是: [batch_size, seq_len, hidden_size]
            item_emb = self.mamba_layers[i](input_emb)

        # ======================== 第6步：提取序列表示 ========================
        # 从最后一个有效位置提取序列表示
        # Input shape: [batch_size, seq_len, hidden_size] -> Output shape: [batch_size, hidden_size]
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)

        return seq_output

    def calculate_loss(self, interaction):
        """
        计算训练损失

        支持两种损失函数：
        1. BPR损失：贝叶斯个性化排序，适合隐式反馈
        2. CE损失：交叉熵损失，适合显式反馈

        Args:
            interaction (dict): 包含用户交互数据的字典
                - ITEM_SEQ: 物品序列 [batch_size, seq_len]
                - ITEM_SEQ_LEN: 序列长度 [batch_size]
                - POS_ITEM_ID: 正样本物品ID [batch_size]

        Returns:
            torch.Tensor: 损失值 (标量)
        """
        # 提取输入数据
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 获取序列表示
        # shape: [batch_size, hidden_size]
        seq_output = self.forward(item_seq, item_seq_len)

        # 提取正样本物品ID
        # shape: [batch_size]
        pos_items = interaction[self.POS_ITEM_ID]
        # neg_items = interaction[self.NEG_ITEM_ID]

        if self.loss_type == "BPR":
            # ======================== BPR损失计算 ========================
            # 获取正样本和负样本物品嵌入
            # shape: [batch_size, hidden_size]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)  # 注意：neg_items变量未定义，这是一个bug

            # 计算正样本和负样本得分
            # shape: [batch_size]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)

            # 计算BPR损失：log sigmoid(pos_score - neg_score)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            # ======================== 交叉熵损失计算 ========================
            # 获取所有物品的嵌入作为分类器权重
            # shape: [n_items, hidden_size]
            test_item_emb = self.item_embedding.weight

            # 计算所有物品的得分
            # shape: [batch_size, n_items]
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

            # 计算交叉熵损失
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        """
        单物品预测（用于评估特定物品的得分）

        Args:
            interaction (dict): 包含用户交互数据的字典
                - ITEM_SEQ: 物品序列 [batch_size, seq_len]
                - ITEM_SEQ_LEN: 序列长度 [batch_size]
                - ITEM_ID: 候选物品ID [batch_size]

        Returns:
            torch.Tensor: 物品得分 [batch_size]
        """
        # 提取输入数据
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        # 获取序列表示
        # shape: [batch_size, hidden_size]
        seq_output = self.forward(item_seq, item_seq_len)

        # 获取测试物品嵌入
        # shape: [batch_size, hidden_size]
        test_item_emb = self.item_embedding(test_item)

        # 计算序列表示与物品嵌入的内积得分
        # shape: [batch_size]
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        全排序预测（用于生成所有物品的推荐排序）

        Args:
            interaction (dict): 包含用户交互数据的字典
                - ITEM_SEQ: 物品序列 [batch_size, seq_len]
                - ITEM_SEQ_LEN: 序列长度 [batch_size]

        Returns:
            torch.Tensor: 所有物品的得分 [batch_size, n_items]
        """
        # 提取输入数据
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        # 获取序列表示
        # shape: [batch_size, hidden_size]
        seq_output = self.forward(item_seq, item_seq_len)

        # 获取所有物品的嵌入
        # shape: [n_items, hidden_size]
        test_items_emb = self.item_embedding.weight

        # 计算序列表示与所有物品嵌入的内积得分
        # shape: [batch_size, n_items]
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        return scores


class MambaLayer(nn.Module):
    """
    Mamba层实现

    该层是M2Rec模型的核心组件，包含以下组件：
    1. Mamba状态空间模型 - 用于序列建模
    2. LayerNorm + Dropout - 归一化和正则化
    3. FeedForward网络 - 非线性变换

    参数量：约为 3 * expand * d_model^2

    Args:
        d_model (int): 模型维度
        d_state (int): SSM状态维度
        d_conv (int): 卷积核大小
        expand (int): 扩展比例
        dropout (float): Dropout概率
        num_layers (int): 总层数（用于判断是否使用残差连接）
    """

    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # ======================== Mamba状态空间模型 ========================
        self.mamba = Mamba(
            d_model=d_model,  # 模型维度
            d_state=d_state,  # SSM状态维度
            d_conv=d_conv,  # 卷积核大小
            expand=expand,  # 扩展比例
        )

        # ======================== 归一化和正则化 ========================
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

        # ======================== 前馈网络 ========================
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        """
        Mamba层前向传播

        Args:
            input_tensor (torch.Tensor): 输入张量 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        # ======================== Mamba状态空间建模 ========================
        # shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        hidden_states = self.mamba(input_tensor)

        # ======================== 残差连接策略 ========================
        if self.num_layers == 1:
            # 单层：不使用残差连接
            hidden_states = self.LayerNorm(self.dropout(hidden_states))
        else:
            # 多层：使用残差连接
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)

        # ======================== 前馈网络 ========================
        # shape保持不变: [batch_size, seq_len, d_model]
        hidden_states = self.ffn(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    """
    前馈神经网络（FFN）

    实现标准的两层前馈网络：
    d_model -> inner_size -> d_model

    包含残差连接和LayerNorm

    Args:
        d_model (int): 输入/输出维度
        inner_size (int): 隐藏层维度，通常为d_model的4倍
        dropout (float): Dropout概率，默认0.2
    """

    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        # ======================== 两层线性变换 ========================
        self.w_1 = nn.Linear(d_model, inner_size)  # 第一层：升维
        self.w_2 = nn.Linear(inner_size, d_model)  # 第二层：降维

        # ======================== 激活函数和正则化 ========================
        self.activation = nn.GELU()  # GELU激活函数
        self.dropout = nn.Dropout(dropout)  # Dropout正则化
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)  # 层归一化

    def forward(self, input_tensor):
        """
        前馈网络前向传播

        Args:
            input_tensor (torch.Tensor): 输入张量 [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: 输出张量 [batch_size, seq_len, d_model]
        """
        # ======================== 第一层变换 ========================
        # shape: [batch_size, seq_len, d_model] -> [batch_size, seq_len, inner_size]
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)  # GELU激活
        hidden_states = self.dropout(hidden_states)  # Dropout正则化

        # ======================== 第二层变换 ========================
        # shape: [batch_size, seq_len, inner_size] -> [batch_size, seq_len, d_model]
        hidden_states = self.w_2(hidden_states)
        hidden_states = self.dropout(hidden_states)  # Dropout正则化

        # ======================== 残差连接 + LayerNorm ========================
        # shape保持: [batch_size, seq_len, d_model]
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BiMamba_Layer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        mamba = {'mamba2': Mamba,  # Mamba2
                 'mamba1': Mamba,
                 # 'mamba3':Mamba,
                 }
        model1 = mamba['mamba1']
        model2 = mamba['mamba2']
        # model3 = mamba['mamba3']
        self.token_forward = model1(
            d_model=self.config['hidden_size'],
            d_state=32,
            d_conv=4,
            expand=2
        )

        self.token_backward = model2(
            d_model=self.config['hidden_size'],
            d_state=32,
            d_conv=4,
            expand=2
        )
        # self.combine = model3(
        # 	d_model = self.config['hidden_size'],
        # 	d_state = 32,
        # 	d_conv = 4,
        # 	expand = 2
        # )
        self.dropout1 = self.config["dropout_prob"]
        self.d_model = self.config["hidden_size"]
        self.activation = nn.GELU()
        # self.project = nn.Linear(config.hidden_size*2,config.hidden_size)
        self.project = nn.Linear(2, 1)
        self.dropout = nn.Dropout(self.dropout1)
        self.LayerNorm = nn.LayerNorm(self.d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=self.d_model, inner_size=self.d_model * 4, dropout=self.dropout1)
        self.num_layers = self.config["num_layers"]

    # def flip(self,hidden_states, lengths):

    #     lengths = lengths[0]
    #     batch_data = []
    #     for i in range(hidden_states.shape[0]):
    #         # import pdb
    #         # pdb.set_trace()
    #         data = hidden_states[i][:lengths[i]].flip(dims=[0])
    #         padding = hidden_states[i][lengths[i]:]
    #         batch_data.append(torch.cat([data,padding],dim=0).unsqueeze(0))
    #     hidden_states = torch.cat(batch_data,dim=0)
    #     return hidden_states
    def flip(self, hidden_states, lengths):
        return hidden_states.flip(dims=[1])

    def forward(self,
                hidden_states: torch.Tensor,
                **kwargs):
        lengths = [kwargs['lengths'].data]
        # lengths = kwargs['lengths']
        bi_layer_name = ['mamba1_bi', 'mamba2_bi']
        hidden_forward = self.token_forward(hidden_states)
        # hidden_forward = self.activation(hidden_forward)
        if self.config['layers_name'] in bi_layer_name:
            # hidden_backward = self.token_backward(torch.flip(hidden_states,[1]))
            # hidden_backward = torch.flip(hidden_backward,[1])
            hidden_backward = self.flip(hidden_states, lengths)
            hidden_backward = self.token_backward(hidden_backward)
            # hidden_backward = self.activation(hidden_backward)
            hidden_backward = hidden_backward.flip(dims=[1])
        # output = torch.cat([hidden_forward,hidden_backward],dim=-1)
        # output = hidden_forward + hidden_backward
        # stacked Mamba layers with residual connections

        # print("hidden_states:", hidden_states.size())
        # output = self.project(hidden_states)
        else:
            output = (hidden_forward + hidden_backward) / 2

        # output = (hidden_forward + hidden_backward)/2  # stacked Mamba layers with residual connections
        # output = hidden_forward

        output = torch.cat([hidden_forward.unsqueeze(-1), hidden_backward.unsqueeze(-1)], dim=-1)
        output = self.project(output).squeeze(-1)

        # output = self.combine(output)
        output1 = self.LayerNorm(self.dropout(output) + hidden_states)
        output2 = self.ffn(output1)
        # output1 = self.LayerNorm(self.dropout(hidden_forward) + hidden_states)
        # output2 = self.ffn(output1)

        # output3 = self.LayerNorm(self.dropout(hidden_backward) + hidden_states)
        # output4 = self.ffn(output3)
        # output = (output2 + output4)/2

        # return (output,)
        return output2


class Bimamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        config['layers_name'] = 'mamba1_bi'
        self.layer = nn.ModuleList([BiMamba_Layer(
            config=config,
        ) for _ in range(config['num_layers'])])

    def get_length(self, x):
        return (x == 0.).sum(dim=-1)

    def forward(self, x, **kwargs):
        item_seq_len = kwargs["item_seq_len"]
        # print("mask:", mask, mask.size())
        # lengths = self.get_length(mask).squeeze()
        # print("first check lengths:", lengths)
        lengths = item_seq_len
        # print("first check lengths:", lengths)
        # pritnln()
        # import pdb
        # pdb.set_trace()
        hidden_states = x
        if kwargs['output_all_encoded_layers']:
            all_hidden_states = ()
            all_hidden_states = all_hidden_states + (x,)
        for i, layer_module in enumerate(self.layer):

            layer_outputs = layer_module(hidden_states, lengths=lengths)
            hidden_states = layer_outputs
            if kwargs['output_all_encoded_layers']:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if kwargs['output_all_encoded_layers']:
            return all_hidden_states
        else:
            return hidden_states