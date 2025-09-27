import copy
import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import MultiHeadAttention
from recbole.model.loss import BPRLoss
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_


class BSARecModel(SequentialRecommender):
    def __init__(self, config, dataset):
        super(BSARecModel, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]  # 隐藏层维度，默认64
        self.loss_type = config["loss_type"]  # 损失函数类型：'BPR'或'CE'
        self.num_layers = config["num_layers"]  # Mamba层数，默认1
        self.dropout_prob = config["dropout_prob"]  # Dropout概率，默认0.2

        # ======================== Mamba块超参数 ========================
        self.d_state = config["d_state"]  # SSM状态扩展因子，默认32
        self.d_conv = config["d_conv"]  # 局部卷积宽度，默认4
        self.expand = config["expand"]  # 块扩展因子，默认2
        self.data = config["dataset"]  # 数据集名称
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]  # 最大序列长度

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()  # 贝叶斯个性化排序损失
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()  # 交叉熵损失
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.item_encoder = BSARecEncoder(config)
        self.utils = Utils(config, self.n_items)
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
        # extended_attention_mask: batch_size * attention_size * seq_length * seq_length
        extended_attention_mask = self.utils.get_attention_mask(item_seq)
        # sequence_emb: batch_size * seq_length * hidden_size
        sequence_emb = self.utils.add_position_embedding(item_seq)
        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )
        # if all_sequence_output:
        #     sequence_output = item_encoded_layers
        # else:
        #     sequence_output = item_encoded_layers[-1]
        sequence_output = item_encoded_layers[-1]
        seq_output = self.gather_indexes(sequence_output, item_seq_len - 1)
        return seq_output



    def calculate_loss(self, interaction):
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

class BSARecEncoder(nn.Module):
    def __init__(self, config):
        super(BSARecEncoder, self).__init__()
        block = BSARecBlock(config)
        self.num_layers = config["num_layers"]  # Mamba层数，默认1
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(self.num_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=False):
        all_encoder_layers = [ hidden_states ]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            # 只输出最后一层的隐状态
            all_encoder_layers.append(hidden_states) # hidden_states => torch.Size([256, 50, 64])
        return all_encoder_layers

class BSARecBlock(nn.Module):
    def __init__(self, config):
        super(BSARecBlock, self).__init__()
        self.layer = BSARecLayer(config)
        self.hidden_size = config["hidden_size"]  # 隐藏层维度，默认64

        # d_model(int): 输入 / 输出维度
        # inner_size(int): 隐藏层维度，通常为d_model的4倍
        self.feed_forward = FeedForward(d_model=self.hidden_size,inner_size=self.hidden_size * 4)

    def forward(self, hidden_states, attention_mask):
        layer_output = self.layer(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(layer_output)
        return feedforward_output

class BSARecLayer(nn.Module):
    def __init__(self, config):
        super(BSARecLayer, self).__init__()
        # n_heads,
        # hidden_size,
        # hidden_dropout_prob,
        # attn_dropout_prob,
        # layer_norm_eps,
        # todo
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # 隐藏层维度，默认64
        self.dropout_prob = config["dropout_prob"]  # Dropout概率，默认0.2
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.alpha = config["alpha"]

        self.filter_layer = FrequencyLayer(config)
        self.attention_layer = MultiHeadAttention(n_heads=self.n_heads,hidden_size=self.hidden_size,hidden_dropout_prob=self.dropout_prob, attn_dropout_prob=self.attn_dropout_prob, layer_norm_eps=1e-12)

    def forward(self, input_tensor, attention_mask):
        dsp = self.filter_layer(input_tensor)
        gsp = self.attention_layer(input_tensor, attention_mask)
        hidden_states = self.alpha * dsp + ( 1 - self.alpha ) * gsp

        return hidden_states

"""
频率层，进行高低频率分离
"""
class FrequencyLayer(nn.Module):
    def __init__(self, config):
        super(FrequencyLayer, self).__init__()
        self.dropout_prob = config["dropout_prob"]  # Dropout概率，默认0.2
        self.hidden_size = config["hidden_size"]  # 隐藏层维度，默认64
        self.c = config["c"] // 2 + 1
        self.out_dropout = nn.Dropout(self.dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.sqrt_beta = nn.Parameter(torch.randn(1, 1, self.hidden_size))

    # 拆分高低频信号
    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        # 转换为频率信号

        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        # 前c个是低频信号
        low_pass[:, self.c:, :] = 0
        # 重新转换回时域
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        # 得到高频时域
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.sqrt_beta**2) * high_pass

        # Add & Norm
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

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
class Utils(nn.Module):
    def __init__(self, config, n_items):
        super(Utils, self).__init__()
        self.n_items = n_items
        self.hidden_size = config["hidden_size"]
        self.dropout_prob = config["dropout_prob"]  # Dropout概率，默认0.2
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
    def get_attention_mask(self, item_seq):
        """
        生成单向注意力掩码（从左到右）

        功能：
        1. 创建单向注意力掩码，只允许模型看到当前位置之前的信息
        2. 结合padding掩码和因果掩码

        参数:
            item_seq: 物品序列 [batch_size, seq_len]

        返回:
            extended_attention_mask: 扩展的注意力掩码
        """
        # 创建padding掩码
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        # 创建因果掩码（上三角矩阵）
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # 上三角矩阵
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)  # 反转：下三角为True
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        # 结合padding掩码和因果掩码
        extended_attention_mask = extended_attention_mask * subsequent_mask

        # 转换为注意力分数格式
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16兼容性
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def add_position_embedding(self, sequence):
        """
        为序列添加位置嵌入

        功能：
        1. 生成位置ID
        2. 获取物品嵌入和位置嵌入
        3. 将两者相加并应用层归一化和dropout

        参数:
            sequence: 输入序列 [batch_size, seq_len]

        返回:
            sequence_emb: 添加位置嵌入后的序列表示
        """
        seq_length = sequence.size(1)

        # 生成位置ID
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)

        # 获取物品嵌入和位置嵌入
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)

        # 将物品嵌入和位置嵌入相加
        # item_embeddings: batch_size * seq_length * hidden_size
        sequence_emb = item_embeddings + position_embeddings

        # 应用层归一化和dropout
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb
