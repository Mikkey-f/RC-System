import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from mamba.mamba_ssm.modules.mamba_simple import Mamba


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

class EchoMamba4Rec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(EchoMamba4Rec, self).__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.data = config["dataset"]

        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]
        self.max_seq_length = config["MAX_ITEM_LIST_LENGTH"]
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
            
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)
        
        self.mamba_layers = nn.ModuleList([
            BiMambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                dropout=self.dropout_prob,
                num_layers=self.num_layers,
                max_seq_length=self.max_seq_length
            ) for _ in range(self.num_layers)
        ])

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

        # ======================== LLM嵌入向量加载 ========================
        # 加载预训练的物品元数据嵌入向量 (item_id -> 1024维向量)
        self.llm_vec = load_data('./dataset/{}/item_meta_emb.pkl'.format(self.data))
        # 为padding物品(ID=0)设置零向量
        self.llm_vec[0] = np.array([0.0] * 1024)

        # 将字典转换为DataFrame，便于批量索引
        # shape: [n_items, 1024] - 每行对应一个物品的1024维LLM嵌入
        self.llm_matrix = pd.DataFrame([self.llm_vec[i] for i in range(len(self.llm_vec))])

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
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
        item_emb = self.item_embedding(item_seq)

        # ======================== 第3步：嵌入融合（当前被注释掉）========================
        # 理论上的融合公式：input_emb = alpha * item_emb + beta * llm_output
        # 注意：下面的融合代码被注释掉了，当前只使用传统嵌入
        input_emb = self.alpha * item_emb + self.beta * llm_output

        input_emb = self.dropout(input_emb)
        input_emb = self.LayerNorm(input_emb)
        
        for i in range(self.num_layers):
            item_emb = self.mamba_layers[i](input_emb)
        
        seq_output = self.gather_indexes(item_emb, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
    
class BiMambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers, max_seq_length):
        super().__init__()
        self.num_layers = num_layers
        
        
        self.filter_layer = FilterLayer(max_seq_length=max_seq_length, hidden_size=d_model, dropout_prob=dropout)

        self.norms_forward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norms_backward = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        
        self.mamba_forwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])
        self.mamba_backwards = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        self.glu = GLU(d_model=d_model, dropout=dropout)
        self.multi_query_transformer_block = MultiQueryTransformerBlock(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=dropout
        )
        # ======================== 前馈网络 ========================
        # self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor):
        
        x = input_tensor
        # x = self.filter_layer(x)
        
        for i in range(self.num_layers):
            forward_states = self.mamba_forwards[i](x)
            forward_states = self.norms_forward[i](self.dropout(forward_states) + x)

            reversed_input = torch.flip(x, [1])  
            backward_states = self.mamba_backwards[i](reversed_input)
            backward_states = torch.flip(backward_states, [1]) 
            backward_states = self.norms_backward[i](self.dropout(backward_states) + x)

            x = forward_states + backward_states

        x = self.glu(x)
        # x = self.ffn(x)
        
        return x


class FilterLayer(nn.Module):
    def __init__(self, max_seq_length, hidden_size, dropout_prob):
        super(FilterLayer, self).__init__()
        
        self.complex_weight = nn.Parameter(
            torch.randn(1, max_seq_length // 2 + 1, hidden_size, 2, dtype=torch.float32) * 0.02
        )
        self.out_dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class MultiQueryTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super(MultiQueryTransformerBlock, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.glu = GLU(d_model, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-Query Attention
        x_transposed = x.permute(1, 0, 2)  # Change shape from [B, S, D] to [S, B, D]
        attn_output, _ = self.multihead_attn(x_transposed, x_transposed, x_transposed)
        attn_output = attn_output.permute(1, 0, 2)  # Change shape back to [B, S, D]
        x = x + attn_output
        x = self.norm1(x)
        x = self.dropout1(x)

        # Feed Forward
        glu_output = self.glu(x)
        x = x + glu_output
        x = self.norm2(x)
        x = self.dropout2(x)

        return x
class GLU(nn.Module):
    def __init__(self, d_model, dropout=0.2):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(d_model, d_model * 2)  
        self.fc2 = nn.Linear(d_model, d_model)  
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, x):
        x_transformed = self.fc1(x)
        value, gate = x_transformed.chunk(2, dim=-1)  
        gated_value = value * torch.sigmoid(gate)
        gated_value = self.fc2(gated_value)  
        return self.LayerNorm(self.dropout(gated_value + x))


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
