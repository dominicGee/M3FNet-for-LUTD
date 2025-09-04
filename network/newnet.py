import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mamba_ssm import Mamba


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=64):
        super(ChannelAttentionModule, self).__init__()
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv1d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv1d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SingleBranch(nn.Module):
    def __init__(self):
        super(SingleBranch, self).__init__()
        num_mamba_layers = 1

        # 三个不同卷积核的卷积层
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv9 = nn.Conv1d(1, 64, kernel_size=11, padding=5)

        # CBAM模块
        self.cbam = CBAM(64)  # 假设CBAM接受通道数作为输入

        # 正则化（在CBAM之后添加）
        self.dropout = nn.Dropout(p=0.2)  # 添加Dropout正则化
        self.layernorm = nn.LayerNorm(64)  # 或添加LayerNorm

        # Mamba模块
        self.mambas = nn.ModuleList(
            [Mamba(d_model=64, d_state=16, d_conv=4, expand=2) for _ in range(num_mamba_layers)]
        )

        # 最后的卷积层
        self.final_conv = nn.Conv1d(64, 64, kernel_size=1)

    def forward(self, x):
        # 使用不同卷积核的卷积层
        x3 = F.elu(self.conv3(x))
        x6 = F.elu(self.conv6(x))
        x9 = F.elu(self.conv9(x))

        # 特征图叠加
        # x = x3 + x6 + x9  # 可以选择叠加或拼接
        x = x3
        # CBAM模块
        # x = self.cbam(x)

        # 正则化
        # x = self.dropout(x)  # Dropout
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm需要调整维度

        x = x.permute(0, 2, 1)

        # Mamba模块添加多次跳连
        residual = x  # 初始跳连
        for mamba in self.mambas:
            x = mamba(x) + residual  # 添加跳连
            residual = x  # 更新跳连

        x = x.permute(0, 2, 1)

        # 最后的卷积层
        x = F.relu(self.final_conv(x))

        return x
class SingleBranch1(nn.Module):
    def __init__(self):
        super(SingleBranch1, self).__init__()
        num_mamba_layers = 2

        # 三个不同卷积核的卷积层
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv9 = nn.Conv1d(1, 64, kernel_size=11, padding=5)

        # CBAM模块
        self.cbam = CBAM(64)  # 假设CBAM接受通道数作为输入

        # 正则化（在CBAM之后添加）
        self.dropout = nn.Dropout(p=0.2)  # 添加Dropout正则化
        self.layernorm = nn.LayerNorm(64)  # 或添加LayerNorm

        # Mamba模块
        self.mambas = nn.ModuleList(
            [Mamba(d_model=64, d_state=16, d_conv=4, expand=2) for _ in range(num_mamba_layers)]
        )

        # 最后的卷积层
        self.final_conv = nn.Conv1d(64, 64, kernel_size=1)

    def forward(self, x):
        # 使用不同卷积核的卷积层
        x3 = F.elu(self.conv3(x))
        x6 = F.elu(self.conv6(x))
        x9 = F.elu(self.conv9(x))

        # 特征图叠加
        # x = x3 + x6 + x9  # 可以选择叠加或拼接
        x = x3
        # CBAM模块
        # x = self.cbam(x)

        # 正则化
        # x = self.dropout(x)  # Dropout
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm需要调整维度

        x = x.permute(0, 2, 1)

        # Mamba模块添加多次跳连
        residual = x  # 初始跳连
        for mamba in self.mambas:
            x = mamba(x) + residual  # 添加跳连
            residual = x  # 更新跳连

        x = x.permute(0, 2, 1)

        # 最后的卷积层
        x = F.relu(self.final_conv(x))

        return x
class SingleBranch_GRU(nn.Module):
    def __init__(self):
        super(SingleBranch_GRU, self).__init__()
        num_gru_layers = 20  # 定义GRU的层数，这里对应原来的Mamba层数

        # 三个不同卷积核的卷积层
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv9 = nn.Conv1d(1, 64, kernel_size=11, padding=5)

        # CBAM模块
        self.cbam = CBAM(64)  # 假设CBAM接受通道数作为输入

        # 正则化（在CBAM之后添加）
        self.dropout = nn.Dropout(p=0.2)  # 添加Dropout正则化
        self.layernorm = nn.LayerNorm(64)  # 或添加LayerNorm

        # GRU模块
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=num_gru_layers, batch_first=True)

        # 最后的卷积层
        self.final_conv = nn.Conv1d(64, 64, kernel_size=1)

    def forward(self, x):
        # 使用不同卷积核的卷积层
        x3 = F.elu(self.conv3(x))
        # x6 = F.elu(self.conv6(x))
        # x9 = F.elu(self.conv9(x))

        # 特征图叠加
        # x = torch.cat((x3, x6, x9), dim=1)  # 可以选择叠加或拼接

        # CBAM模块
        # x3 = self.cbam(x3)
        # x6 = self.cbam(x6)
        # x9 = self.cbam(x9)

        # 正则化
        x3 = self.dropout(x3)  # Dropout
        # x6 = self.dropout(x6)
        # x9 = self.dropout(x9)
        x3 = self.layernorm(x3.permute(0, 2, 1)).permute(0, 2, 1)
        # x6 = self.layernorm(x6.permute(0, 2, 1)).permute(0, 2, 1)
        # x9 = self.layernorm(x9.permute(0, 2, 1)).permute(0, 2, 1)
        x3 = x3.permute(0, 2, 1)  # 调整回原来的维度顺序
        # x6 = x6.permute(0, 2, 1)
        # x9 = x9.permute(0, 2, 1)
        _, gru_output1 = self.gru(x3)  # 获取GRU输出，忽略最后一个时刻的隐藏状态输出（这里不需要额外使用）
        # gru_output2, _ = self.gru(x6)
        # gru_output3, _ = self.gru(x9)
        x3 = gru_output1  # 用GRU输出更新x
        # x6 = gru_output2
        # x9 = gru_output3
        # x3 = x3.permute(0, 2, 1)  # 调整回原来的维度顺序
        # x6 = x6.permute(0, 2, 1)
        # x9 = x9.permute(0, 2, 1)

        # 最后的卷积层
        x3 = F.relu(self.final_conv(x3))
        # x6 = F.relu(self.final_conv(x6))
        # x9 = F.relu(self.final_conv(x9))
        # x = torch.cat((x3, x6, x9), dim=1)
        return x3

class SingleBranch_att(nn.Module):
    def __init__(self):
        super(SingleBranch_att, self).__init__()
        num_heads = 8  # 定义多头自注意力的头数
        num_layers = 5  # 对应原来的Mamba层数
        num_mamba_layers = 20

        self.conv3 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # CBAM模块
        self.cbam = CBAM(64)  # 假设CBAM接受通道数作为输入

        # 正则化（在CBAM之后添加）
        self.dropout = nn.Dropout(p=0.2)  # 添加Dropout正则化
        self.layernorm = nn.LayerNorm(64)  # 或添加LayerNorm

        # 多头自注意力模块
        self.multihead_attentions = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=0.1) for _ in range(num_layers)]
        )
        self.mambas = nn.ModuleList(
            [Mamba(d_model=64, d_state=16, d_conv=4, expand=2) for _ in range(num_mamba_layers)])  # 假设MambaBlock需要输入通道数

        # 最后的卷积层
        self.final_conv = nn.Conv1d(64, 64, kernel_size=1)

    def forward(self, x):
        # 使用不同卷积核的卷积层
        x3 = F.elu(self.conv3(x))
        x3 = F.relu(self.conv6(x3))

        x = x3
        x = self.layernorm(x.permute(0, 2, 1)).permute(0, 2, 1)  # LayerNorm需要调整维度

        x = x.permute(0, 2, 1)

        # 多头自注意力模块添加多次跳连
        residual = x  # 初始跳连
        # for multihead_attention in self.multihead_attentions:
        #     # 调整输入维度以适配多头自注意力的输入要求（batch_size, sequence_length, embed_dim）
        #     query = key = value = x
        #     x, _ = multihead_attention(query, key, value)
        #     x = x + residual  # 添加跳连
        #     residual = x  # 更新跳连

        for mamba in self.mambas:
            x = mamba(x) + residual  # 添加跳连
            residual = x  # 更新跳连

        x = x.permute(0, 2, 1)

        # 最后的卷积层
        x = F.relu(self.final_conv(x))

        return x

class UTransNet(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(UTransNet, self).__init__()
        # 分支1的1D卷积层
        self.singleBranch1 = SingleBranch()
        self.singleBranch2 = SingleBranch()
        self.singleBranch3 = SingleBranch()

        # 特征融合后的1D卷积层
        self.fusion_conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # 自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        # 全连接层
        self.fc1 = nn.Linear(64 * 500, 128)  # 注意：根据您的数据调整维度

        self.fc_task1 = nn.Linear(128, num_classes_task1)
        self.fc_task2 = nn.Linear(128, num_classes_task2)

        self.dropout = nn.Dropout(0.5)
        self.l2_reg = 1e-4  # L2正则化系数

    def forward(self, seq1, seq2, seq3):
        seq1 = seq1.unsqueeze(1)  # 增加通道维度
        seq2 = seq2.unsqueeze(1)  # 增加通道维度
        seq3 = seq3.unsqueeze(1)  # 增加通道维度

        # 处理第一序列
        x1 = F.relu(self.singleBranch1(seq1))
        x1 = F.max_pool1d(x1, 2)

        # 处理第二序列
        x2 = F.relu(self.singleBranch2(seq2))
        x2 = F.max_pool1d(x2, 2)

        # 处理第三序列
        x3 = F.relu(self.singleBranch3(seq3))
        x3 = F.max_pool1d(x3, 2)

        # 特征融合
        # x = torch.cat((x1, x2, x3), dim=1)
        x1 = x1.permute(2, 0, 1)  # (L, N, C)
        x2 = x2.permute(2, 0, 1)
        x3 = x3.permute(2, 0, 1)
        x, _ = self.attention(x1, x2, x3)
        x = x.permute(1, 2, 0)  # (N, C, L)

        x = F.relu(self.fusion_conv(x))
        x = F.max_pool1d(x, 2)

        # 展平特征
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 任务1输出
        out_task1 = self.fc_task1(x)
        # 任务2输出
        out_task2 = self.fc_task2(x)

        return out_task1, out_task2


    def regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss
class UTransNet2(nn.Module):
    def __init__(self, num_classes_task1, num_classes_task2):
        super(UTransNet2, self).__init__()
        # 分支1的1D卷积层
        self.singleBranch1 = SingleBranch1()
        self.singleBranch2 = SingleBranch1()
        self.singleBranch3 = SingleBranch1()

        # 特征融合后的1D卷积层
        self.fusion_conv = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        # 自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)
        # 全连接层
        self.fc1 = nn.Linear(64 * 500, 128)  # 注意：根据您的数据调整维度

        self.fc_task1 = nn.Linear(128, num_classes_task1)
        self.fc_task2 = nn.Linear(128, num_classes_task2)

        self.dropout = nn.Dropout(0.5)
        self.l2_reg = 1e-4  # L2正则化系数

    def forward(self, seq1, seq2, seq3):
        seq1 = seq1.unsqueeze(1)  # 增加通道维度
        seq2 = seq2.unsqueeze(1)  # 增加通道维度
        seq3 = seq3.unsqueeze(1)  # 增加通道维度

        # 处理第一序列
        x1 = F.relu(self.singleBranch1(seq1))
        x1 = F.max_pool1d(x1, 2)

        # 处理第二序列
        x2 = F.relu(self.singleBranch2(seq2))
        x2 = F.max_pool1d(x2, 2)

        # 处理第三序列
        x3 = F.relu(self.singleBranch3(seq3))
        x3 = F.max_pool1d(x3, 2)

        # 特征融合
        # x = torch.cat((x1, x2, x3), dim=1)
        x1 = x1.permute(2, 0, 1)  # (L, N, C)
        x2 = x2.permute(2, 0, 1)
        x3 = x3.permute(2, 0, 1)
        x, _ = self.attention(x1, x2, x3)
        x = x.permute(1, 2, 0)  # (N, C, L)

        x = F.relu(self.fusion_conv(x))
        x = F.max_pool1d(x, 2)

        # 展平特征
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 任务1输出
        out_task1 = self.fc_task1(x)
        # 任务2输出
        out_task2 = self.fc_task2(x)

        return out_task1, out_task2


    def regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.sum(param ** 2)
        return self.l2_reg * l2_loss
