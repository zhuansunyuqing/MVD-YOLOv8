import torch
import torch.nn as nn
import torch.nn.functional as F


#对同一张量进行不同的特征提取
class FeatureExtraction(nn.Module):
    def __init__(self, channels, k_size):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = k_size, padding = (k_size-1)//2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x

#使用ECA的通道注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)
    
    
#使用CBAM的空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
 
    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
    
    

    
class Weight(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(Weight, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        
        self.weight1 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1, bias=False)
        self.weight2 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1, bias=False)
        self.weight3 = nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1, bias=False)

    def forward(self, x1, x2, x3):
        # Reduce channels with 1x1 convolutions and compute attention weights
        x1_reduced = self.conv1(x1)
        x2_reduced = self.conv2(x2)
        x3_reduced = self.conv3(x3)
        
        # Compute attention weights using 1x1 convolution
        w1 = self.weight1(x1_reduced)
        w2 = self.weight2(x2_reduced)
        w3 = self.weight3(x3_reduced)
        
        # Combine weights using softmax to normalize
        weights = torch.cat([w1, w2, w3], dim=1)
        weights = F.softmax(weights, dim=1)
        
        # Split the weights back
        w1, w2, w3 = torch.split(weights, 1, dim=1)
        
        # Fuse the features with the attention weights
        out = w1 * x1 + w2 * x2 + w3 * x3
        return out
    
#将分好的特征图进行注意力处理
class CSFA(nn.Module):
    def __init__(self, in_channel, kernel_size=7):
        super().__init__()
        self.feature_L = FeatureExtraction(in_channel, 1)
        self.feature_M = FeatureExtraction(in_channel, 3)
        self.feature_S = FeatureExtraction(in_channel, 5)
        self.channel_attention = ChannelAttention(in_channel)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.weight = Weight(in_channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):        
        x_L = self.feature_L(x)
        x_L = self.channel_attention(x_L)
        x_L = self.spatial_attention(x_L)
        
        x_M = self.feature_M(x)
        x_M = self.channel_attention(x_M)
        x_M = self.spatial_attention(x_M)
        
        x_S = self.feature_S(x)
        x_S = self.channel_attention(x_S)
        x_S = self.spatial_attention(x_S)
        x = self.weight(x_S, x_M, x_L)
        return self.sigmoid(x)
    



























class ChannelAttention(nn.Module):
 
    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))
 
 
class SpatialAttention(nn.Module):
    """Spatial-attention module."""
 
    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()
 
    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))
 
 
class CBAM(nn.Module):
    """Convolutional Block Attention Module."""
 
    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)
 
    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))