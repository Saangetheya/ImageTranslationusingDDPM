# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin


# 3D version of attention block
class Attention3D(nn.Module):
    def __init__(
            self,
            query_dim,
            heads=8,
            dim_head=64,
            dropout=0.0,
            bias=False,
            out_bias=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout)
        ) if out_bias or dropout > 0 else nn.Linear(inner_dim, query_dim, bias=out_bias)

    def forward(self, x):
        # x shape: [batch, channels, time, height, width]
        b, c, t, h, w = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(b, c, t, h * w).permute(0, 2, 3, 1)  # [b, t, h*w, c]

        # Apply self-attention
        q = self.to_q(x_flat)  # [b, t, h*w, inner_dim]
        k = self.to_k(x_flat)
        v = self.to_v(x_flat)

        # Split heads
        q = q.reshape(b, t, h * w, self.heads, -1).permute(0, 1, 3, 2, 4)  # [b, t, heads, h*w, dim_head]
        k = k.reshape(b, t, h * w, self.heads, -1).permute(0, 1, 3, 2, 4)
        v = v.reshape(b, t, h * w, self.heads, -1).permute(0, 1, 3, 2, 4)

        # Attention calculation
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # [b, t, heads, h*w, h*w]
        attn = dots.softmax(dim=-1)

        # Apply attention weights
        out = torch.matmul(attn, v)  # [b, t, heads, h*w, dim_head]

        # Reshape
        out = out.permute(0, 1, 3, 2, 4).reshape(b, t, h * w, -1)  # [b, t, h*w, inner_dim]

        # Project back to original dimension
        out = self.to_out(out)  # [b, t, h*w, c]

        # Reshape to original shape
        out = out.permute(0, 3, 1, 2).reshape(b, c, t, h, w)

        return out


# 3D version of downsample module
class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2

        if use_conv:
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=3, stride=(1, stride, stride), padding=padding
            )
        else:
            self.conv = nn.AvgPool3d(kernel_size=(1, stride, stride), stride=(1, stride, stride))

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1, 0, 0)  # 保持时间维度不变
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        hidden_states = self.conv(hidden_states)

        return hidden_states


# 3D version of upsample module
class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv3d(
                self.channels, self.out_channels, kernel_size=3, padding=1
            )

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        # Upsample only in spatial dimensions, keep temporal dimension unchanged
        hidden_states = F.interpolate(
            hidden_states, scale_factor=(1, 2, 2), mode="nearest"
        )

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states


# 3D version of ResnetBlock
class ResnetBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-5,
            non_linearity="swish",
            time_embedding_norm="default",
            kernel=None,
            output_scale_factor=1.0,
            use_in_shortcut=None,
            up=False,
            down=False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        # Normalize input
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        # First convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        # Normalize intermediate feature map
        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        # Nonlinear activation function
        self.nonlinearity = nn.SiLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Second convolution
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # Handle residual connection
        if self.in_channels != out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def forward(self, input_tensor, temb=None):
        hidden_states = input_tensor

        # Pre-normalization
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # First convolution
        hidden_states = self.conv1(hidden_states)

        # Add time embedding
        if temb is not None and self.time_emb_proj is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))
            temb = temb[:, :, None, None, None]  # Add dimension to match 3D data
            hidden_states = hidden_states + temb

        # Second normalization
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        # Dropout
        hidden_states = self.dropout(hidden_states)

        # Second convolution
        hidden_states = self.conv2(hidden_states)

        # Handle residual connection
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        # Scale by output factor
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


# 3D version of UNetMidBlock
class UNetMidBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            temb_channels,
            dropout=0.0,
            resnet_eps=1e-5,
            resnet_act_fn="swish",
            output_scale_factor=1.0,
            resnet_time_scale_shift="default",
            attention_head_dim=None,
            resnet_groups=32,
            attn_groups=None,
            add_attention=True,
    ):
        super().__init__()
        self.add_attention = add_attention
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # ResNet block
        self.resnet1 = ResnetBlock3D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            eps=resnet_eps,
            output_scale_factor=output_scale_factor,
        )

        # Attention block
        if self.add_attention:
            self.attn1 = Attention3D(
                query_dim=in_channels,
                heads=attention_head_dim or in_channels // 64,
                dim_head=64,
                dropout=dropout,
            )

        # Second ResNet block
        self.resnet2 = ResnetBlock3D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            eps=resnet_eps,
            output_scale_factor=output_scale_factor,
        )

    def forward(self, hidden_states, temb=None):
        hidden_states = self.resnet1(hidden_states, temb)

        if self.add_attention:
            hidden_states = self.attn1(hidden_states)

        hidden_states = self.resnet2(hidden_states, temb)

        return hidden_states


# 3D version of DownBlock
class DownBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-5,
            resnet_act_fn="swish",
            resnet_groups=32,
            resnet_time_scale_shift="default",
            attention_head_dim=None,
            add_downsample=True,
            downsample_padding=1,
            downsample_type="conv",
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            in_channels_res = in_channels if i == 0 else out_channels

            # Add ResNet block
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels_res,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                )
            )

        self.add_downsample = add_downsample
        if add_downsample:
            self.downsampler = Downsample3D(
                out_channels, use_conv=(downsample_type == "conv"),
                padding=downsample_padding, out_channels=out_channels
            )

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


# 3D version of DownBlock with attention mechanism
class AttnDownBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-5,
            resnet_act_fn="swish",
            resnet_groups=32,
            resnet_time_scale_shift="default",
            attention_head_dim=None,
            add_downsample=True,
            downsample_padding=1,
            downsample_type="conv",
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            in_channels_res = in_channels if i == 0 else out_channels

            # Add ResNet block
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels_res,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                )
            )

            # Add attention block
            self.attentions.append(
                Attention3D(
                    query_dim=out_channels,
                    heads=attention_head_dim or out_channels // 64,
                    dim_head=64,
                    dropout=dropout,
                )
            )

        self.add_downsample = add_downsample
        if add_downsample:
            self.downsampler = Downsample3D(
                out_channels, use_conv=(downsample_type == "conv"),
                padding=downsample_padding, out_channels=out_channels
            )

    def forward(self, hidden_states, temb=None):
        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsampler(hidden_states)
            output_states += (hidden_states,)

        return hidden_states, output_states


# 3D version of UpBlock
class UpBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            prev_output_channel,
            out_channels,
            temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-5,
            resnet_act_fn="swish",
            resnet_groups=32,
            resnet_time_scale_shift="default",
            attention_head_dim=None,
            add_upsample=True,
            upsample_type="conv",
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # Add ResNet block
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                )
            )

        self.add_upsample = add_upsample
        if add_upsample:
            self.upsampler = Upsample3D(
                out_channels, use_conv=(upsample_type == "conv"),
                out_channels=out_channels
            )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet in self.resnets:
            # Concatenate residual connection
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # Pass through ResNet block
            hidden_states = resnet(hidden_states, temb)

        if self.add_upsample:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


# 3D version of UpBlock with attention mechanism
class AttnUpBlock3D(nn.Module):
    def __init__(
            self,
            in_channels,
            prev_output_channel,
            out_channels,
            temb_channels,
            dropout=0.0,
            num_layers=1,
            resnet_eps=1e-5,
            resnet_act_fn="swish",
            resnet_groups=32,
            resnet_time_scale_shift="default",
            attention_head_dim=None,
            add_upsample=True,
            upsample_type="conv",
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            # Add ResNet block
            self.resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                )
            )

            # Add attention block
            self.attentions.append(
                Attention3D(
                    query_dim=out_channels,
                    heads=attention_head_dim or out_channels // 64,
                    dim_head=64,
                    dropout=dropout,
                )
            )

        self.add_upsample = add_upsample
        if add_upsample:
            self.upsampler = Upsample3D(
                out_channels, use_conv=(upsample_type == "conv"),
                out_channels=out_channels
            )

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None):
        for resnet, attn in zip(self.resnets, self.attentions):
            # Concatenate residual connection
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            # Pass through ResNet block and attention block
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states)

        if self.add_upsample:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


# Helper function to get 3D down block
def get_down_block_3d(
        down_block_type,
        num_layers,
        in_channels,
        out_channels,
        temb_channels,
        add_downsample,
        resnet_eps,
        resnet_act_fn,
        resnet_groups,
        resnet_time_scale_shift,
        attention_head_dim,
        downsample_padding,
        downsample_type,
        dropout=0.0,
):
    down_block_type = down_block_type.replace("2D", "3D")
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            dropout=dropout,
        )
    elif down_block_type == "AttnDownBlock3D":
        return AttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            downsample_padding=downsample_padding,
            downsample_type=downsample_type,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown down block type: {down_block_type}")


# Helper function to get 3D up block
def get_up_block_3d(
        up_block_type,
        num_layers,
        in_channels,
        out_channels,
        prev_output_channel,
        temb_channels,
        add_upsample,
        resnet_eps,
        resnet_act_fn,
        resnet_groups,
        resnet_time_scale_shift,
        attention_head_dim,
        upsample_type,
        dropout=0.0,
):
    up_block_type = up_block_type.replace("2D", "3D")
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            upsample_type=upsample_type,
            dropout=dropout,
        )
    elif up_block_type == "AttnUpBlock3D":
        return AttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            prev_output_channel=prev_output_channel,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            attention_head_dim=attention_head_dim,
            upsample_type=upsample_type,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown up block type: {up_block_type}")


@dataclass
class UNet3DOutput(BaseOutput):
    """
    Output of the [`UNet3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, temporal, height, width)`):
            Hidden state output of the model's last layer.
    """
    sample: torch.Tensor


class UNet3DModel(ModelMixin, ConfigMixin):
    """
    A 3D UNet model that receives a noisy sample and timestep, and returns an output of sample shape.

    This model inherits from [`ModelMixin`]. Check the parent class documentation for all model-common methods.

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample. Dimensions must be multiples of `2 ** (len(block_out_channels) - 1)`.
        in_channels (`int`, *optional*, defaults to 3): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 3): Number of channels in the output.
        center_input_sample (`bool`, *optional*, defaults to `False`): Whether to center the input sample.
        time_embedding_type (`str`, *optional*, defaults to `"positional"`): Type of time embedding to use.
        freq_shift (`int`, *optional*, defaults to 0): Frequency shift for Fourier time embedding.
        flip_sin_to_cos (`bool`, *optional*, defaults to `True`): Whether to flip sin to cos for Fourier time embedding.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("DownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D", "AttnDownBlock3D")`):
            Tuple of down block types.
        mid_block_type (`str`, *optional*, defaults to `"UNetMidBlock3D"`): Block type for the UNet middle part.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("AttnUpBlock3D", "AttnUpBlock3D", "AttnUpBlock3D", "UpBlock3D")`):
            Tuple of up block types.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(224, 448, 672, 896)`):
            Tuple of block output channels.
        layers_per_block (`int`, *optional*, defaults to `2`): Number of layers per block.
        mid_block_scale_factor (`float`, *optional*, defaults to `1`): Scale factor for the middle block.
        downsample_padding (`int`, *optional*, defaults to `1`): Padding for downsample convolution.
        act_fn (`str`, *optional*, defaults to `"silu"`): Activation function to use.
        attention_head_dim (`int`, *optional*, defaults to `8`): Dimension of attention heads.
        norm_num_groups (`int`, *optional*, defaults to `32`): Number of groups for group normalization.
        norm_eps (`float`, *optional*, defaults to `1e-5`): Epsilon value for normalization.
        temporal_length (`int`, *optional*, defaults to `5`): Length of the temporal dimension.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: Optional[Union[int, Tuple[int, int]]] = None,
            in_channels: int = 3,
            out_channels: int = 3,
            center_input_sample: bool = False,
            time_embedding_type: str = "positional",
            time_embedding_dim: Optional[int] = None,
            freq_shift: int = 0,
            flip_sin_to_cos: bool = True,
            down_block_types: Tuple[str, ...] = (
                    "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types: Tuple[str, ...] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
            layers_per_block: int = 2,
            mid_block_scale_factor: float = 1,
            downsample_padding: int = 1,
            downsample_type: str = "conv",
            upsample_type: str = "conv",
            dropout: float = 0.0,
            act_fn: str = "silu",
            attention_head_dim: Optional[int] = 8,
            norm_num_groups: int = 32,
            attn_norm_num_groups: Optional[int] = None,
            norm_eps: float = 1e-5,
            resnet_time_scale_shift: str = "default",
            add_attention: bool = True,
            class_embed_type: Optional[str] = None,
            num_class_embeds: Optional[int] = None,
            num_train_timesteps: Optional[int] = None,
            temporal_length: int = 5,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.temporal_length = temporal_length
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"The same number of elements must be provided for `down_block_types` and `up_block_types`. `down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"The same number of elements must be provided for `block_out_channels` and `down_block_types`. `block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        # Input convolution
        self.conv_in = nn.Conv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1, 1))

        # Time embedding
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]
        elif time_embedding_type == "learned":
            self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # Class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # Downsampling path
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block_3d(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                downsample_padding=downsample_padding,
                resnet_time_scale_shift=resnet_time_scale_shift,
                downsample_type=downsample_type,
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

        # Middle block
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=mid_block_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift,
            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
            resnet_groups=norm_num_groups,
            attn_groups=attn_norm_num_groups,
            add_attention=add_attention,
        )

        # Upsampling path
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block_3d(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim if attention_head_dim is not None else output_channel,
                resnet_time_scale_shift=resnet_time_scale_shift,
                upsample_type=upsample_type,
                dropout=dropout,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Output layers
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
            self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            class_labels: Optional[torch.Tensor] = None,
            return_dict: bool = True,
    ) -> Union[UNet3DOutput, Tuple]:
        r"""
        Forward method for [`UNet3DModel`].

        Args:
            sample (`torch.Tensor`):
                Noisy input tensor with shape `(batch, channel, temporal, height, width)`.
            timestep (`torch.Tensor` or `float` or `int`): Timestep for denoising an input.
            class_labels (`torch.Tensor`, *optional*, defaults to `None`):
                Optional class labels for conditional generation. Their embeddings will be added to the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return [`~models.unets.unet_3d.UNet3DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_3d.UNet3DOutput`] or `tuple`:
                If `return_dict` is True, returns [`~models.unets.unet_3d.UNet3DOutput`],
                otherwise returns a tuple with the sample tensor as the first element.
        """
        # 0. Center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. Time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # Broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. So we need to cast here.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding must be initialized to use class conditioning")

        # 2. Pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. Down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. Mid
        sample = self.mid_block(sample, emb)

        # 5. Up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. Post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet3DOutput(sample=sample)