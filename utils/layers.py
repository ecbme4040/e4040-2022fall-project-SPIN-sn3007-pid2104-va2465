import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras import regularizers

from utils.layer_funcs import *


#a simple 2 Layered Perceptron layer with gelu(gaussian error linear units)
class Mlp(tf.keras.layers.Layer):
    def __init__(self, in_units, hidden_units=None, out_units=None, drop=0., regularizer=None,pre_name=''):
        super().__init__()
        out_units = out_units or in_units
        hidden_units = hidden_units or in_units
        self.dense1 = Dense(hidden_units,activation="gelu",kernel_regularizer= regularizer,name=f'{pre_name}/mlp/dense1' )#regularizer=tf.keras.regularizers.L2(0.1)
        self.dense2 = Dense(out_units,kernel_regularizer= regularizer,name=f'{pre_name}/mlp/dense2' )
        self.drop = Dropout(drop)
        self.pre_name = pre_name


    def call(self, x):
        x = self.dense1(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        return x


# This is the Transformer architecture inspired from Vision transformer where attention is applied to localised fields of input defined as windows.
# Windows reduces the size of the attention-matrix to be computed and making them feasible.
class winAttention(tf.keras.layers.Layer):
    def __init__(self, dim, win_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,regularizer=None,pre_name=''):
        super().__init__()

        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.pre_name=pre_name

        self.qkv = Dense(dim * 3, use_bias=qkv_bias,kernel_regularizer= regularizer,name=f'{self.pre_name}/win_attn/qkv')
        self.attn_drop = Dropout(attn_drop)
        self.proj = Dense(dim,kernel_regularizer= regularizer,name=f'{self.pre_name}/win_attn/proj')
        self.proj_drop = Dropout(proj_drop)


    def build(self, input_shape):

        self.rltv_position_bias_table = self.add_weight(f'{self.pre_name}/win_attn/rltv_pos_bias_table', 
                                                            shape=( (2 * self.win_size[0] - 1) * (2 * self.win_size[1] - 1), self.num_heads),
                                                            initializer=tf.initializers.Zeros(), trainable=True)

        coords_h = np.arange(self.win_size[0]) # Window partition initial coordinates height
        coords_w = np.arange(self.win_size[1]) # Window partition initial coordinates width

        coords = np.stack(np.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flat = coords.reshape(2, -1)

        rltv_coords = coords_flat[:, :, None] - coords_flat[:, None, :] # taking relative coordinattes
        rltv_coords = rltv_coords.transpose([1, 2, 0])
        rltv_coords[:, :, 0] += self.win_size[0] - 1
        rltv_coords[:, :, 1] += self.win_size[1] - 1
        rltv_coords[:, :, 0] *= 2 * self.win_size[1] - 1

        rltv_position_index = rltv_coords.sum(-1).astype(np.int64)
        self.rltv_position_index = tf.Variable(initial_value=tf.convert_to_tensor( rltv_position_index), trainable=False,name=f'{self.pre_name}/win_attn/rel_pos_index')
        self.built = True


    def call(self, x, mask=None):# implementation of the attention mechanism
        B, N, C = x.get_shape().as_list()
        qkv = tf.transpose(tf.reshape(self.qkv(x), 
                            shape=[-1, N, 3, self.num_heads, C // self.num_heads]), perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ tf.transpose(k, perm=[0, 1, 3, 2]))
        rltv_position_bias = tf.gather(self.rltv_position_bias_table, tf.reshape(self.rltv_position_index, shape=[-1]))
        rltv_position_bias = tf.reshape(rltv_position_bias, 
                                    shape=[self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1])
        rltv_position_bias = tf.transpose(
                                    rltv_position_bias, perm=[2, 0, 1])
        attn = attn + tf.expand_dims(rltv_position_bias, axis=0)

        if mask is not None:
            nW = mask.get_shape()[0]  # tf.shape(mask)[0]
            attn = tf.reshape(attn, shape=[-1, nW, self.num_heads, N, N]) + tf.cast(
                tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0), attn.dtype)
            attn = tf.reshape(attn, shape=[-1, self.num_heads, N, N])
            attn = tf.nn.softmax(attn, axis=-1)
        else:
            attn = tf.nn.softmax(attn, axis=-1)

        attn = self.attn_drop(attn)

        x = tf.transpose((attn @ v), perm=[0, 2, 1, 3])
        x = tf.reshape(x, shape=[-1, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# randomly drops some dataflow paths
class DropPath(tf.keras.layers.Layer):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        return drop_path(x, self.drop_prob, training)

# the main Swintransformer block.
# major differnce from ViT:
# not only windowed attention is applied but the windows are then shifted so that relative positional information 
# between the data that was not initially in the same window, is retained.
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, num_heads, win_size=7, shift_size=0, mlp_ratio=4., pre_name='',
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path_prob=0., regularizer=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-window_size"
        self.pre_name=pre_name

        self.norm1 = LayerNormalization(epsilon=1e-5,name=f'{self.pre_name}/layer_norm1')
        self.attn = winAttention(dim, win_size=(self.win_size, self.win_size), num_heads=num_heads,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,regularizer=regularizer)
        self.drop_path = DropPath( drop_path_prob if drop_path_prob > 0. else 0.)
        
        self.norm2 = LayerNormalization(epsilon=1e-5,name=f'{self.pre_name}/layer_norm2')
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_units=dim, hidden_units=mlp_hidden_dim,
                       drop=drop,regularizer=regularizer)

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = np.zeros([1, H, W, 1])
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            img_mask = tf.convert_to_tensor(img_mask)
            mask_wins = win_partition(img_mask, self.win_size)
            mask_wins = tf.reshape(
                mask_wins, shape=[-1, self.win_size * self.win_size])
            attn_mask = tf.expand_dims(
                mask_wins, axis=1) - tf.expand_dims(mask_wins, axis=2)
            attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
            self.attn_mask = tf.Variable(
                initial_value=attn_mask, trainable=False,name=f'{self.pre_name}/attn_mask')
        else:
            self.attn_mask = None

        self.built = True

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, shape=[-1, H, W, C])

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x

        # partition windows
        x_wins = win_partition(shifted_x, self.win_size)
        x_wins = tf.reshape(
            x_wins, shape=[-1, self.win_size * self.win_size, C])

        # W-MSA/SW-MSA
        attn_wins = self.attn(x_wins, mask=self.attn_mask)

        # merge wins
        attn_wins = tf.reshape(
            attn_wins, shape=[-1, self.win_size, self.win_size, C])
        shifted_x = win_reverse(attn_wins, self.win_size, H, W, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(shifted_x, shift=[
                        self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = tf.reshape(x, shape=[-1, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

#reshapes each layer output into different patches and regroups them and then passed through a dense loayer to reduce the last dimension by2
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_resolution, dim, regularizer=None, pre_name=''):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = Dense(2 * dim, use_bias=False,kernel_regularizer= regularizer,name=f'{pre_name}/downsample/reduction')
        self.norm = LayerNormalization(epsilon=1e-5,name=f'{pre_name}/downsample/layer_norm')

    def call(self, x):
        H, W = self.input_resolution
        B, L, C = x.get_shape().as_list()
        assert L == H * W, "input has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even values."

        x = tf.reshape(x, shape=[-1, H, W, C])

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = tf.reshape(x, shape=[-1, (H // 2) * (W // 2), 4 * C])

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(tf.keras.layers.Layer):
    def __init__(self, dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path_prob=0.,  downsample=None, regularizer=None,pre_name=''):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth


        # build blocks
        self.blocks = tf.keras.Sequential([SwinTransformerBlock(
                                           dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, win_size=win_size,
                                           shift_size=0 if (i % 2 == 0) else win_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, list) else drop_path_prob,
                                           regularizer=regularizer,
                                           pre_name=f'{pre_name}/swin_blocks{i}') for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, regularizer= regularizer, pre_name=pre_name)
        else:
            self.downsample = None

    def call(self, x):
        x = self.blocks(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(tf.keras.layers.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(4, 4), in_chans=3, embed_dim=96):
        super().__init__(name='patch_embed')

        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size)
        self.norm = LayerNormalization(epsilon=1e-5,name="layer_norm")


    def call(self, x):
        B, H, W, C = x.get_shape().as_list()
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input_size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = tf.reshape(x, shape=[-1, (H // self.patch_size[0]) * (W // self.patch_size[0]), self.embed_dim]) # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x