import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv2D, LayerNormalization, GlobalAveragePooling1D
from utils.layers import * 


# this code has been ported from pytorch to tensorflow 
# based on https://github.com/microsoft/Swin-Transformer.git

# @inproceedings{liu2021Swin,
#   title={Swin Transformer: Hierarchical Vision Transformer using Shifted wins},
#   author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
#   booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
#   year={2021}
# }

swin_params = {
    'swin_best': dict(input_size=(32, 32), win_size=4, embed_dim=128, depths=[2, 4, 4], num_heads=[32, 32, 32], patch_size=(2,2),drop=[0.1,0.0,0.1]),
    'swin_proj': dict(input_size=(32, 32), win_size=4, embed_dim=96, depths=[2, 2, 6], num_heads=[32, 32, 32],patch_size=(2,2),drop=[0.0,0.0,0.1]),
}

#Swin parameters that define the architecture using 'swin_best with best accuracy for this model'

# 1.) the window_size should be a factor of the input dimension for optimal infromation retention hence 
# win_size=4 against the original selection of win_size=7 for input_size=224

#2.) emebedded dimensions of Swin-B variant for better encoding

#3.) depth was reduced from 4 layers to 3 ; as to not overfit the low dimensional input data to a very complex model.

#4.) Patch_size was selected by (2,2) as the default (4,4) was too big for an image of 32*32

#5.) drop_out rate fro MLP was selected to be 0.1, as it gave the most optimal results


# The following is a generic implementaion of the swintransformer Model 
class SwinTransformerModel(tf.keras.Model):
    def __init__(self, model_name='swin_cifar10', fc_layer=False,
                 img_size=(32,32), patch_size=(2,2), in_chans=3, 
                 num_classes=1000, embed_dim=128, depths=[2, 4,4], 
                 num_heads=[32, 32, 32], win_size=4, mlp_ratio=4., 
                 qkv_bias=True, qk_scale=None,drop_rate=0.0, 
                 attn_drop_rate=0.0, drop_path_rate=0.1,
                 ape=False, patch_norm=True, 
                 regularizer= None,**kwargs,):

        super().__init__(name=model_name)

        self.fc_layer = fc_layer # should a final fully connected layer be implemented?
        self.num_classes = num_classes # for classification
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        # Converts to an embedding dimension for better representation
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)  

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.add_weight('absolute_pos_embed',
                                                      shape=(1, num_patches, embed_dim),
                                                      initializer=tf.initializers.Zeros())

        self.pos_drop = Dropout(drop_rate)

        # stochastic depth => picks differnt paths of propagtion with probability drop_path_rate
        dpr = [x for x in np.linspace(0., drop_path_rate, sum(depths))]

        # build n swin transformer layers for each n in "depth"
        self.basic_layers = tf.keras.Sequential([BasicLayer(
                                                dim=int(embed_dim * 2 ** i_layer),
                                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                                  patches_resolution[1] // (2 ** i_layer)),
                                                depth=depths[i_layer],
                                                num_heads=num_heads[i_layer],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop_rate, attn_drop=attn_drop_rate,
                                                drop_path_prob=dpr[sum(depths[:i_layer]):sum(
                                                    depths[:i_layer + 1])],
                                                downsample=PatchMerging if (
                                                    i_layer < self.num_layers - 1) else None,
                                                regularizer= regularizer,
                                                pre_name=f'layers{i_layer}') for i_layer in range(self.num_layers)])

        # Layer Normalization performs a normalization across the output of the previous layer
        self.norm = LayerNormalization(epsilon=1e-5, name='norm') 
        self.avgpool = GlobalAveragePooling1D()

        if self.fc_layer: #adds a final dense(fully_connected layer)
            self.head = Dense(num_classes, name='head')
        else:
            self.head = None

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        x = self.basic_layers(x)
        x = self.norm(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        return x

    def call(self, x):
        x = self.forward_features(x)
        if self.fc_layer:
            x = self.head(x)
        return x


# SwinTransformer initializer based on the swin_params dictionary
def SwinTransformer(model_name='swin_best', num_classes=1000, fc_layer=True, swin_params=swin_params,regularizer= None):
    swin_config = swin_params[model_name]
    dropout=swin_config['drop']

    net = SwinTransformerModel(
        model_name=model_name, fc_layer=fc_layer, num_classes=num_classes, 
        img_size=swin_config['input_size'], win_size=swin_config['win_size'], 
        embed_dim=swin_config['embed_dim'], depths=swin_config['depths'], 
        num_heads=swin_config['num_heads'],patch_size=swin_config['patch_size'],
        regularizer= regularizer, drop_rate=dropout[0], attn_drop_rate=dropout[1], drop_path_rate=dropout[2]
    )

    input_shape=(swin_config['input_size'][0], swin_config['input_size'][1], 3)

    net(tf.keras.Input(shape=input_shape)) # declaring the input shape to the Swin Transformer

    return net
