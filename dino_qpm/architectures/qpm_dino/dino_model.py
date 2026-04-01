import re
from math import sqrt

import torch
import torch.nn as nn
import yaml
from dino_qpm.architectures.FinalLayer import FinalLayer
from dino_qpm.architectures.qpm_dino.layers import MaskingLayer, BatchNorm1dPermuted, PrototypeLayer


class Dino2Div(nn.Module, FinalLayer):
    def __init__(self, config: dict,
                 num_classes: int = 200):
        self.init_vars(config=config,
                       num_classes=num_classes)

        if self.n_layers > 0:
            if self.deprecated_notation:
                self.handle_layers_dep()

            else:
                self.handle_layers()

        else:
            self.seq = None

        if self.n_layers == 0 and self.n_features != self.dino_channels:
            print(f">>> Warning: n_features {self.n_features} is not equal to dino_channels {self.dino_channels}. "
                  f"Forcing n_features to dino_channels as n_layers=0.")
            self.n_features = self.dino_channels

        if self.proto_layer is not None and self.proto_method != "pipnet":
            n_features = self.n_prototypes

        else:
            n_features = self.n_features

        # Add final layer
        FinalLayer.__init__(self,
                            num_classes=self.num_classes,
                            n_features=n_features,
                            dropout=self.dropout,
                            config=config,
                            arch=config["arch"])

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor = None,
                with_feature_maps: bool = False,
                with_final_features: bool = False,
                with_masks: bool = False,
                with_feat_vec: bool = False,
                ret_pre_feat_maps: bool = False,
                epoch: int = None):
        feat_maps, feat_vec, pre_feat_maps = self.pre_transform_forward(x,
                                                                        ret_pre_feat_maps=ret_pre_feat_maps,
                                                                        epoch=epoch)

        return self.transform_output(feat_vec=feat_vec,
                                     feature_maps=feat_maps,
                                     with_feature_maps=with_feature_maps,
                                     with_final_features=with_final_features,
                                     with_masks=with_masks,
                                     mask=mask,
                                     with_feat_vec=with_feat_vec,
                                     pre_feat_maps=pre_feat_maps)

    def pre_transform_forward(self, x: torch.Tensor, mask: torch.Tensor = None, ret_pre_feat_maps: bool = False, epoch: int = None):
        if mask is not None and len(mask.shape) == 2:
            # Extend mask to match the batch size
            mask = mask.unsqueeze(0)

        feat_maps, feat_vec = self.run_mlp(x, mask)

        self.feat_embeddings = feat_maps.reshape(feat_maps.shape[0],
                                                 feat_maps.shape[1], -1).permute(0, 2, 1)

        # At this point feat_vec and feat_maps exist
        # feat_vec: B x #Channels
        # feat_maps: B x #Channels x H x W
        # Transform feat_vec and feat_maps
        # using the Prototype Layer
        # feat_maps become similarity measure with to feature map
        # corresponding prototype
        # feat_vec entry of prototype becomes some sort of pooling (max/average)
        # over that feature map
        if self.proto_layer is not None:
            if self.proto_method == "pipnet":
                # Apply softmax over embedding dimension D (prototype dimension) of feat_maps
                # to get normalized similarity scores with prototypes
                feat_maps = feat_maps.softmax(dim=1)

                # feat_vec is the max_pooling over feat_maps
                feat_maps_reshaped = feat_maps.reshape(feat_maps.shape[0],
                                                       feat_maps.shape[1], -1)

                feat_vec = feat_maps_reshaped.max(dim=-1)[0]

            else:
                mode = "dense" if self.selection is None else "finetune"

                if ret_pre_feat_maps:
                    pre_feat_maps = feat_maps.clone()

                else:
                    pre_feat_maps = None

                # if random noise is specified, add it to the feature maps
                # and maintain original magnitude; sort of simulate
                # a very small rotation in feature space
                if self.random_noise > 0.0:
                    noise = torch.randn_like(
                        self.feat_embeddings) * self.random_noise

                    perturbed = self.feat_embeddings + noise

                    # Maintain original magnitude
                    self.feat_embeddings = perturbed * \
                        (torch.norm(self.feat_embeddings, dim=-1, keepdim=True) /
                         torch.norm(perturbed, dim=-1, keepdim=True))

                    feat_maps = self.feat_embeddings.permute(
                        0, 2, 1).reshape(feat_maps.shape)

                # Use a high epoch value during inference to ensure multi-embedding logic activates
                # if prototypes have been expanded to multi-embedding mode
                inference_epoch = epoch if epoch is not None else 9999

                feat_vec, feat_maps = self.proto_layer(
                    feat_vec, feat_maps, mode=mode, epoch=inference_epoch)

        else:
            pre_feat_maps = None

        if self.scale_feat_vec:
            feat_vec = feat_vec * self.scale

        if self.relu_after_scaling:
            feat_vec = torch.relu(feat_vec)

        # if self.learn_masking:
        #     thresholds = feat_maps[:, -1].unsqueeze(1)
        #     feat_maps = feat_maps[:, :-1]

        #     mask = self.masking_layer(feat_maps,
        #                               thresholds)

        #     feat_maps = feat_maps * mask

        # elif mask is not None:
        #     mask_expanded = mask.unsqueeze(1)
        #     feat_maps = feat_maps * mask_expanded.float()

        return feat_maps, feat_vec, pre_feat_maps

    def run_mlp(self, x: torch.Tensor,
                mask: torch.Tensor = None):
        # Split behaviour dependent on architecture
        # and feature vector type
        if (self.arch_type == "normal" and (self.feat_vec_type == "normal" or
                                            self.feat_vec_type == "mean_avg_pooling" or
                                            self.feat_vec_type == "avg_pooling" or
                                            self.feat_vec_type == "max_pooling")):
            if self.seq is not None:
                if self.residual:
                    x = self.seq(x) + x

                else:
                    x = self.seq(x)

            elif self.residual:
                raise ValueError(
                    "Residual connection specified but no layers defined.")

            # Feature map: B x #Tokens x #Channels
            # Feature vector: B x #Channels
            feat_map_flat = x[:, :-1, :]

            if self.feat_vec_type == "normal":
                feat_vec = x[:, -1, :]

            elif self.feat_vec_type == "avg_pooling" or \
                    self.feat_vec_type == "max_pooling":
                feat_vec = torch.mean(feat_map_flat, dim=1)

            elif self.feat_vec_type == "mean_avg_pooling":
                feat_vec_orig = x[:, -1, :]
                feat_vec_patches = torch.mean(feat_map_flat, dim=1)
                feat_vec = (feat_vec_orig + feat_vec_patches) / 2

            else:
                raise NotImplementedError(
                    f"feat_vec_type {self.feat_vec_type} not supported")

            # Batch size: number of samples
            batch_size = feat_map_flat.shape[0]

            # Quadratic size of feature map
            map_size = int(sqrt(feat_map_flat.shape[1]))

            # Reshape feature map to have quadratic shape
            # and not be flat
            if self.mlp_arch == "linear":
                feat_map_flat = feat_map_flat.transpose(1, 2)
                feat_maps = feat_map_flat.reshape(batch_size,
                                                  self.n_features,
                                                  map_size,
                                                  map_size)

            elif self.mlp_arch == "transformer":
                feat_map_flat = feat_map_flat.transpose(1, 2)
                feat_maps = feat_map_flat.reshape(batch_size,
                                                  self.dino_channels,
                                                  map_size,
                                                  map_size)

            else:
                raise NotImplementedError(
                    f"model_arch {self.mlp_arch} not supported")

        elif self.arch_type == "concat" and (self.feat_vec_type == "avg_pooling" or
                                             self.feat_vec_type == "max_pooling"):
            # Concat doubles number of feature channels
            output_token = x[:, -1, :]
            feat_map_flat = x[:, :-1, :]

            if mask is not None and self.use_pre_concat_mask:
                flat_mask = mask.reshape(mask.shape[0], -1).float()
                feat_map_flat = feat_map_flat * flat_mask.unsqueeze(2)

            feat_rep = output_token.unsqueeze(
                1).repeat(1, feat_map_flat.shape[1], 1)

            feat_map_flat_concat = torch.concatenate([feat_map_flat,
                                                      feat_rep],
                                                     dim=2)

            if mask is not None and self.use_post_concat_mask:
                flat_mask = mask.reshape(mask.shape[0], -1).float()
                feat_map_flat_concat = feat_map_flat_concat * \
                    flat_mask.unsqueeze(2)

            # Apply neural network to input
            if self.residual:
                feat_map_flat_concat = self.seq(
                    feat_map_flat_concat) + feat_map_flat_concat
            else:
                feat_map_flat_concat = self.seq(feat_map_flat_concat)

            # Batch size: number of samples
            batch_size = feat_map_flat_concat.shape[0]

            # Quadratic size of feature map
            map_size = int(sqrt(feat_map_flat_concat.shape[1]))

            n_channels = feat_map_flat_concat.shape[2]

            if self.mlp_arch == "linear":
                # Reshape feature map to have quadratic shape
                # and not be flat
                feat_map_flat_concat = feat_map_flat_concat.transpose(1, 2)
                feat_maps = feat_map_flat_concat.reshape(batch_size,
                                                         n_channels,
                                                         map_size,
                                                         map_size)

            elif self.mlp_arch == "transformer":
                feat_map_flat_concat = feat_map_flat_concat.transpose(1, 2)
                feat_maps = feat_map_flat_concat.reshape(batch_size,
                                                         self.dino_channels,
                                                         map_size,
                                                         map_size)

            if self.feat_vec_type == "avg_pooling":
                feat_vec = self.avgpool(feat_maps)

            elif self.feat_vec_type == "max_pooling":
                feat_vec = self.maxpool(feat_maps)

            else:
                raise NotImplementedError(
                    f"feat_vec_type {feat_vec} not supported")

        else:
            raise NotImplementedError(f"Combination of arch_type={self.arch_type} and "
                                      f"feat_vec_type={self.feat_vec_type} not supported")

        return feat_maps, feat_vec

    def get_feat_vec_embeddings(self, x: torch.Tensor, epoch: int = None):
        if self.proto_layer is not None:
            similarity_maps, feat_vec, feat_maps = self.pre_transform_forward(
                x, ret_pre_feat_maps=True, epoch=epoch)
        else:
            feat_maps, feat_vec = self.run_mlp(x)

        return feat_maps, feat_vec

    def handle_layers(self):
        if self.mlp_arch == "linear":
            self.seq = nn.Sequential()

            for num in range(self.n_layers):
                self.add_lin(pos=num)

                if self.use_batch_norm:
                    self.add_batch_norm()

                self.add_activation()

                if self.use_dropout and num < self.n_layers - 1:
                    self.add_dropout()

        elif self.mlp_arch == "transformer":
            raise NotImplementedError

        else:
            raise ValueError(
                f"Unknown model architecture: {self.mlp_arch}")

    def handle_layers_dep(self):
        if self.mlp_arch == "linear":
            self.seq = nn.Sequential()

            self.n_layers = len(self.layer_strings)
            for idx, layer_string, in enumerate(self.layer_strings):
                self.add_lin_from_str(layer_string,
                                      idx)

        elif self.mlp_arch == "transformer":
            self.seq = nn.TransformerEncoderLayer(d_model=self.dino_channels,
                                                  nhead=8,
                                                  dim_feedforward=2048,
                                                  batch_first=True, )

    def add_lin(self, pos: int):
        if pos == 0 and self.n_layers > 1:
            layer = torch.nn.Linear(self.dino_channels, self.hidden_size)
            out_channels = self.hidden_size

        elif pos == 0 and self.n_layers == 1:
            layer = torch.nn.Linear(self.dino_channels, self.n_features)
            out_channels = self.n_features

        elif pos == self.n_layers - 1:
            layer = torch.nn.Linear(self.hidden_size, self.n_features)
            out_channels = self.n_features

        else:
            layer = torch.nn.Linear(self.hidden_size, self.hidden_size)
            out_channels = self.hidden_size

        self.last_out_channels = out_channels
        self.seq.append(layer)

    def add_activation(self):
        if self.activation_func == "relu":
            self.seq.append(nn.ReLU())

        elif self.activation_func == "sigmoid":
            self.seq.append(nn.Sigmoid())

        elif self.activation_func == "gelu":
            self.seq.append(nn.GELU())

        elif self.activation_func == "leaky_relu":
            self.seq.append(nn.LeakyReLU())

        elif self.activation_func == "tanh":
            self.seq.append(nn.Tanh())

        else:
            raise NotImplementedError(
                f"activation_func {self.activation_func} not supported")

    def add_batch_norm(self):
        self.seq.append(BatchNorm1dPermuted(self.last_out_channels))

    def add_dropout(self):
        self.seq.append(nn.Dropout(p=self.dropout))

    def add_lin_from_str(self, layer: str,
                         idx: int):
        layer_type = match_pattern(r'^[^(]+',
                                   layer)

        if layer_type == "linear":
            in_channels, out_channels = extract_in_out(layer)

            if in_channels == "feat_channels" or idx == 0:
                in_channels = self.dino_channels

            elif in_channels == "hidden_size":
                in_channels = self.hidden_size

            else:
                in_channels = int(in_channels)

            if out_channels == "n_features":
                out_channels = self.n_features

            elif out_channels == "hidden_size":
                out_channels = self.hidden_size

            else:
                out_channels = int(out_channels)

            if idx == self.n_layers - 1 and self.learn_masking:
                out_channels += 1

            self.seq.append(nn.Linear(in_channels,
                                      out_channels))

            self.last_out_channels = out_channels

        elif layer_type == "batch_norm":
            self.add_batch_norm()

        elif layer_type == "dropout":
            self.add_dropout()

        elif layer_type in ["relu", "gelu", "leaky_relu", "sigmoid"]:
            self.activation_func = layer_type
            self.add_activation()

        elif layer_type is None:
            pass

        else:
            raise NotImplementedError(f"Layer type {layer_type} not supported")

    def set_random_noise(self, noise: float):
        self.random_noise = noise

    def init_vars(self, config: dict,
                  num_classes: int):
        if "layers" in config["model"].keys():
            self.deprecated_notation = True

        else:
            self.deprecated_notation = False

        super(Dino2Div, self).__init__()
        self.config = config

        # Initialize the model with the given configuration
        self.n_features = config["model"]["n_features"]

        if config.get("model", {}).get("hidden_size") is not None:
            self.hidden_size = config["model"]["hidden_size"]
        else:
            print(">>> Using default hidden size of 2048.")
            self.hidden_size = 2048

        self.num_classes = num_classes
        self.mlp_arch = config["model"]["arch"]
        self.backbone_arch = config["model_type"]

        self.feat_vec_type = config["model"]["feat_vec_type"]
        self.arch_type = config["model"]["arch_type"]
        self.residual = config["model"]["residual"]
        self.use_pre_concat_mask = config["model"]["use_pre_concat_mask"]
        self.use_post_concat_mask = config["model"]["use_post_concat_mask"]
        self.random_noise = 0.0
        self.scale_feat_vec = config["model"].get("scale_feat_vec", False)
        self.relu_after_scaling = config["model"].get(
            "relu_after_scaling", False)

        self.learn_masking = config["model"]["masking"] == "learn_masking"
        self.proto_method = config["model"].get("proto_method", "other")

        if config["model"]["masking"] == "none":
            config["model"]["masking"] = None

        elif config["model"]["masking"] is not None:
            raise NotImplementedError(
                "Masking not supported anymore, please use masking=null. ")

        self.dropout = config["dense"]["dropout"]

        model_type = config["model_type"]

        if self.learn_masking:
            self.masking_layer = MaskingLayer(mode="soft")

        if "small" in model_type:
            self.dino_channels = 384

        elif "base" in model_type:
            self.dino_channels = 768

        elif "large" in model_type:
            self.dino_channels = 1024

        elif "giant" in model_type or "sinder" in model_type:
            self.dino_channels = 1536

        else:
            raise ValueError(f">>> Unknown model type: {model_type}")

        if self.arch_type == "concat":
            self.dino_channels *= 2

        if self.deprecated_notation:
            self.layer_strings = config["model"]["layers"]
            self.last_out_channels = self.dino_channels

        else:
            self.n_layers = config["model"]["n_layers"]
            self.use_batch_norm = config["model"]["use_batch_norm"]
            self.use_dropout = config["model"]["use_dropout"]
            self.activation_func = config["model"]["activation"]

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = torch.nn.AdaptiveMaxPool2d((1, 1))

        if config["model"].get("n_prototypes", 0) > 0 and self.proto_method != "pipnet":
            self.n_prototypes = config["model"]["n_prototypes"]

            # Initialize the prototype layer with proper parameters
            self.proto_layer = PrototypeLayer(
                n_prototypes=self.n_prototypes,
                embed_dim=self.n_features,
                config=config
            )

        else:
            self.proto_layer = None

        if self.scale_feat_vec:
            if self.proto_layer is not None:
                dim = self.n_prototypes
            else:
                dim = self.n_features

            self.scale = torch.nn.Parameter(torch.ones(1, dim))


def match_pattern(pattern: str, text: str):
    if "(" in text:
        # Perform regex search
        match = re.search(pattern, text)

        # Extract the result if a match is found
        if match:
            result = match.group(0)

            return result

        else:
            raise ValueError(f"No pattern found in text: {text}")

    else:
        # If no parentheses, just return the text
        return text


def extract_in_out(text: str):
    # Regex pattern to capture both arguments (works for both strings and numbers)
    pattern = r'\(\s*([^\s,]+)\s*,\s*([^\s,)]+)\s*\)'

    # Perform regex search
    match = re.search(pattern, text)

    # Extract the results if a match is found
    if match:
        first_arg = match.group(1)
        second_arg = match.group(2)

        return first_arg, second_arg
    else:
        raise ValueError(f"No pattern found in text: {text}")


if __name__ == '__main__':
    config_path = "../../configs/dinov2.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = Dino2Div(config=config)
