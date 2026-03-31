import torch
from CleanCodeRelease.architectures.SLDDLevel import SLDDLevel
from torch import nn


class FinalLayer:
    def __init__(self,
                 num_classes: int,
                 n_features: int,
                 dropout: float = 0.2,
                 config: dict = None,
                 arch: str = None):
        super().__init__()
        self.arch = arch
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # has to be called adaptive_maxpool
        # since resnet inherits from FinalLayer and has
        # maxpool as attribute already defined
        self.adaptive_maxpool = torch.nn.AdaptiveMaxPool2d((1, 1))

        self.linear = nn.Linear(n_features,
                                num_classes)
        self.featureDropout = torch.nn.Dropout(dropout)
        self.selection = None

    @property
    def is_vit(self) -> bool:
        """Check if architecture is a Vision Transformer (DINO-based)."""
        return self.arch is not None and "dino" in self.arch.lower()

    def transform_output(self,
                         feature_maps,
                         with_feature_maps,
                         with_final_features,
                         with_masks: bool = False,
                         feat_vec=None,
                         mask=None,
                         with_feat_vec: bool = False,
                         pre_feat_maps=None):
        # ViT/DINO: feat_vec already contains pooled features
        # CNN/ResNet: needs to pool from feature_maps
        if self.is_vit:
            x = feat_vec
        else:
            x = self.avgpool(feature_maps)

        # Apply feature selection for sparse classification
        if self.selection is not None:
            x = x[:, self.selection]

        # Apply selection to feature_maps for return (evaluation expects selected features)
        selected_feature_maps = feature_maps
        if self.selection is not None:
            selected_feature_maps = feature_maps[:, self.selection]

        pre_out = torch.flatten(x, 1)

        final_features = self.featureDropout(pre_out)
        final = self.linear(final_features)

        final = [final]

        if pre_feat_maps is not None:
            final.append(pre_feat_maps)

        if with_feat_vec:
            final.append(x)

        if with_feature_maps:
            final.append(selected_feature_maps)

        if with_final_features:
            final.append(final_features)

        if with_masks:
            # Masks are not used anymore, but kept for compatibility; this is probably just a placeholder
            final.append(mask)

        if len(final) == 1:
            final = final[0]

        return final

    def set_model_sldd(self,
                       selection,
                       weight_at_selection,
                       mean,
                       std,
                       bias=None,
                       retrain_normalisation=True,
                       dropout=0.1):
        self.selection = selection

        self.linear = SLDDLevel(selection=selection,
                                weight_at_selection=weight_at_selection,
                                mean=mean,
                                std=std,
                                bias=bias,
                                retrain_normalisation=retrain_normalisation)

        self.featureDropout = torch.nn.Dropout(dropout)
