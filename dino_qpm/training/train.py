import schedulefree
import torch
from dino_qpm.helpers.data import select_mask
from dino_qpm.training.losses.FeatureDiversityLoss import FeatureDiversityLoss, PrototypeDiversityLoss
from dino_qpm.training.losses.general import get_acc, get_l1_loss, feature_similarity_loss, get_iou_loss, \
    get_l1_weights_loss, get_l1_fv_loss
from tqdm import tqdm
from dino_qpm.training.utils import VariableLossLogPrinter, TrainingLogger, get_prototype_training_logs

from dino_qpm.training.losses.FeatureDiversityLoss import FeatureDiversityLoss
from dino_qpm.training.losses.ConservationOfSimilarity import ConservationOfFeatureSimilarity
from dino_qpm.training.optim import get_optimizer
from tqdm import trange

from dino_qpm.training.losses.FeatureGroundlingLoss import get_feature_grounding_loss
from dino_qpm.training.losses.RepresentativePrototypeLoss import get_repr_prot_loss
from dino_qpm.architectures.qpm_dino.load_model import load_model as load_backbone_model
from dino_qpm.architectures.registry import is_vision_foundation_model


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer | schedulefree.AdamWScheduleFree | schedulefree.SGDScheduleFree,
          fdl: FeatureDiversityLoss,
          epoch: int,
          config: dict,
          mode: str,
          test_fdl: FeatureDiversityLoss = None,
          fdl_avg: FeatureDiversityLoss = None,
          eps: float = 1e-6,
          alignment_info=None,
          logger: TrainingLogger = None,
          cofs: ConservationOfFeatureSimilarity = None,
          pdl: PrototypeDiversityLoss = None,
          backbone_model: torch.nn.Module = None) -> torch.nn.Module:
    """
    Train a model for one epoch.

    Args
    ---
        model (torch.nn.Module): The model to be trained
        train_loader (torch.utils.data.DataLoader): The data loader containing the training data
        optimizer (torch.optim.Optimizer): The optimizer for the model
        fdl (FeatureDiversityLoss): The Feature Diversity Loss module
        epoch (int): The current epoch
        config (dict): Configuration dictionary
        mode (str): Training mode
        test_fdl (FeatureDiversityLoss, optional): The Feature Diversity Loss module for the test set, 
            if given, the differences in the losses will be logged. Defaults to None.
        fdl_avg (FeatureDiversityLoss, optional): Average FDL module. Defaults to None.
        eps (float): Small epsilon value for numerical stability. Defaults to 1e-6.
        alignment_info: Information from prototype alignment. Defaults to None.
        logger (TrainingLogger, optional): Logger instance for CSV logging and visualization. Defaults to None.

    Returns
    ---
        torch.nn.Module: The trained model
    """
    arch_uses_vit = is_vision_foundation_model(config)

    grounding_loss_weight = config[mode].get("grounding_loss_weight", 0)
    rpl_weight = config[mode].get("rpl_weight", 0)

    if arch_uses_vit:
        ret_mask = config["model"]["masking"] == "learn_masking"

    else:
        ret_mask = False

    if ret_mask:
        iou_weight = config[mode]["iou_weight"]

    else:
        iou_weight = 0

    if arch_uses_vit:
        gamma = config[mode]["fsl"]

    else:
        gamma = 0

    if arch_uses_vit:
        l1_fm_weight = config[mode]["l1_fm_weight"]
    else:
        l1_fm_weight = 0

    if arch_uses_vit:
        l1_fv_weight = config[mode]["l1_fv_weight"]
    else:
        l1_fv_weight = 0

    if arch_uses_vit:
        l1_w_weight = config[mode]["l1_w_weight"]
    else:
        l1_w_weight = 0

    model.train()

    if "schedulefree" in config["optimizer"]:
        optimizer.train()

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    VariableLossPrinter = VariableLossLogPrinter()
    model = model.to(device)
    iterator = tqdm(enumerate(train_loader), total=len(train_loader))

    for _, batch in iterator:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            data, target, _ = batch
        else:
            raise ValueError(
                f"Unexpected batch format with type {type(batch)} and length {len(batch)}")

        if arch_uses_vit:
            x = data[0]
            masks = data[1]
            masks_on_device = masks.to(device)

        elif isinstance(data, list) and len(data) >= 2:
            # ResNet with masks (e.g., CUB2011 with segmentation masks)
            x = data[0]
            masks = data[1]
            masks_on_device = masks.to(device)

        else:
            x = data
            masks_on_device = None

        on_device = x.to(device)
        target_on_device = target.to(device)

        # --- BATCHED BACKBONE FEATURE EXTRACTION ---
        # If backbone_model is provided, x contains raw images that need feature extraction
        if backbone_model is not None:
            with torch.no_grad():
                # Run backbone on the full batch (much more efficient than per-sample)
                feat_maps, feat_vecs = backbone_model.get_feat_maps_and_vecs(
                    on_device,
                    use_norm=config["data"]["use_norm"],
                    max_layer_num=config["data"]["layer_num"]
                )
                # Get last layer features
                feat_map = feat_maps[-1]  # (batch, patches, dim)
                feat_vec = feat_vecs[-1]  # (batch, dim)

                # Concatenate feat_map and feat_vec: (batch, patches+1, dim)
                on_device = torch.cat((feat_map, feat_vec.unsqueeze(1)), dim=1)

        if arch_uses_vit:
            selected_mask = select_mask(masks_on_device,
                                        mask_type=config["model"].get("masking", None))

            output, feature_maps, final_features, model_mask = model(on_device,
                                                                     mask=selected_mask,
                                                                     with_feature_maps=True,
                                                                     with_final_features=True,
                                                                     with_masks=True)

        else:
            output, feature_maps, final_features = model(on_device,
                                                         with_feature_maps=True,
                                                         with_final_features=True)
            model_mask = None

        if config[mode].get("ce_free_epochs", 0) > 0 and epoch >= (config[mode]["epochs"] - config[mode]["ce_free_epochs"]):
            loss = torch.tensor(0.0, device=device)
        else:
            loss = torch.nn.functional.cross_entropy(output, target_on_device)

        if arch_uses_vit:
            selected_mask = select_mask(masks_on_device,
                                        mask_type=config["loss"]["mask_type"])

        else:
            selected_mask = None

        if pdl is not None:
            pdl_loss = pdl()

        fdl_loss = fdl(feature_maps=feature_maps,
                       outputs=output,
                       mask=selected_mask,
                       feat_vec=final_features)

        if cofs is not None and config["model"].get("use_prototypes", False) and cofs.gamma > eps:
            cos_loss = cofs(frozen_embeddings=on_device[:, :-1, :],
                            feature_embeddings=model.feat_embeddings,
                            proto_sim=feature_maps.reshape(feature_maps.shape[0],
                                                           feature_maps.shape[1], -1),
                            labels=target_on_device)
        else:
            cos_loss = torch.tensor(0.0, device=device)

        if fdl_avg is not None:
            fdl_avg_loss = fdl_avg(feature_maps=feature_maps,
                                   outputs=output,
                                   mask=selected_mask,
                                   feat_vec=None)

        if gamma > eps:
            fsl_loss = feature_similarity_loss(
                out_feat=feature_maps.reshape(
                    feature_maps.shape[0], feature_maps.shape[1], -1).transpose(1, 2),
                in_feat=on_device[:, :-1, :])

        # l1_fm_loss for feature map weights
        if l1_fm_weight > eps:
            l1_fm_loss = get_l1_loss(feature_maps=feature_maps,
                                     mask=selected_mask)

        if l1_fv_weight > eps:
            l1_fv_loss = get_l1_fv_loss(
                feat_vec=final_features, norm_with_max=config[mode].get("norm_with_max", False))

        if grounding_loss_weight > eps:
            grounding_loss = get_feature_grounding_loss(features=final_features,
                                                        target=target_on_device,
                                                        weight=model.linear.weight)

        if rpl_weight > eps and model.proto_layer is not None:
            rpl_loss = get_repr_prot_loss(feat_embeddings=model.feat_embeddings,
                                          prototypes=model.proto_layer.prototypes,
                                          selection=model.selection if hasattr(
                                              model, "selection") else None,
                                          similarity_method=model.proto_layer.similarity_method,
                                          gamma=model.proto_layer.rbf_gamma if hasattr(model.proto_layer, 'rbf_gamma') else 1e-3)

        if l1_w_weight > eps:
            l1_w_loss = get_l1_weights_loss(model=model)

        if iou_weight > eps and ret_mask:
            iou_loss = get_iou_loss(gt_mask=selected_mask,
                                    model_mask=model_mask)

        if test_fdl is not None:
            test_fdl_loss = test_fdl(feature_maps, target, output)
            VariableLossPrinter.log_loss(
                "Error in Losses", fdl_loss - test_fdl_loss, on_device.size(0))
            VariableLossPrinter.log_loss(
                "FactorLosses", fdl_loss / test_fdl_loss, on_device.size(0))

        total_loss = loss + fdl_loss

        if pdl is not None and abs(pdl_loss) > eps:
            total_loss += pdl_loss

        if cofs is not None and config["model"]["use_prototypes"] and abs(cos_loss) > eps:
            total_loss += cos_loss

        if fdl_avg is not None and abs(fdl_avg_loss) > eps:
            total_loss += fdl_avg_loss

        if gamma > eps:
            total_loss += gamma * fsl_loss

        if l1_fm_weight > eps:
            total_loss += l1_fm_weight * l1_fm_loss

        if l1_w_weight > eps:
            total_loss += l1_w_weight * l1_w_loss

        if l1_fv_weight > eps:
            total_loss += l1_fv_weight * l1_fv_loss

        if iou_weight > eps:
            total_loss += iou_weight * iou_loss

        if grounding_loss_weight > eps:
            total_loss += grounding_loss_weight * grounding_loss

        if rpl_weight > eps:
            total_loss += rpl_weight * rpl_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        acc = get_acc(output, target_on_device)

        VariableLossPrinter.log_loss("Acc", acc, on_device.size(0))
        VariableLossPrinter.log_loss("CE", loss.item(), on_device.size(0))

        logs = {
            f"Acc": VariableLossPrinter.losses["Acc"].avg,
            f"CE-Loss": VariableLossPrinter.losses["CE"].avg,
        }

        if abs(fdl_loss) > eps:
            VariableLossPrinter.log_loss(
                "FDL", fdl_loss.item(), on_device.size(0))
            logs.update({f"FDL": VariableLossPrinter.losses["FDL"].avg})

        if pdl is not None and abs(pdl_loss) > eps:
            VariableLossPrinter.log_loss(
                "PDL", pdl_loss.item(), on_device.size(0))

            logs["PDL"] = VariableLossPrinter.losses["PDL"].avg

        if cofs is not None and config["model"].get("use_prototypes", False) and abs(cos_loss) > eps:
            VariableLossPrinter.log_loss(
                "CoFS", cos_loss.item(), on_device.size(0))

            logs["CoFS"] = VariableLossPrinter.losses["CoFS"].avg

        if fdl_avg is not None and abs(fdl_avg_loss) > eps:
            VariableLossPrinter.log_loss(
                "FDL-Avg", fdl_avg_loss.item(), on_device.size(0))

            logs["FDL-Avg"] = VariableLossPrinter.losses["FDL-Avg"].avg

        if gamma > eps:
            VariableLossPrinter.log_loss(
                "FSL", fsl_loss.item(), on_device.size(0))

            logs["FSL"] = VariableLossPrinter.losses["FSL"].avg

        if l1_fm_weight > eps:
            VariableLossPrinter.log_loss(
                "L1-FM", l1_fm_loss.item(), on_device.size(0))

            logs["L1-FM"] = VariableLossPrinter.losses["L1-FM"].avg

        if l1_w_weight > eps:
            VariableLossPrinter.log_loss(
                "L1-W", l1_w_loss.item(), on_device.size(0))

            logs["L1-W"] = VariableLossPrinter.losses["L1-W"].avg

        if l1_fv_weight > eps:
            VariableLossPrinter.log_loss(
                "L1-FV", l1_fv_loss.item(), on_device.size(0))

            logs["L1-FV"] = VariableLossPrinter.losses["L1-FV"].avg

        if iou_weight > eps and ret_mask:
            VariableLossPrinter.log_loss(
                "IoU", iou_loss.item(), on_device.size(0))

            logs["IoU"] = VariableLossPrinter.losses["IoU"].avg

        if grounding_loss_weight > eps:
            VariableLossPrinter.log_loss(
                "Grounding", grounding_loss.item(), on_device.size(0))

            logs["Grounding"] = VariableLossPrinter.losses["Grounding"].avg

        if rpl_weight > eps:
            VariableLossPrinter.log_loss(
                "RPL", rpl_loss.item(), on_device.size(0))

            logs["RPL"] = VariableLossPrinter.losses["RPL"].avg

        VariableLossPrinter.log_loss(
            "Total", total_loss.item(), on_device.size(0))

        logs.update({f"Total-Loss": VariableLossPrinter.losses["Total"].avg})

        iterator.set_description(
            f"Train Epoch:{epoch} | {VariableLossPrinter.get_loss_string()}")

    # Add prototype-specific logs
    prototype_logs = get_prototype_training_logs(
        model, alignment_info, epoch, mode, config)
    logs.update(prototype_logs)

    # Log all training metrics using the logger
    if logger:
        logger.log(logs, epoch, "train")

    return model


def test(model: torch.nn.Module,
         test_loader: torch.utils.data.DataLoader,
         epoch: int,
         mode: str,
         config: dict,
         optimizer,
         alignment_info=None,
         logger: TrainingLogger = None,
         backbone_model: torch.nn.Module = None) -> tuple[float, float]:
    """
    Evaluate a model on a test set.

    Args
    ---
        model (torch.nn.Module): The model to be evaluated
        test_loader (torch.utils.data.DataLoader): The data loader containing the test data
        epoch (int): The current epoch

    Returns
    ---
        None
    """
    model.eval()

    if "schedulefree" in config["optimizer"]:
        optimizer.eval()

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(device)
    VariableLossPrinter = VariableLossLogPrinter()
    iterator = tqdm(enumerate(test_loader), total=len(test_loader))

    for _, batch in iterator:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            data, target = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            data, target, _ = batch
        else:
            raise ValueError(
                f"Unexpected batch format with type {type(batch)} and length {len(batch)}")

        if is_vision_foundation_model(config):
            x = data[0]
            mask = data[1]
            masks_on_device = mask.to(device)

        elif isinstance(data, list) and len(data) >= 2:
            # ResNet with masks (e.g., CUB2011 with segmentation masks)
            x = data[0]
            masks_on_device = data[1].to(device)

        else:
            x = data
            masks_on_device = None

        on_device = x.to(device)
        target_on_device = target.to(device)

        # If backbone_model is provided, x contains raw images.
        if backbone_model is not None:
            with torch.no_grad():
                feat_maps, feat_vecs = backbone_model.get_feat_maps_and_vecs(
                    on_device,
                    use_norm=config["data"]["use_norm"],
                    max_layer_num=config["data"]["layer_num"]
                )
                feat_map = feat_maps[-1]
                feat_vec = feat_vecs[-1]
                on_device = torch.cat((feat_map, feat_vec.unsqueeze(1)), dim=1)

        if is_vision_foundation_model(config):
            selected_mask = select_mask(masks_on_device,
                                        mask_type=config["model"]["masking"])

            output = model(on_device,
                           mask=selected_mask,
                           with_feature_maps=False)
        else:
            output = model(on_device,
                           with_feature_maps=False)

        loss = torch.nn.functional.cross_entropy(output, target_on_device)
        acc = get_acc(output, target_on_device)

        VariableLossPrinter.log_loss("Acc", acc, on_device.size(0))
        VariableLossPrinter.log_loss("CE", loss.item(), on_device.size(0))

        iterator.set_description(
            f"Test Epoch:{epoch} | {VariableLossPrinter.get_loss_string()}")

    logs = {
        f"Acc": VariableLossPrinter.losses["Acc"].avg,
        f"CE-Loss": VariableLossPrinter.losses["CE"].avg
    }

    if config["model"].get("use_prototypes", False):
        # Add prototype-specific logs for test phase
        prototype_logs = get_prototype_training_logs(
            model, alignment_info, epoch, mode, config)
        logs.update(prototype_logs)

    # Log all test metrics using the logger
    if logger:
        logger.log(logs, epoch, "test")

    return VariableLossPrinter.losses["Acc"].avg, VariableLossPrinter.losses["CE"].avg


def train_n_epochs(model,
                   beta,
                   optimization_schedule,
                   train_loader,
                   test_loader,
                   config,
                   mode: str,
                   beta_avg: float = 0.0,
                   log_dir: str = None,
                   backbone_model: torch.nn.Module = None):
    """
    Train a model for multiple epochs.

    Args:
        log_dir (str, optional): Directory to save training logs to CSV files and visualizations. 
                               If None, no training logging to files is performed.
        backbone_model (torch.nn.Module, optional): Backbone model for on-the-fly feature extraction.
                               If provided, raw images will be processed through this model in batches.
    """
    # Initialize the logger
    logger = TrainingLogger(log_dir, mode) if log_dir else None
    use_prototypes = config["model"].get("use_prototypes", False)

    optimizer, schedule, epochs = get_optimizer(model=model,
                                                schedulingClass=optimization_schedule,
                                                mode=mode,
                                                config=config)
    pdl_weight = config[mode].get("pdl", 0)
    pdl = PrototypeDiversityLoss(pdl_weight,
                                 model.proto_layer.prototypes) if use_prototypes and pdl_weight > 0 else None

    fdl = FeatureDiversityLoss(beta,
                               model.linear)

    if use_prototypes:
        cofs = ConservationOfFeatureSimilarity(
            k=config[mode].get("cofs_k", 100),
            per_prototype=config[mode].get("per_prototype", False),
            gamma=config[mode].get("cofs_weight", 1),
            similarity_method=model.proto_layer.similarity_method if hasattr(
                model, 'proto_layer') and model.proto_layer is not None else 'cosine',
            rbf_gamma=model.proto_layer.rbf_gamma if hasattr(model, 'proto_layer') and model.proto_layer is not None and hasattr(model.proto_layer, 'rbf_gamma') else 1e-3)

    else:
        cofs = None

    if not use_prototypes:
        fdl_avg = FeatureDiversityLoss(beta_avg,
                                       model.linear)

    else:
        fdl_avg = None

    # Initialize prototype layer if needed
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Calculate maximum allowed epochs based on percentage
    max_epoch_percentage = config.get('model', {}).get(
        'proto_max_epoch_percentage', 1.0)
    if max_epoch_percentage < 1.0:
        print(
            f"Warning: proto_max_epoch_percentage ({max_epoch_percentage}) is below 1.0, using 1.0")
        max_epoch_percentage = 1.0

    # Get original epochs from config based on mode
    if mode == 'dense':
        original_epochs = config.get('dense', {}).get('epochs', epochs)
    elif mode == 'finetune':
        original_epochs = config.get('finetune', {}).get('epochs', epochs)
    else:
        original_epochs = epochs

    max_allowed_epochs = int(epochs * max_epoch_percentage)

    print(
        f"Training for {original_epochs} epochs (max allowed: {max_allowed_epochs} epochs)")

    best_acc = 0
    best_model_state = None
    alignment_info = None

    if is_vision_foundation_model(config):
        model.set_random_noise(config[mode].get("random_noise", 0.0))

    # Use max_allowed_epochs for the actual training loop
    for epoch in trange(max_allowed_epochs):
        model = train(model,
                      train_loader,
                      optimizer,
                      fdl,
                      epoch,
                      config=config,
                      mode=mode,
                      fdl_avg=fdl_avg,
                      alignment_info=alignment_info,
                      logger=logger,
                      cofs=cofs,
                      pdl=pdl,
                      backbone_model=backbone_model)

        if schedule is not None:
            schedule.step()

        if epoch % 5 == 0 or epoch + 1 == max_allowed_epochs or epoch + 1 == original_epochs:
            acc, loss = test(model,
                             test_loader,
                             epoch,
                             mode=mode,
                             config=config,
                             optimizer=optimizer,
                             alignment_info=alignment_info,
                             logger=logger,
                             backbone_model=backbone_model)

    # Generate visualizations and close logger
    if logger:
        logger.visualize_training(save_plots=True)
        logger.close()
        print(
            f"Training completed. Logs and visualizations saved to: {log_dir}")

    return model, acc, loss
