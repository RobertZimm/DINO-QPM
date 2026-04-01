import torch


def calc_sim_matrix(mat: torch.Tensor,
                    sim_measure: str = "cos"):
    # BxN_PatchesxN_Channels
    # Calculate measure along n_channels
    if sim_measure == "cos":
        # Normalize matrix just like in the definition of the cosine similarity
        mat_norm = torch.nn.functional.normalize(mat,
                                                 p=2,
                                                 dim=2)

        # Then calculate the matrix product with its transpose as a
        # generalisation for the scalar product
        return torch.matmul(mat_norm,
                            mat_norm.transpose(1, 2))

    else:
        NotImplementedError(f"Similarity measure {sim_measure} not supported")


def get_iou_loss(gt_mask: torch.Tensor = None,
                 model_mask: torch.Tensor = None,
                 eps: float = 1e-10):
    if gt_mask is None or model_mask is None:
        return 0

    gt_mask = gt_mask.unsqueeze(1)

    intersection = torch.sum(gt_mask * model_mask, dim=(2, 3))
    union = gt_mask.sum(dim=(2, 3)) + model_mask.sum(dim=(2, 3)) - intersection

    iou = intersection / (union + eps)

    # The loss is 1 - IoU
    return (1 - iou).mean()


def get_l1_fv_loss(feat_vec: torch.Tensor, norm_with_max: bool = False, eps: float = 1e-10) -> torch.Tensor:
    if norm_with_max:
        clamped_max = torch.clamp(torch.max(feat_vec, dim=1)[0], min=eps).unsqueeze(1)
        feat_vec_norm = feat_vec / clamped_max
        return torch.mean(torch.abs(feat_vec_norm))
        
    return torch.mean(torch.abs(feat_vec))


def get_l1_weights_loss(model: torch.nn.Module) -> torch.Tensor:
    """
    Calculates the mean L1 regularization loss for a given PyTorch model.

    The mean L1 loss is the average of the absolute values of all the model's
    individual parameter elements (weights and biases).
    This encourages sparsity in the model.

    Args:
        model (nn.Module): The PyTorch model for which to calculate the mean L1 loss.

    Returns:
        torch.Tensor: A scalar tensor representing the mean L1 loss.
                      Returns a tensor with value 0.0 if the model has no parameters.
    """
    # Determine the device of the model's parameters
    # Fallback to CPU if model has no parameters yet
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")  # Default device if no parameters

    l1_abs_sum = torch.tensor(0.0, requires_grad=False, device=device)  # Sum doesn't need grad initially
    total_elements = 0

    # Check if model has parameters
    if not list(model.parameters()):
        # Return a zero tensor with requires_grad=True if no parameters
        # This helps avoid issues in loss calculation if the model is empty
        return torch.tensor(0.0, requires_grad=True, device=device)

    for param in model.parameters():
        if param.requires_grad:  # Only consider parameters that require gradients
            l1_abs_sum = l1_abs_sum + torch.abs(param).sum()
            total_elements += param.numel()  # numel() gives the total number of elements in the tensor

    if total_elements == 0:
        # This case handles if all parameters have requires_grad=False
        # or if the model somehow has parameters but none require grad.
        return torch.tensor(0.0, requires_grad=True, device=device)

    # Calculate the mean. The division will make it require gradients
    # if l1_abs_sum was derived from parameters requiring gradients.
    # Explicitly ensure requires_grad=True if l1_abs_sum was from params with grad.
    mean_l1 = l1_abs_sum / total_elements

    # If any parameter required grad, the sum (and thus mean) should also.
    # PyTorch usually handles this correctly, but we can be explicit.
    # If l1_abs_sum was built from parameters that require_grad, mean_l1 will also require_grad.
    # If we want to ensure it, and if any param had requires_grad=True:
    if any(p.requires_grad for p in model.parameters()):
        # This step is often not strictly necessary if l1_abs_sum was derived from grad-requiring params,
        # but it makes the intent clear.
        # A simpler way: if l1_abs_sum itself requires grad (which it will if params do)
        # then mean_l1 will too.
        # For safety, if we re-create, ensure requires_grad=True.
        # However, simple division should propagate requires_grad.
        pass  # mean_l1 should already have requires_grad=True if l1_abs_sum did.

    return l1_abs_sum


def get_l1_loss(feature_maps: torch.Tensor,
                mask: torch.Tensor = None):
    if mask is None:
        return torch.mean(torch.abs(feature_maps))

    return torch.mean(torch.abs(feature_maps))

    mask_expanded = mask.unsqueeze(1).float()

    # Only calculate the L1 loss for the background
    return torch.mean(torch.abs(feature_maps * (1 - mask_expanded)))


def feature_similarity_loss(in_feat: torch.Tensor,
                            out_feat: torch.Tensor,
                            sim_measure: str = "cos"):
    # BxN_PatchesxN_Channels
    sim_in = calc_sim_matrix(mat=in_feat,
                             sim_measure=sim_measure)

    sim_out = calc_sim_matrix(mat=out_feat,
                              sim_measure=sim_measure)

    # Difference between similarity of input feature maps
    # and similarity of output feature maps
    diff = sim_in - sim_out

    n_patches = sim_in.shape[1]

    # return frobenius norm of difference
    return (1 / n_patches) * torch.norm(diff)


def get_acc(outputs: torch.Tensor,
            targets: torch.Tensor):
    """
    Compute the accuracy of the model given its outputs and the targets.

    Args
    ---
        outputs (torch.Tensor): The output of the model
        targets (torch.Tensor): The correct targets

    Returns
    ---
        float: The accuracy of the model
    """
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()

    return correct / total * 100
