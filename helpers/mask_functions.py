import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import minmax_scale


# use some type of exponential decay;
# probably more reasonable, with some scaling factor in the exponent
def decay_func(current_n: int,
               min_n_clusters: int,
               a: float = 0.2,
               mode: str = "exponential") -> float:
    if mode == "linear":
        val = 1 - a * (current_n - min_n_clusters)

    elif mode == "exponential":
        val = np.exp(-a * (current_n - min_n_clusters))

    else:
        raise ValueError(f"Unknown decay function mode: {mode}. Use 'linear' or 'exponential'.")

    return val


def collect_attn(mask: np.ndarray,
                 attn: torch.Tensor,
                 current_n: int,
                 min_clusters: int,
                 a: float = 0.005) -> float:
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()

    attn = minmax_scale(attn)

    c_a = np.sum(mask * attn) / np.sum(attn)
    abs_diff = np.sum(np.abs((4 / 5) * mask - attn))

    if abs_diff == 0:
        return np.inf

    score = (c_a / abs_diff) * decay_func(current_n, min_clusters, a=a)

    return score


def gmm_score(mask, attn):
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()

    attn = minmax_scale(attn)

    gmm = GaussianMixture(n_components=2).fit(attn.reshape(-1, 1))
    att_labeled = gmm.predict(attn.reshape(-1, 1)).reshape(attn.shape)
    sorted_mean_arg = np.argsort(gmm.means_.flatten())

    attn_mask = np.zeros_like(att_labeled)
    attn_mask[att_labeled == sorted_mean_arg[1]] = 1

    collect_attn = np.sum(mask * attn) / np.sum(attn)

    abs_diff = np.logical_or(mask, attn_mask).sum()

    if abs_diff == 0:
        return np.inf

    max_score = attn.sum() / attn_mask.sum()
    score = (collect_attn / abs_diff) / max_score

    return score


def choose_segmentation(feature_segmentations: np.ndarray,
                        attn: torch.Tensor,
                        patch_tokens: torch.Tensor,
                        assignment: dict,
                        mode: str = "collect_attn") -> np.ndarray:
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu().numpy()

    n_masks = feature_segmentations.shape[0]
    scores = []

    min_n_clusters = assignment[0]

    for idx in range(n_masks):
        current_n = assignment[idx]
        mask = feature_segmentations[idx]

        if mode == "decay":
            n_elements = np.sum(mask)

            # Decay function used to tend to give bigger
            # clusters from earlier iterations more weight
            # Since it is better to have some background in the
            # mask other than missing parts of the object
            cluster_normalization = decay_func(current_n,
                                               min_n_clusters,
                                               a=0.15,
                                               mode="exponential")

            score = (1 / n_elements) * np.sum(mask * attn) * cluster_normalization

        elif mode == "collect_attn":
            score = collect_attn(mask,
                                 attn,
                                 current_n,
                                 min_n_clusters)

        elif mode == "collect_attn_recall":
            score = collect_attn(mask,
                                 attn,
                                 current_n,
                                 min_n_clusters,
                                 a=0.3)

        elif mode == "max_attn":
            score = np.sum(mask * attn)

        elif mode == "gmm_score":
            score = gmm_score(mask, attn)

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'decay' or 'IoU'.")

        scores.append(score)

    max_arg = np.argmax(scores)
    best_fit = feature_segmentations[max_arg]

    # big_5 = torch.topk(torch.Tensor(attn), k=20).indices
    # top_5_tokens = torch.Tensor(patch_tokens[big_5])
    #
    # avg_fg_token = torch.mean(top_5_tokens, dim=0)
    #
    # for idx, element in enumerate(best_fit):
    #     if element:
    #         token = torch.Tensor(patch_tokens[idx])
    #
    #         sim = torch.nn.functional.cosine_similarity(token, avg_fg_token, dim=0)
    #
    #         if sim <= 0.6:
    #             best_fit[idx] = 0

    return best_fit


def convert_labels_to_masks(labels: list[int]) -> np.ndarray:
    n_clusters = len(set(labels))
    masks = np.zeros((n_clusters, len(labels)),
                     dtype=np.uint8)

    for idx, element in enumerate(labels):
        for n in range(n_clusters):
            if element == n:
                masks[n, idx] = 1

    return masks


def reduce_colors(img: np.ndarray | torch.Tensor,
                  n_colors: int = 3) -> tuple[np.ndarray | torch.Tensor, list]:
    if img.shape[-1] != 3:
        raise ValueError("image must have 3 channels")

    # Flatten Image to make it fit for KMeans
    flat_img = img.reshape(-1, 3)

    # Initialize kmeans with flattend image
    kmeans = KMeans(n_clusters=n_colors,
                    n_init="auto").fit(flat_img)

    # Get centroids (n_colors different colors with 3 channels each)
    centroids = kmeans.cluster_centers_

    # Get labels for every pixel of the original image
    # Labels tell which centroid corresponds to each pixel
    img_labels = kmeans.labels_

    # Combine centroids and labels to get recolored image
    # And reshape it back to its original shape
    recolored_img = centroids[img_labels].reshape(img.shape)

    return recolored_img, img_labels


def custom_combined_morph(mask,
                          erosion_neighbours=8,
                          dilation_neighbours=1):
    # Erosion into dilation
    mask = custom_morph(mask,
                        neccessary_neighbours=dilation_neighbours)
    mask = custom_morph(mask,
                        neccessary_neighbours=erosion_neighbours)

    return mask


def custom_morph(mask: np.ndarray, neccessary_neighbours: int):
    """
    :param mask: A binary mask.
    :param neccessary_neighbours: The lower the value, the closer the operation is to a dilation. The higher the value, the closer the operation is to an erosion.
    :return: Mask with morphological operation applied.
    """
    padded_mask = np.pad(mask,
                         pad_width=((1, 1), (1, 1)),
                         mode="constant")

    out_mask = np.zeros_like(mask)

    for (i, j) in np.ndindex(mask.shape):
        neighbourhood = padded_mask[i:i + 3, j:j + 3]

        if neighbourhood.sum() >= neccessary_neighbours:
            out_mask[i, j] = 1

    return out_mask
