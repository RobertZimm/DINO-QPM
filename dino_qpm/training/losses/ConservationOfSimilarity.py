import torch
from dino_qpm.architectures.qpm_dino.similarity_functions import (
    compute_cosine_similarity,
    compute_log_l2_similarity,
    compute_rbf_similarity
)


class ConservationOfFeatureSimilarity(torch.nn.Module):
    def __init__(self, per_prototype: bool = False, k: int = 5, gamma: float = 1.0,
                 similarity_method: str = 'cosine', rbf_gamma: float = 1e-3):
        super().__init__()
        self.per_prototype = per_prototype
        self.k = k
        self.gamma = gamma  # Loss weight
        self.similarity_method = similarity_method
        self.rbf_gamma = rbf_gamma  # RBF kernel parameter

    def forward(self, frozen_embeddings: torch.Tensor,
                feature_embeddings: torch.Tensor,
                proto_sim: torch.Tensor,
                labels: torch.Tensor):
        '''
        frozen_embeddings: (B, num_patches, D) - embeddings from frozen model
        feature_embeddings: (B, num_patches, N_f) - embeddings from current model
        proto_sim: (B, num_prototypes, num_patches) - class prototypes
        labels: (B) - ground truth labels
        '''
        self.num_prototypes = proto_sim.shape[1]

        ranking = self.create_ranking(
            frozen_embeddings, feature_embeddings, labels, proto_sim)

        self.selection = self.find_selection(
            ranking)

        return self.calc_loss()

    def create_ranking(self, frozen_embeddings: torch.Tensor,
                       feature_embeddings: torch.Tensor,
                       labels: torch.Tensor,
                       proto_sim: torch.Tensor):
        _, num_patches, _ = feature_embeddings.shape

        self.feat_sim = self.calc_self_sim(feature_embeddings)
        self.frozen_sim = self.calc_self_sim(frozen_embeddings)

        max_proto, self.max_proto_indices = self.approx_calc_max_proto_sim(
            proto_sim)

        are_same_class = self.extend_labels(labels, num_patches)

        # Initialize ranking with zeros
        ranking = torch.zeros_like(self.feat_sim)

        # Create upper triangle mask (excluding diagonal)
        upper_triangle_mask = torch.triu(
            torch.ones_like(ranking), diagonal=1).bool()

        # Extract upper triangle values for each component to avoid full matrix operations
        feat_sim_upper = self.feat_sim[upper_triangle_mask]
        frozen_sim_upper = self.frozen_sim[upper_triangle_mask]
        are_same_class_upper = are_same_class[upper_triangle_mask]
        max_proto_upper = max_proto[upper_triangle_mask]

        # Only calculate ranking for upper triangle to avoid duplicate calculations
        ranking[upper_triangle_mask] = (feat_sim_upper - frozen_sim_upper) * \
            (1 - are_same_class_upper.float()) * max_proto_upper

        return ranking

    def approx_calc_max_proto_sim(self, proto_sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        proto_sim is of shape (B, num_prototypes, N)
        returns a matrix of shape (B*N, B*N) where each entry is the maximum
        product of similarity to a certain prototype between the two patches,
        also returns another matrix of the same shape containing which prototype 
        was responsible for the maximum similarity
        '''
        batch_size, num_prototypes, num_patches = proto_sim.shape

        # Reshape proto_sim to (B*N, num_prototypes) for easier computation
        proto_sim_reshaped = proto_sim.transpose(
            1, 2).reshape(batch_size * num_patches, num_prototypes)

        max_proto_sim, max_proto_indices = torch.max(proto_sim_reshaped, dim=1)

        is_same_proto = max_proto_indices.unsqueeze(
            0) == max_proto_indices.unsqueeze(1)

        # Only use upper triangle to avoid duplicate calculations
        is_same_proto = is_same_proto & torch.triu(
            torch.ones_like(is_same_proto), diagonal=1).bool()

        # Iterate over is_same_proto and use max_proto_sim(i) * max_proto_sim(j) if is_same_proto[i,j] else 0
        max_proto_sim_matrix = torch.zeros(
            (batch_size * num_patches, batch_size * num_patches), device=proto_sim.device)

        max_proto_sim_matrix[is_same_proto] = (max_proto_sim.unsqueeze(
            0) * max_proto_sim.unsqueeze(1))[is_same_proto]

        # Get indices where max_proto_sim_matrix is non-zero
        max_proto_indices = torch.nonzero(max_proto_sim_matrix, as_tuple=False)

        return max_proto_sim_matrix, max_proto_indices

    def calc_max_proto_sim(self, proto_sim: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        proto_sim is of shape (B, num_prototypes, N)
        returns a matrix of shape (B*N, B*N) where each entry is the maximum
        product of similarity to a certain prototype between the two patches,
        also returns another matrix of the same shape containing which prototype 
        was responsible for the maximum similarity
        '''
        batch_size, num_prototypes, num_patches = proto_sim.shape

        # Reshape proto_sim to (B*N, num_prototypes) for easier computation
        proto_sim_reshaped = proto_sim.transpose(
            1, 2).reshape(batch_size * num_patches, num_prototypes)

        # Compute pairwise products for each prototype
        # proto_sim_reshaped[i, p] * proto_sim_reshaped[j, p] for all i, j, p
        products = proto_sim_reshaped.unsqueeze(
            1) * proto_sim_reshaped.unsqueeze(0)
        # Shape: (B*N, B*N, num_prototypes)

        # Find maximum product across prototypes for each pair of patches
        max_proto_sim, max_proto_indices = torch.max(products, dim=2)
        # Shape: (B*N, B*N) for both matrices

        return max_proto_sim, max_proto_indices

    @staticmethod
    def extend_labels(labels: torch.Tensor, num_patches: int) -> torch.Tensor:
        '''
        labels is of shape (B) containing class labels for each sample in the batch
        returns extended labels of shape (B*N, B*N) where N is number of patches
        extended labels is binary equal to 1 if both patches have the same label and
        0 otherwise
        '''
        ext_labels = labels.unsqueeze(1).expand(-1, num_patches).reshape(-1)

        ret_labels = ext_labels.unsqueeze(0) == ext_labels.unsqueeze(1)

        return ret_labels

    def find_selection(self, ranking: torch.Tensor) -> torch.Tensor:
        if self.per_prototype:
            occurences = {i: 0 for i in range(self.num_prototypes)}
            sorted_ranking, indices = sort_tensor(ranking)
            sel = []

            for idx, element in enumerate(sorted_ranking):
                proto_idx = self.max_proto_indices[indices[idx][0],
                                                   indices[idx][1]].item()
                if occurences[proto_idx] < self.k:
                    occurences[proto_idx] += 1
                    sel.append(indices[idx])

        else:
            # Get the k highest elements and their 2D indices
            sorted_ranking, indices = sort_tensor(ranking)
            # Take the last k elements (highest values) since sort_tensor sorts in ascending order
            sel = indices[-self.k:]
            return sel

    def calc_loss(self):
        return self.gamma * torch.abs(
            self.feat_sim[self.selection] - self.frozen_sim[self.selection]).mean()

    def calc_self_sim(self, x: torch.Tensor) -> torch.Tensor:
        '''
        returns similarity matrix of shape (B*N, B*N) where B is batch size and N is number of patches
        using the specified similarity measure from self.similarity_method
        '''
        batch_size, num_patches, embed_dim = x.shape

        # Reshape x to (B*N, 1, embed_dim) for similarity functions
        x_reshaped = x.reshape(batch_size * num_patches, 1, embed_dim)

        # Create "prototypes" from the same tensor reshaped to (B*N, embed_dim)
        x_as_protos = x.reshape(batch_size * num_patches, embed_dim)

        if self.similarity_method == 'cosine':
            sim_matrix = compute_cosine_similarity(
                x_reshaped, x_as_protos)  # (B*N, B*N)
        elif self.similarity_method == 'log_l2':
            sim_matrix = compute_log_l2_similarity(x_reshaped, x_as_protos)
        elif self.similarity_method == 'rbf':
            sim_matrix = compute_rbf_similarity(
                x_reshaped, x_as_protos, gamma=self.rbf_gamma)
        else:
            raise ValueError(
                f"Unknown similarity method: {self.similarity_method}")

        return sim_matrix


def sort_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    Sorts the input tensor and returns sorted values and their original indices
    '''
    # Flatten and sort
    flattened = tensor.flatten()
    sorted_values, flat_indices = torch.sort(flattened)

    # Convert flat indices back to original multi-dimensional indices
    original_indices = torch.unravel_index(flat_indices, tensor.shape)
    # Returns tuple of tensors: (row_indices, col_indices)

    # Convert to coordinate pairs if needed
    coords = torch.stack(original_indices, dim=1)
    # Shape: (N, 2) where each row is [row, col]

    return sorted_values, coords
