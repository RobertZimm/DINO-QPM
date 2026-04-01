full_vals = {
    "zero": [0],
    "proto_pre_pooling_mode": [None, "softmax_temp", "softmax_conv_norm"],
    "proto_similarity_method": ["cosine", "log_l2", "rbf"],
    "n_f_star": [20, 30, 40, 50, 60],
    "n_f_c": [3, 4, 5, 6, 7],
    "on_off_float": [0.0, 0.5],
    "epochs": [100, 200],
    "lin_params": [0, 0.1, 0.5, 1, 5, 20],
    "low_log_dec": [0],
    "mid_log": [1e-4, 0.5e-3, 1e-3, 5e-3, 1e-2, 1e-1, 1],
    "low_log": [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10],
    "low_log_red": [1e-5, 1e-3, 1],
    "low_log_mid": [0, 1e-5, 1e-3, 1e-1, 1, 5],
    "high_log": [0, 1, 10, 100, 1000],
    "other_base_2": [128, 256, 512, 1024, 2048, 4096],
    "sparse_high_base_2": [512, 1024, 2048, 4096],
    "low_base_2": [8, 16, 32, 64, 100, 128, 256],
    "zero_to_zero_five": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "approach": [("normal", "normal"),
                 ("concat", "avg_pooling"),
                 ("normal", "avg_pooling"),
                 ("normal", "mean_avg_pooling")],
    "model_type": ["small_reg", "base_reg", "large_reg"],
    "best_approaches": [("normal", "normal", "neco_base_reg"),
                        ("normal", "normal", "base_reg"),
                        ("normal", "avg_pooling", "base_reg"),
                        ("normal", "mean_avg_pooling", "base_reg")],
    "qpm_sel_pairs": [(20, 3),
                      (25, 3),
                      (30, 4),
                      (35, 4),
                      (40, 5),
                      (50, 3),
                      (50, 5),
                      (50, 8),
                      (80, 10),
                      (100, 10),
                      (100, 15)],
    "n_layers": [1, 2, 3, 4],
    "activation": ["relu", "gelu", "sigmoid"],  # "sigmoid", "leaky_relu",
    "on_off": [True, False],
    "fitzpatrick_split": ['verified', 'random', 'source_a', 'source_b',
                          'fitz_3-6', 'fitz_1-2_5-6', 'fitz_1-4'],
    "proto_pre_pooling_mode": [None, "softmax_temp", "softmax_conv_norm"],
    "temperature": [0.2, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 5.0],
    "pooling_type": ["avg", "max", ("smooth_max", "max"), ("smooth_max", "avg"),
                     ("smooth_max", "close_to_avg"), ("smooth_max", "very_soft")],
    "random_noise": [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1],
    "ignore_first_n_components": [1, 2, 3, 10, 20, 50],
    "n_clusters": [2, 3, 5, 10],
    "epochs_per_loop": [10, 15],
    "gamma": [0.5, 0.75, 0.9, 0.95, 0.99]
}

reduced_vals = {
    "lin_params": [0, 0.1, 0.5, 5, 20],
    "high_log": [0, 1, 1000],
    "sparse_high_base_2": [2048, 4096, 8192],
    "low_log": [0, 0.196],
    "high_base_2": [2 ** i for i in range(8, 14, 2)],
    "low_base_2": [8, 32],
    "zero_to_zero_five": [0, 0.2, 0.4],
    # "best_approaches": [("normal", "normal", "base_reg"),
    #                     ("normal", "avg_pooling", "base_reg")],
    # "n_layers": [2, 4],
    "approach": [("normal", "normal"),
                 ("concat", "avg_pooling"),
                 ("normal", "avg_pooling"),],
    "best_approaches": [("normal", "normal", "base_reg"),
                        ("normal", "avg_pooling", "base_reg"),
                        ("normal", "mean_avg_pooling", "base_reg")],
}

# Param
low_log_dec = ["cofs_weight; finetune"]

n_f_star = ["n_f_star"]

n_f_c = ["n_f_c"]

lin = []

zero = ["rpl_weight"]

on_off_float = []

mid_log = ["l1_fv_weight"]

low_log_mid = ["grounding_loss_weight"]

low_log_red = [  # "l1_fv_weight",
    # "rpl_weight"
]

low_log = ["l1_w_weight",
           "start_lr",
           "weight_decay",
           "beta_avg",
           "fdl",
           "gamma",
           "l1_fm_weight",
           "iou_weight",
           "cofs_weight; dense",
           "pdl"]

low_base_2 = ["batch_size",]  # "cofs_k"]

zero_to_zero_five = ["dropout"]

sparse_high_base_2 = ["hidden_size", "cofs_k"]

other_base_2 = ["n_prototypes", "n_features"]

on_off = ["use_dropout", "use_batch_norm", "proto_use_feat_vec",
          "proto_relu", "scale_feat_vec", "relu_after_scaling", "norm_with_max"]

temperature = ["proto_softmax_tau"]

random_noise = ["random_noise"]

log_dir = ["log_dir"]

ignore_first_n_components = ["ignore_first_n_components"]

param_mapping = {
    "zero": zero,
    "n_f_star": n_f_star,
    "n_f_c": n_f_c,
    "on_off_float": on_off_float,
    "mid_log": mid_log,
    "low_log_mid": low_log_mid,
    "low_log_red": low_log_red,
    "low_log_dec": low_log_dec,
    "lin_params": lin,
    "low_log": low_log,
    "low_base_2": low_base_2,
    "zero_to_zero_five": zero_to_zero_five,
    "sparse_high_base_2": sparse_high_base_2,
    "on_off": on_off,
    "other_base_2": other_base_2,
    "temperature": temperature,
    "random_noise": random_noise,
    "log_dir": log_dir,
    "ignore_first_n_components": ignore_first_n_components
}
