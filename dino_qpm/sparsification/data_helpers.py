# import os
# import sys
#
# from Datasets.Robin import RobinClass
# from scripts.datasetProblems.createCUB200classMapping import load_cub_class_mapping, get_FGVC_class_mapping, \
#     get_StanfordCars_class_mapping, get_Nabirds_mapping
#
# sys.path.append('..')
# import torch as ch
# from torch.utils.data import TensorDataset
# from robustness.robustness.datasets import DATASETS as VISION_DATASETS
# from robustness.robustness.tools.label_maps import CLASS_DICT
# from language.datasets import DATASETS as LANGUAGE_DATASETS
# from language.model import LANGUAGE_MODEL_DICT
# from transformers import AutoTokenizer
#
#
# def get_label_mapping(dataset_name):
#     if dataset_name == 'imagenet':
#         return CLASS_DICT['ImageNet']
#     elif dataset_name in ['places-10', "places365"]:
#         return CD_PLACES
#     elif dataset_name == "CIFAR-10":
#         return CLASS_DICT["CIFAR"]
#     elif dataset_name == 'sst':
#         return {0: 'negative', 1: 'positive'}
#     elif dataset_name == "Fashion-Mnist":
#         return CLASS_DICT[dataset_name]
#     elif dataset_name == "CUB200" or dataset_name in ["TravelingBirds", "CravelingBirds", "FravelingBirds"]:
#         return load_cub_class_mapping()
#     elif dataset_name == "FGVCAircraft":
#         return get_FGVC_class_mapping()
#     elif dataset_name == "StanfordCars":
#         return get_StanfordCars_class_mapping()
#     elif dataset_name == "GTSRB":
#         return {x: "Label_" + str(x) for x in range(200)}
#     elif dataset_name == "NABirds":
#         return get_Nabirds_mapping()
#         return {x: "Label_" + str(x) for x in range(555)}
#     elif dataset_name == "Robin":
#         class_dict = RobinClass.class_dict
#         return {i: x for x, i in class_dict.items()}
#     elif 'jigsaw' in dataset_name:
#         category = dataset_name.split('jigsaw-')[1] if 'alt' not in dataset_name \
#             else dataset_name.split('jigsaw-alt-')[1]
#         return {0: f'not {category}', 1: f'{category}'}
#     else:
#         raise ValueError("Dataset not currently supported...")
#
#
# def load_dataset(dataset_name, dataset_path, dataset_type,
#                  batch_size, num_workers,
#                  maxlen_train=256, maxlen_val=256,
#                  shuffle=False, model_path=None, return_sentences=False, dataset_params=None, augmentation_params=None,
#                  noisied_fitting=False):
#     if dataset_type == 'vision':
#         changed = False
#         # if dataset_name == 'places-10':
#         #     changed=True
#         #     dataset_name = 'places365'
#         if dataset_name not in VISION_DATASETS:
#             raise ValueError("Vision dataset not currently supported...")
#         dataset = VISION_DATASETS[dataset_name](os.path.expandvars(dataset_path), custom_class_args=dataset_params,
#                                                 augmentation_params=augmentation_params)
#
#         # if dataset_name == 'places365' and changed:
#         #     dataset.num_classes = 10
#
#         train_loader, test_loader = dataset.make_loaders(num_workers,
#                                                          batch_size,
#                                                          data_aug=noisied_fitting,
#                                                          shuffle_train=shuffle,
#                                                          shuffle_val=shuffle)
#         return dataset, train_loader, test_loader
#     else:
#         if model_path is None:
#             model_path = LANGUAGE_MODEL_DICT[dataset_name]
#
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#
#         kwargs = {} if 'jigsaw' not in dataset_name else \
#             {'label': dataset_name[11:] if 'alt' in dataset_name \
#                 else dataset_name[7:]}
#         kwargs['return_sentences'] = return_sentences
#         train_set = LANGUAGE_DATASETS(dataset_name)(filename=f'{dataset_path}/train.tsv',
#                                                     maxlen=maxlen_train,
#                                                     tokenizer=tokenizer,
#                                                     **kwargs)
#         test_set = LANGUAGE_DATASETS(dataset_name)(filename=f'{dataset_path}/test.tsv',
#                                                    maxlen=maxlen_val,
#                                                    tokenizer=tokenizer,
#                                                    **kwargs)
#         train_loader = ch.utils.data.DataLoader(dataset=train_set,
#                                                 batch_size=batch_size,
#                                                 num_workers=num_workers)
#         test_loader = ch.utils.data.DataLoader(dataset=test_set,
#                                                batch_size=batch_size,
#                                                num_workers=num_workers)
#         # assert len(np.unique(train_set.df['label'].values)) == len(np.unique(test_set.df['label'].values))
#         train_set.num_classes = 2
#         # train_loader.dataset.targets = train_loader.dataset.df['label'].values
#         # test_loader.dataset.targets = test_loader.dataset.df['label'].values
#
#         return train_set, train_loader, test_loader
#
#
# class IndexedTensorDataset(ch.utils.data.TensorDataset):
#     def __getitem__(self, index):
#         val = super(IndexedTensorDataset, self).__getitem__(index)
#         return val + (index,)
import torch


class NormalizedRepresentation(torch.nn.Module):
    def __init__(self, loader, metadata, device='cuda', tol=1e-5):
        super(NormalizedRepresentation, self).__init__()

        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = torch.clamp(metadata['X']['std'], tol)

    def forward(self, X):
        return (X - self.mu.to(self.device)) / self.sigma.to(self.device)
