import numpy as np
import torch
import pandas as pd
import json
import importlib
from tdc_starter_kit import utils
from tdc_starter_kit import wrn

from torch.utils.data import Subset

from functools import partial
from tqdm import tqdm

from vit_pytorch.simple_vit import posemb_sincos_2d
from einops import rearrange

# Fur updating Activation Clustering on each import
import trojanzoo_.torch_adopted.defenses.backdoor.training_filtering.activation_clustering_features
importlib.reload(trojanzoo_.torch_adopted.defenses.backdoor.training_filtering.activation_clustering_features)
ActivationClustering = trojanzoo_.torch_adopted.defenses.backdoor.training_filtering.activation_clustering_features.ActivationClustering

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

# Data loading utils
def load_specs(poisoned_path):
    specifications = {}
    infos = {}
    for i in range(500):
        index = str(i).zfill(4)
        specifications[index] = torch.load(poisoned_path.format(index, 'attack_specification.pt'))
        with open(poisoned_path.format(index, 'info.json'), "r") as f:
            infos[index] = json.load(f)

    return specifications, infos

def load_models(keys, poisoned_path):
    models = {}
    for key in keys:
        models[key] = torch.load(poisoned_path.format(key, 'model.pt'))
    return models

def filter_by_dataset(dataset, infos):
    return list(filter(
        lambda x: infos[x]['dataset'] == dataset,
        infos.keys()
    ))

def filter_by_trigger_type(trigger_type, infos):
    return list(filter(
        lambda x: infos[x]['trigger_type'] == trigger_type,
        infos.keys()
    ))

def get_poisoned_dataset(clean_dataset, attack_specification, poisoned_indices=None):
    if attack_specification is None:
        return clean_dataset

    poisoned_dataset = utils.PoisonedDataset(clean_dataset, attack_specification)

    if poisoned_indices is not None:
        poisoned_dataset.poisoned_indices = poisoned_indices

    return poisoned_dataset


# Metrics utils
def get_metrics(result, poisoned_dataset):
    tp = result[poisoned_dataset.poisoned_indices].sum()
    fp = result.sum() - tp
    fn = poisoned_dataset.poisoned_indices.shape[0] - tp
    tn = result.shape[0] - poisoned_dataset.poisoned_indices.shape[0] - fp

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1


# Feature utils
def mnist_classifier(model, x):
    return model.main[-3:](x)


def mnist_feature_extractor(model, x):
    return model.main[:-3](x)


def _get_ac_result(
    dataset,
    model,
    classifier,
    feature_extractor,
    batch_size=300,
    nb_clusters=6,
    nb_dims=12,
    dim_reduction_method="FastICA",
):
    dev_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.cuda()

    activation_clustering = ActivationClustering(
        classifier, feature_extractor,
        dev_dataloader, nb_clusters,
        nb_dims, reduce_method=dim_reduction_method)

    return activation_clustering.calculate_features()


def get_ac_result(
    model,
    attack_specification,
    clean_dataset,
    **kwargs
):
    model = model.eval()

    if type(model).__name__ == "MNIST_Network":
        classifier = partial(mnist_classifier, model)
        feature_extractor = partial(mnist_feature_extractor, model)
    elif type(model).__name__ == "WideResNet":
        classifier = model._classifier
        feature_extractor = model.feature_extractor
    elif type(model).__name__ == "SimpleViT":
        def feature_extractor(img):
            # *_, h, w, dtype = *img.shape, img.dtype
            x = model.to_patch_embedding(img)
            pe = posemb_sincos_2d(x)
            x = rearrange(x, 'b ... d -> b (...) d') + pe
            x = model.transformer(x)
            x = x.mean(dim = 1)
            return model.to_latent(x)

        def classifier(x):
            return model.linear_head(x)
    else:
        raise NotImplementedError()

    poisoned_dataset = get_poisoned_dataset(clean_dataset, attack_specification)
    return poisoned_dataset, _get_ac_result(poisoned_dataset, model, classifier, feature_extractor, **kwargs)


def fix_features(ac_features):
    # it is not 1d arrays, so we cant put it to dataset
    del ac_features['all_fm']
    del ac_features['all_reduced_fm']

    top2 = torch.topk(ac_features['classifier_ensurance'], 2, dim=1).values
    ac_features['classifier_ensurance_second'] = top2[:, 1]
    ac_features['classifier_ensurance_first'] = top2[:, 0]
    del ac_features['classifier_ensurance']


def get_min_img_dist_to_cluster_means_feature(
        keys,
        df,
        original_is_poisoned,
        clean_dataset,
        specifications,
        batch_size=1000
):
    # original_lables may be poisoned if dataset is
    all_feature = pd.Series(dtype=np.float64)
    image_shape = clean_dataset[0][0].shape
    for key in tqdm(keys, leave=False):
        key_feature = pd.Series(data=np.full(len(clean_dataset), np.inf), name="min_img_dist_to_cluster_means")
        attack_specification = specifications[key]
        poisoned_dataset = get_poisoned_dataset(
            clean_dataset,
            attack_specification,
            poisoned_indices=original_is_poisoned[df.key == int(key)].nonzero()[0])

        class_to_cluster_means = {}


        # получаем кластера для каждого класса и усредняем их
        cur_key_dataset = df[df.key == int(key)]
        cur_classes = cur_key_dataset.all_pred_label.unique()
        for target_class in cur_classes:
            cur_class_dataset = cur_key_dataset[cur_key_dataset.all_pred_label == target_class]

            cluster_means = [] # cluster -> mean image
            for cluster in cur_class_dataset.all_clusters.unique():
                images_for_cluster_idx = ((cur_key_dataset.all_pred_label == target_class) & (cur_key_dataset.all_clusters == cluster)).values.nonzero()[0]
                dev_dataloader = torch.utils.data.DataLoader(
                    Subset(poisoned_dataset, images_for_cluster_idx),
                    batch_size=batch_size, shuffle=False, num_workers=1)

                mean_image = torch.zeros(image_shape)
                for images_set, _ in dev_dataloader:
                    mean_image += images_set.sum(dim=0)

                mean_image = mean_image / images_for_cluster_idx.shape[0]

                cluster_means.append((cluster, mean_image))

            class_to_cluster_means[target_class] = cluster_means

        for target_class in cur_classes:
            for cluster, cluster_mean in class_to_cluster_means[target_class]:
                ind = ((cur_key_dataset.all_pred_label != target_class)
                    | (cur_key_dataset.all_clusters != cluster)).values.nonzero()[0]

                dev_dataloader = torch.utils.data.DataLoader(
                    Subset(poisoned_dataset, ind),
                    batch_size=batch_size, shuffle=False, num_workers=1)

                i = 0
                for images_set, _ in dev_dataloader:
                    mean_image += images_set.sum(dim=0)
                    cur_ind = ind[i*batch_size:i*batch_size+images_set.shape[0]]
                    key_feature.loc[cur_ind] = np.minimum(
                        key_feature.loc[cur_ind].values,
                        torch.norm(images_set - cluster_mean.unsqueeze(0), dim=(1,2,3), p=1).numpy()
                    )

        all_feature = pd.concat((all_feature, key_feature), ignore_index=True)
    return all_feature
