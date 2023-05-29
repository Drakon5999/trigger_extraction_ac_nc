import torch
from sklearn.decomposition import *
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score
# from tqdm import tqdm

from typing import TYPE_CHECKING
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data

import sklearn.decomposition._fastica as fastica
import numpy as np
from tenacity import retry, stop_after_attempt, wait_none

def _retryable_ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.
    Used internally by FastICA --main loop
    """
    W = fastica._sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = fastica._sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        # np.einsum allows having the lowest memory footprint.
        # It is faster than np.diag(np.dot(W1, W.T)).
        lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        raise ValueError("FastICA did not converge. Consider increasing "
                            "tolerance or the maximum number of iterations.")

    return W, ii + 1

# override _ica_par to use retry
fastica._ica_par = _retryable_ica_par


class ActivationClustering():
    def __init__(
        self,
        classifier: Callable[[torch.Tensor], int],
        feature_extractor: Callable[[torch.Tensor], torch.Tensor],
        dataloader: torch.utils.data.DataLoader,
        nb_clusters: int = 2,
        nb_dims: int = 10,
        reduce_method: str = 'FastICA',
        clustering_method: str = 'KMeans',
        cluster_analysis: str = 'silhouette_score',
    ):
        self.nb_clusters = nb_clusters
        self.nb_dims = nb_dims
        self.reduce_method = reduce_method
        self.cluster_analysis = cluster_analysis
        self.dataloader = dataloader
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.projector_reserved = PCA(n_components=self.nb_dims)

        match self.reduce_method:
            case 'FastICA':
                self.projector = fastica.FastICA(n_components=self.nb_dims, whiten='unit-variance', max_iter=2000, tol=1e-5)
            case 'PCA':
                self.projector = PCA(n_components=self.nb_dims)
            case 'FA':
                self.projector = FactorAnalysis(n_components=self.nb_dims)
            case 'IncrementalPCA':
                self.projector = IncrementalPCA(n_components=self.nb_dims)
            case 'KernelPCA':
                self.projector = KernelPCA(n_components=self.nb_dims)
            case 'LatentDirichletAllocation':
                self.projector = LatentDirichletAllocation(n_components=self.nb_dims)
            case 'MiniBatchSparsePCA':
                self.projector = MiniBatchSparsePCA(n_components=self.nb_dims)
            case 'NMF':
                self.projector = NMF(n_components=self.nb_dims)
            case 'MiniBatchNMF':
                self.projector = MiniBatchNMF(n_components=self.nb_dims)
            case 'SparsePCA':
                self.projector = SparsePCA(n_components=self.nb_dims)
            case 'TruncatedSVD':
                self.projector = TruncatedSVD(n_components=self.nb_dims)
            case _:
                raise ValueError(self.reduce_method + ' dimensionality reduction method not supported.')
        match clustering_method:
            case 'KMeans':
                self.clusterer = KMeans(n_clusters=self.nb_clusters, n_init="auto")
            case 'OPTICS':
                self.clusterer = OPTICS(n_jobs=1)

    @retry(stop=stop_after_attempt(5), wait=wait_none())
    def _get_projector_value(self, fm: torch.Tensor):
        return torch.as_tensor(self.projector.fit_transform(fm.numpy()))

    def calculate_features(self) -> torch.Tensor:
        all_fm = []
        all_pred_label = []
        loader = self.dataloader
        labels = set()

        # classifier ensurance feature for each sample
        classifier_ensurance = []
        # for _input, _label in tqdm(loader, leave=False):
        for _input, _label in loader:
            labels.update([l.item() for l in _label])
            # this is model features
            fm = self.feature_extractor(_input.to(self.device))
            pred_probs = torch.nn.functional.softmax(self.classifier(fm), dim=1)
            pred_label = torch.argmax(pred_probs, dim=1)
            assert len(pred_label) > 1, 'classifier should return value for each class'

            classifier_ensurance.append(pred_probs.detach().cpu())
            # we use flatten because feature extractor can return non 1d feature maps
            all_fm.append(torch.flatten(fm.detach().cpu(), 1, -1))
            all_pred_label.append(pred_label.detach().cpu())

        classifier_ensurance = torch.cat(classifier_ensurance)
        all_fm = torch.cat(all_fm)
        all_pred_label = torch.cat(all_pred_label)
        assert all_pred_label.shape == (len(all_fm),), 'all_pred_label and all_fm should have same length'

        idx_list: list[torch.Tensor] = []
        reduced_fm_centers_list: list[torch.Tensor] = []
        kwargs_list: list[dict[str, torch.Tensor]] = []
        all_clusters = {}
        all_clusters_flatten = torch.empty(all_pred_label.shape, dtype=torch.int)
        all_sample_silhuette = np.empty(all_pred_label.shape)
        all_sample_distance_to_cluster_centroid = np.empty(all_pred_label.shape)
        all_sample_mean_distance_to_cluster_centroid_amoung_cluster = np.empty(all_pred_label.shape)
        all_sample_relative_cluster_size = np.empty(all_pred_label.shape)
        all_sample_activation_norm = np.empty(all_pred_label.shape)
        all_sample_min_distance_to_other_classes = np.empty(all_pred_label.shape)
        all_reduced_fm = torch.empty(all_pred_label.shape[0], self.nb_dims)
        # for _class in tqdm(labels, leave=False):
        for _class in labels:
            idx = all_pred_label == _class
            fm = all_fm[idx]
            try:
                reduced_fm = self._get_projector_value(fm)
            except Exception:
                # if we can't reduce the dimension, we just user PCA projector
                reduced_fm = torch.as_tensor(self.projector_reserved.fit_transform(fm.numpy()))
            all_reduced_fm[idx] = reduced_fm.clone().detach()
            cluster_class = torch.as_tensor(self.clusterer.fit_predict(reduced_fm))
            all_clusters_flatten[idx] = cluster_class.clone().detach()
            all_clusters[_class] = cluster_class.clone().detach()
            kwargs_list.append(dict(cluster_class=cluster_class, reduced_fm=reduced_fm))
            idx_list.append(idx)

            reduced_fm_centers_list.append(torch.median(reduced_fm, dim=0).values)
            all_sample_silhuette[idx] = silhouette_score(reduced_fm, cluster_class)
            # TODO: should we norm all_sample_distance_to_cluster_centroid and other features?

            # mean distance from each sample to its cluster centroid
            for i in range(self.nb_clusters):
                class_cluster_idx = cluster_class == i
                cluster_centroid = reduced_fm[class_cluster_idx].mean(dim=0)
                all_sample_distance_to_cluster_centroid[idx][class_cluster_idx] = torch.norm(reduced_fm[class_cluster_idx] - cluster_centroid.unsqueeze(0), p=2, dim=1).numpy()

                all_sample_mean_distance_to_cluster_centroid_amoung_cluster[idx][class_cluster_idx] = all_sample_distance_to_cluster_centroid[idx][class_cluster_idx].mean()

                all_sample_relative_cluster_size[idx][class_cluster_idx] = (class_cluster_idx).sum() / len(cluster_class)

                all_sample_activation_norm[idx][class_cluster_idx] = torch.norm(fm[i], 2).item()

        reduced_fm_centers = torch.stack(reduced_fm_centers_list)

        # for _class in tqdm(labels, leave=False):
        for _class in labels:
            # calculate minimum amoung distances for each sample to other classes
            no_this_class_center_mask = torch.ones((reduced_fm_centers.shape[0],), dtype=torch.bool)
            no_this_class_center_mask[_class] = 0
            min_norm = np.empty((idx_list[_class].sum(),))
            min_norm[:] = np.inf
            for other_class_center in reduced_fm_centers[no_this_class_center_mask, : ]:
                assert other_class_center.shape == (self.nb_dims,), 'other_class_center should be 1d'
                min_norm = np.minimum(
                    min_norm,
                    torch.norm(
                        all_reduced_fm[idx_list[_class]] - other_class_center.unsqueeze(0),
                        p=2, dim=1
                    ).numpy()
                )

            all_sample_min_distance_to_other_classes[idx_list[_class]] = min_norm


        self.all_clusters = all_clusters
        self.all_ac_features = {
            'classifier_ensurance': classifier_ensurance,
            'all_fm': all_fm,
            'all_reduced_fm': all_reduced_fm,
            'all_pred_label': all_pred_label,
            # NOTE: cluster index may repeat amoung different predicted labels
            # but it is different clusters
            'all_clusters': all_clusters_flatten,
            'all_sample_silhuette': all_sample_silhuette,
            'all_sample_distance_to_cluster_centroid': all_sample_distance_to_cluster_centroid,
            'all_sample_mean_distance_to_cluster_centroid_amoung_cluster': all_sample_mean_distance_to_cluster_centroid_amoung_cluster,
            'all_sample_relative_cluster_size': all_sample_relative_cluster_size,
            'all_sample_activation_norm': all_sample_activation_norm,
            'all_sample_min_distance_to_other_classes': all_sample_min_distance_to_other_classes,
        }
        return self.all_ac_features
