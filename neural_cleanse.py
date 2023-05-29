import torch
import typing as tp
from collections.abc import Callable
# from tqdm import tqdm
from trojanzoo.utils.metric import normalize_mad
from trojanzoo.utils.logger import MetricLogger
from torch import optim


def get_raw(x):
    x = torch.tanh(x)
    return x / 2 + 0.5

def get_encoded(x):
    return torch.atanh((x - 0.5) * (2))


def format_list(_list: list, _format: str = ':8.3f') -> str:
    return '[' + ', '.join(['{{{}}}'.format(_format).format(a) for a in _list]) + ']'


class NeuralCleanse():
    r"""Neural Cleanse proposed by Bolun Wang and Ben Y. Zhao
    from University of Chicago in IEEE S&P 2019.

    It is a model inspection backdoor defense
    (It further dynamically adjust mask norm cost in the loss
    and set an early stop strategy.)

    For each class, Neural Cleanse tries to optimize a recovered trigger
    that any input with the trigger attached will be classified to that class.
    If there is an outlier among all potential triggers, it means the model is poisoned.

    See Also:
        * paper: `Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks`_
        * code: https://github.com/bolunwang/backdoor

    Args:
        nc_cost_multiplier (float): Norm loss cost multiplier.
            Defaults to ``1.5``.
        nc_patience (float): Early stop nc_patience.
            Defaults to ``10.0``.
        nc_asr_threshold (float): ASR threshold in cost adjustment.
            Defaults to ``0.99``.
        nc_early_stop_threshold (float): Threshold in early stop check.
            Defaults to ``0.99``.

    Attributes:
        cost_multiplier_up (float): Value to multiply when increasing cost.
            It equals to ``nc_cost_multiplier``.
        cost_multiplier_down (float): Value to divide when decreasing cost.
            It's set as ``nc_cost_multiplier ** 1.5``.

    Attributes:
        init_cost (float): Initial cost of mask norm loss.
        cost (float): Current cost of mask norm loss.

    .. _Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks:
        https://ieeexplore.ieee.org/document/8835365
    """
    name: str = 'neural_cleanse'

    def __init__(
            self,
            model: Callable, classes: list[tp.Union[str, int]],
            dataset: torch.utils.data.DataLoader, img_shape,
            cost=1e-4, nc_cost_multiplier: float = 1.5, nc_patience: float = 10.0,
            nc_asr_threshold: float = 0.99,
            nc_early_stop_threshold: float = 0.99, device='cuda', defense_remask_epoch=10,
            initial_pattern = None, defense_remask_lr = 0.1,
            **kwargs):
        self.model = model.cuda()
        self.model.eval()
        self.classes = classes
        self.dataset = dataset
        self.init_cost = cost
        self.cost_multiplier_up = nc_cost_multiplier
        self.cost_multiplier_down = nc_cost_multiplier ** 1.5
        self.nc_asr_threshold = nc_asr_threshold
        self.nc_early_stop_threshold = nc_early_stop_threshold
        self.nc_patience = nc_patience
        self.early_stop_patience = self.nc_patience * 2
        self.defense_remask_epoch = defense_remask_epoch
        self.defense_remask_lr = defense_remask_lr
        self.alpha = 1.
        self.device = device
        self.img_shape = img_shape
        self.pattern = torch.rand(img_shape, device=self.device)
        self.mask = torch.rand(img_shape, device=self.device)
        self.cost = self.init_cost
        self.initial_pattern = initial_pattern

    def patch_images(self, images, mask, pattern):
        """
        Add patch to images
        """
        mask = mask.unsqueeze(0)
        pattern = pattern.unsqueeze(0)
        return mask * (self.alpha * pattern + (1 - self.alpha) * images) + (1 -mask) * images

    def detect(self):
        self.pattern = torch.rand(self.img_shape, device=self.device)
        self.mask = torch.rand(self.img_shape, device=self.device)

        mask_list, pattern_list, loss_list, asr_list = self.get_mark_loss_list()
        mask_norms: torch.Tensor = (mask_list * pattern_list)[:, -1].flatten(start_dim=1).norm(p=1, dim=1)
        mask_norm_list: list[float] = mask_norms.tolist()
        print()
        print('asr           : ' + format_list(asr_list))
        print('mask norms    : ' + format_list(mask_norm_list))
        print('loss          : ' + format_list(loss_list))
        print()
        print('asr MAD       : ' + format_list(normalize_mad(asr_list).tolist()))
        print('mask norm MAD : ' + format_list(normalize_mad(mask_norms).tolist()))
        print('loss MAD      : ' + format_list(normalize_mad(loss_list).tolist()))

        return mask_list, pattern_list, loss_list, asr_list

    def get_mark_loss_list(self, verbose: bool = True,
                           **kwargs) -> tuple[torch.Tensor, list[float], list[float]]:
        r"""Get list of mark, loss, asr of recovered trigger for each class.

        Args:
            verbose (bool): Whether to output jaccard index for each trigger.
                It's also passed to :meth:`optimize_mark()`.
            **kwargs: Keyword arguments passed to :meth:`optimize_mark()`.

        Returns:
            (torch.Tensor, list[float], list[float]):
                list of mark, loss, asr with length ``num_classes``.
            asr - accuracy of the model after adding the trigger.
        """
        mask_list: list[torch.Tensor] = []
        pattern_list: list[torch.Tensor] = []
        loss_list: list[float] = []
        asr_list: list[float] = []
        # todo: parallel to avoid for loop
        for label in range(len(self.classes)):
            print('Class: ', (label, len(self.classes)))
            mask, pattern, loss = self.optimize_mark(label, verbose=verbose, **kwargs)
            # validate model on poisoned images
            asr = self.validate_model(mask, pattern, label)
            mask_list.append(mask)
            pattern_list.append(pattern)
            loss_list.append(loss)
            asr_list.append(asr)
        mask_list_tensor = torch.stack(mask_list)
        pattern_list_tensor = torch.stack(pattern_list)
        return mask_list_tensor, pattern_list_tensor, loss_list, asr_list

    def validate_model(self, mask, pattern, target_label):
        """
        Calculate accuracy on poisoned images
        """
        #self.model.eval()
        correct = 0
        total = 0
        for images, labels in self.dataset:
            images.to(self.device)
            poisoned_images = self.patch_images(images.to(self.device), mask, pattern)
            labels = labels.to(self.device)
            outputs = self.model(poisoned_images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == target_label).sum().item()
        return correct / total

    def optimize_mark(self, label: int,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:
        r"""
        Args:
            label (int): The class label to optimize.
            loader (collections.abc.Iterable):
                Data loader to optimize trigger.
                Defaults to ``self.dataset.loader['train']``.
            logger_header (str): Header string of logger.
                Defaults to ``''``.
            verbose (bool): Whether to use logger for output.
                Defaults to ``True``.
            **kwargs: Keyword arguments passed to :meth:`loss()`.

        Returns:
            (torch.Tensor, torch.Tensor):
                Optimized mark tensor with shape ``(C + 1, H, W)``
                and loss tensor.
        """

        # parameters to update cost
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # counter for early stop
        self.early_stop_counter = 0
        self.early_stop_norm_best = float('inf')

        atanh_mask = torch.randn(self.mask.size(), requires_grad=True, device=self.device)

        if self.initial_pattern is not None:
            atanh_pattern_root = get_encoded(self.initial_pattern).detach().clone()
            atanh_pattern_root = atanh_pattern_root.cuda()
            atanh_pattern_root.requires_grad = True
            optimizer = optim.Adam([atanh_mask], lr=self.defense_remask_lr, betas=(0.5, 0.9))
        else:
            atanh_pattern_root = torch.randn(self.pattern.size(), requires_grad=True, device=self.device)
            optimizer = optim.Adam([atanh_pattern_root, atanh_mask], lr=self.defense_remask_lr, betas=(0.5, 0.9))

        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.defense_remask_epoch)

        # best optimization results
        norm_best: float = float('inf')
        loss_best: float = None

        logger = MetricLogger(indent=4)
        logger.create_meters(loss='{last_value:.3f}',
                             acc='{last_value:.3f}',
                             norm='{last_value:.3f}',
                             entropy='{last_value:.3f}',)
        batch_logger = MetricLogger()
        logger.create_meters(loss=None, acc=None, entropy=None)

        iterator = range(self.defense_remask_epoch)
        if verbose:
            iterator = logger.log_every(iterator, header=logger_header)
        loss = torch.nn.CrossEntropyLoss()
        self.model.eval()
        for _ in iterator:
            batch_logger.reset()
            # for _input, _label in tqdm(self.dataset, leave=False):
            for _input, _label in self.dataset:
                optimizer.zero_grad()

                atanh_pattern = atanh_pattern_root
                _input = _input.detach().to(self.device)
                pattern = get_raw(atanh_pattern)    # (c+1, h, w)
                mask = get_raw(atanh_mask)   # (c+1, h, w)
                trigger_input = self.patch_images(_input, mask, pattern).cuda()
                trigger_label = (label * torch.ones_like(_label)).cuda()
                trigger_output = self.model(trigger_input)

                batch_acc = trigger_label.eq(trigger_output.argmax(1)).float().mean()

                batch_entropy = torch.nn.functional.cross_entropy(trigger_output, trigger_label)
                # batch_entropy = loss(trigger_output, trigger_label)
                if self.initial_pattern is not None:
                    batch_norm: torch.Tensor = ((((pattern < 0.00001) * mask) + mask).norm(p=1) / (mask * (pattern > 0.00001)).norm(p=1))
                else:
                    batch_norm: torch.Tensor = (mask + pattern).norm(p=1)

                batch_loss = batch_entropy + self.cost * batch_norm

                batch_loss.backward()
                optimizer.step()

                batch_size = _label.size(0)
                self.pattern = pattern.detach().clone()
                self.mask = mask.detach().clone()
                batch_logger.update(n=batch_size,
                                    loss=batch_loss.item(),
                                    acc=batch_acc.item(),
                                    entropy=batch_entropy.item())
            lr_scheduler.step()
            self.pattern = pattern.detach().clone()    # (c+1, h, w)
            self.mask = mask.detach().clone()    # (c+1, h, w)

            # check to save best mask or not
            loss = batch_logger.meters['loss'].global_avg
            acc = batch_logger.meters['acc'].global_avg
            # with no grad
            with torch.no_grad():
                norm = float((self.mask * self.pattern).norm(p=1))

            entropy = batch_logger.meters['entropy'].global_avg

            if norm < norm_best:
                mask_best = self.mask.detach().clone()
                pattern_best = self.pattern.detach().clone()
                loss_best = loss
                logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy)

            if self.check_early_stop(loss=loss, acc=acc, norm=norm, entropy=entropy):
                print('early stop')
                break

        self.mask = mask_best
        self.pattern = pattern_best
        return mask_best, pattern_best, loss_best

    def check_early_stop(self, acc: float, norm: float, **kwargs) -> bool:
        # update cost
        if self.cost == 0 and acc >= self.nc_asr_threshold:
            self.cost_set_counter += 1
            if self.cost_set_counter >= self.nc_patience:
                self.cost = self.init_cost
                self.cost_up_counter = 0
                self.cost_down_counter = 0
                self.cost_up_flag = False
                self.cost_down_flag = False
                # print(f'initialize cost to {self.cost:.2f}%.2f')
        else:
            self.cost_set_counter = 0

        if acc >= self.nc_asr_threshold:
            self.cost_up_counter += 1
            self.cost_down_counter = 0
        else:
            self.cost_up_counter = 0
            self.cost_down_counter += 1

        if self.cost_up_counter >= self.nc_patience:
            self.cost_up_counter = 0
            # prints(f'up cost from {self.cost:.4f} to {self.cost * self.cost_multiplier_up:.4f}',
            #        indent=4)
            self.cost *= self.cost_multiplier_up
            self.cost_up_flag = True
        elif self.cost_down_counter >= self.nc_patience:
            self.cost_down_counter = 0
            # prints(f'down cost from {self.cost:.4f} to {self.cost / self.cost_multiplier_down:.4f}',
            #        indent=4)
            self.cost /= self.cost_multiplier_down
            self.cost_down_flag = True

        early_stop = False
        # check early stop
        if norm < float('inf'):
            if norm >= self.nc_early_stop_threshold * self.early_stop_norm_best:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0
        self.early_stop_norm_best = min(norm, self.early_stop_norm_best)

        if self.cost_down_flag and self.cost_up_flag and self.early_stop_counter >= self.early_stop_patience:
            early_stop = True

        return early_stop
