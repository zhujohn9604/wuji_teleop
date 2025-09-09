import torch
from collections import defaultdict

def concat_batches(offline_batch: dict[str, torch.Tensor], online_batch: dict[str, torch.Tensor], dim=1):
    batch = defaultdict(list)

    for k in set(offline_batch.keys()) | set(online_batch.keys()):
        v_off = offline_batch.get(k, None)
        v_on = online_batch.get(k, None)

        if isinstance(v_off, dict) or isinstance(v_on, dict):
            # 递归处理嵌套字典
            batch[k] = concat_batches(v_off or {}, v_on or {}, dim=dim)
        else:
            # 对齐张量维度并连接
            shapes = []
            for v in [v_off, v_on]:
                if v is not None:
                    shapes.append(v.shape)

            # 维度自动对齐（PyTorch风格）
            if v_off is None:
                batch[k] = v_on
            elif v_on is None:
                batch[k] = v_off
            else:
                batch[k] = torch.cat((v_off, v_on), dim=dim)

    # 转换为普通字典输出
    return dict(batch)

def unpack_batch(batch):
    """
    Helps to minimize CPU to GPU transfer.
    Assuming that if next_observation is missing, it's combined with observation:

    :param batch: a batch of data from the replay buffer, a dataset dict
    :return: a batch of unpacked data, a dataset dict
    """

    for pixel_key in batch["observations"].keys():
        if pixel_key not in batch["next_observations"]:
            obs_pixels = batch["observations"][pixel_key][:, :-1, ...]
            next_obs_pixels = batch["observations"][pixel_key][:, 1:, ...]

            batch["observations"][pixel_key] = obs_pixels
            batch["next_observations"][pixel_key] = next_obs_pixels

    return batch
