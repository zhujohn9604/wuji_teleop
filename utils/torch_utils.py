import imp
import os
import re
import time
import glob
import torch
import asyncio
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Callable

################
### checkpoint
################

def save_checkpoint(model: torch.nn.Module,
                    step: int,
                    checkpoint_dir: str,
                    keep_last: Optional[int] = None) -> None:
    """
    保存带有时间戳的检查点，支持灵活的历史版本管理

    参数:
        model: 要保存的PyTorch模型
        step: 当前训练步数(16位数字)
        checkpoint_dir: 检查点存储目录
        keep_last: 保留的最新检查点数量(None表示保留所有)
    """
    # 创建带有时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"checkpoint_{step:016d}_{timestamp}.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 构建检查点内容(支持扩展保存训练状态)
    checkpoint = {
        'step': step,
        'timestamp': timestamp,
        'model_state_dict': model.state_dict()
    }

    # 保存检查点
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_name}")

    # 历史版本清理逻辑
    if keep_last is not None and keep_last > 0:
        _cleanup_old_checkpoints(checkpoint_dir, keep_last)

def load_checkpoint(model: torch.nn.Module,
                    checkpoint_dir: str,
                    step: Optional[int] = None) -> Tuple[int, bool]:
    """
    智能加载检查点，支持时间戳识别

    参数:
        model: 要加载的模型
        checkpoint_dir: 检查点目录
        step: 指定加载的步数(None加载最新)

    返回:
        (加载的step, 是否成功加载)
    """
    if step is not None:
        # 构建匹配模式
        pattern = re.compile(rf"checkpoint_{step:016d}_\d{{8}}_\d{{6}}.pth")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if pattern.match(f)]

        if not checkpoints:
            print(f"Warning: No checkpoint found for step {step}")
            return -1, False
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
    else:
        # 自动查找最新检查点
        checkpoint_path, step = _find_latest_checkpoint(checkpoint_dir)
        if checkpoint_path is None:
            print("Warning: No checkpoints available")
            return -1, False

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 设备兼容性处理
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded checkpoint [Step:{step}] [Time:{checkpoint['timestamp']}]")
    return step, True

def _cleanup_old_checkpoints(checkpoint_dir: str, keep_last: int) -> None:
    """清理旧版本，保留最新的keep_last个"""
    checkpoints = _list_checkpoints(checkpoint_dir)
    if len(checkpoints) > keep_last:
        for ckpt in checkpoints[:-keep_last]:
            os.remove(ckpt)
            print(f"Removed old checkpoint: {os.path.basename(ckpt)}")

def _find_latest_checkpoint(checkpoint_dir: str) -> Tuple[Optional[str], int]:
    """智能查找最新检查点"""
    checkpoints = _list_checkpoints(checkpoint_dir)
    return (checkpoints[-1], _extract_step(checkpoints[-1])) if checkpoints else (None, -1)

def _list_checkpoints(checkpoint_dir: str) -> list:
    """获取按步数排序的检查点列表"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*_*.pth"))
    checkpoints.sort(key=lambda x: _extract_step(x))
    return checkpoints

def _extract_step(checkpoint_path: str) -> int:
    """从文件名中提取步数"""
    match = re.search(r"checkpoint_(\d{16})_", checkpoint_path)
    return int(match.group(1)) if match else -1

################
### device
################

def device_put(ts: dict | torch.Tensor | np.ndarray, device: torch.device) -> dict:
    if not isinstance(ts, dict):
        return torch.Tensor(ts).to(device)
    return {k: device_put(v, device) for k, v in ts.items()}

def device_get(ts: dict | torch.Tensor) -> dict:
    if isinstance(ts, torch.Tensor):
        return ts.cpu()
    return {k: device_get(v) for k, v in ts.items()}

def xavier_uniform_init(m: torch.nn.Module) -> None:
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
