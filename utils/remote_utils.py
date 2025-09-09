from typing import Callable

import threading
import subprocess
import os
import re
import time
import shutil

import torch
import asyncio

from omegaconf import OmegaConf

import multiprocessing as mp

from .print_utils import print_green, print_red, print_yellow, print_blue


class RemoteOperator:
    def __init__(self, remote_server, max_retries=3, retry_delay=5):
        self.remote_server = remote_server  # 直接使用 user@hostname 格式
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def retry_subprocess(self, cmd, show_progress=False):
        retry_count = 0
        cmd_success = False
        while not cmd_success and retry_count < self.max_retries:
            try:
                # print(f"执行命令: {' '.join(cmd)}")
                if show_progress:
                    # For progress display, don't capture output but pipe directly to terminal
                    process = subprocess.Popen(cmd, text=True)
                    process.communicate()
                    exit_code = process.returncode
                    if exit_code != 0:
                        raise subprocess.CalledProcessError(exit_code, cmd)
                else:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)

                cmd_success = True
                return True
            except subprocess.CalledProcessError as e:
                print(f"命令失败 (尝试 {retry_count + 1}/{self.max_retries}): {e}")
                if not show_progress:  # Only show stderr if we're not already showing progress
                    print("错误输出:", e.stderr)
                retry_count += 1
                if retry_count < self.max_retries:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
            except Exception as e:
                print(f"命令发生未知错误: {str(e)}")
                raise RuntimeError(f"命令执行失败: {str(e)}")

        if not cmd_success:
            # print_red(f"命令执行失败: 已达到最大重试次数 {self.max_retries} 次")
            # return False
            raise RuntimeError(f"命令执行失败: 已达到最大重试次数 {self.max_retries} 次")
        return False

    def rsync_upload(self, local_path: str, remote_path: str, delete_local: bool = False):
        print(f"[UPLOAD] 开始上传: {local_path} -> {self.remote_server}:{remote_path}")
        self.mkdir(remote_path)
        rsync_cmd = [
            'rsync',
            '-avz',  # 归档模式，保持属性，压缩传输
            '--delete',  # 删除目标中源没有的文件
            '--partial',  # 保留部分传输的文件
            # '--progress',  # 显示进度
            # '--info=progress2',  # 更紧凑的进度显示
            local_path + '/',  # 注意末尾的/表示同步目录内容
            f"{self.remote_server}:{remote_path}"
        ]

        success = self.retry_subprocess(rsync_cmd, show_progress=False)
        if success:
            print_green(f"[UPLOAD] 上传成功: {local_path} -> {self.remote_server}:{remote_path}")
            if delete_local:
                shutil.rmtree(local_path, ignore_errors=True)
                print(f"[UPLOAD] 本地文件已删除: {local_path}")
        else:
            print(f"[UPLOAD] 上传失败: {local_path} -> {self.remote_server}:{remote_path}")
        return success

    def rsync_download_file(self, remote_path: str, local_dir: str):
        print(f"[DOWNLOAD] 开始下载: {self.remote_server}:{remote_path} -> {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        assert os.path.isdir(local_dir), f"{local_dir} is not a directory"

        rsync_cmd = [
            'rsync',
            '-avz',  # Archive mode, preserve attributes, compress
            '--delete',  # Delete files in target that are not in source
            '--partial',  # 保留部分传输的文件
            '--progress',  # 显示进度
            '--info=progress2',  # 更紧凑的进度显示
            f"{self.remote_server}:{remote_path}",  # No trailing / for single file
            local_dir
        ]

        success = self.retry_subprocess(rsync_cmd, show_progress=True)
        if success:
            print(f"[DOWNLOAD] 下载成功: {self.remote_server}:{remote_path} -> {local_dir}")
        else:
            print(f"[DOWNLOAD] 下载失败: {self.remote_server}:{remote_path} -> {local_dir}")
        return success

    def rsync_download_dir(self, remote_path: str, local_dir: str, exclude_list: list = None):
        """下载整个远程目录到本地

        参数:
            remote_path: 远程目录路径
            local_dir: 本地目标目录路径
            exclude_list: 要排除的文件或目录列表
        """
        print(f"[DOWNLOAD] 开始下载目录: {self.remote_server}:{remote_path} -> {local_dir}")
        os.makedirs(local_dir, exist_ok=True)
        assert os.path.isdir(local_dir), f"{local_dir} is not a directory"

        rsync_cmd = [
            'rsync',
            '-avz',  # 归档模式，保持属性，压缩传输
            '--partial',  # 保留部分传输的文件
            '--delete',  # 删除目标中源没有的文件
            '--progress',  # 显示进度
            '--info=progress2',  # 更紧凑的进度显示
        ]

        # 添加排除选项
        if exclude_list and len(exclude_list) > 0:
            for item in exclude_list:
                rsync_cmd.extend(['--exclude', item])

        rsync_cmd.extend([
            f"{self.remote_server}:{remote_path}/",  # 添加/表示同步目录内容
            local_dir
        ])

        success = self.retry_subprocess(rsync_cmd, show_progress=True)
        if success:
            print(f"[DOWNLOAD] 目录下载成功: {self.remote_server}:{remote_path} -> {local_dir}")
        else:
            print(f"[DOWNLOAD] 目录下载失败: {self.remote_server}:{remote_path} -> {local_dir}")
        return success

    def mkdir(self, remote_path: str):
        mkdir_cmd = f"mkdir -p {remote_path}"
        ssh_mkdir_cmd = ['ssh', self.remote_server, mkdir_cmd]

        self.retry_subprocess(ssh_mkdir_cmd)

    def mv(self, remote_src_path: str, remote_dest_path: str):
        mv_cmd = f"mv {remote_src_path} {remote_dest_path}"
        ssh_cmd = ['ssh', self.remote_server, mv_cmd]

        self.retry_subprocess(ssh_cmd)

    def rsync_upload_atomic(self, local_path: str, remote_path: str, remote_tmp_path: str, suffix: str = ""):
        assert os.path.isdir(local_path), f"{local_path} is not a directory"

        # "local_path & remote_path have the same folder name"
        # assert os.path.basename(local_path) == os.path.basename(remote_path), f"local_path & remote_path should have the same folder name, but got {os.path.basename(local_path)} & {os.path.basename(remote_path)}"
        # assert os.path.basename(local_path) == os.path.basename(remote_tmp_path), f"local_path & remote_tmp_path should have the same folder name, but got {os.path.basename(local_path)} & {os.path.basename(remote_tmp_path)}"
        self.rsync_upload(local_path, remote_tmp_path)
        remote_path = remote_path.rstrip("/") + suffix
        self.mv(remote_tmp_path, remote_path)

    def ls(self, remote_path: str, pattern: str = None) -> list[str]:
        """列出远程目录内容，可选按模式过滤

        参数:
            remote_path: 远程路径
            pattern: 可选的文件名匹配模式(正则表达式)

        返回:
            匹配的文件/目录列表
        """
        ls_cmd = f"ls {remote_path}"
        ssh_cmd = ['ssh', self.remote_server, ls_cmd]

        try:
            result = subprocess.run(ssh_cmd, check=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)

            items = []
            for line in result.stdout.splitlines():
                item = line.strip()
                if not pattern or re.fullmatch(pattern, item):
                    items.append(item)
            return items

        except subprocess.CalledProcessError as e:
            print(f"Error listing remote directory: {e.stderr}")
            return []

class RemoteCheckpointLoader(mp.Process):
    def __init__(
        self,
        remote_server: str,
        remote_root_dir: str,
        local_tmp_folder: str,
        set_global_model_callback: Callable,
        set_fix_models_callback: Callable,
        local_max_checkpoints: int,
        specify_checkpoint: int = None,
        interval: int = 5,
        dummy: bool = False,
        device: torch.device = torch.device("cuda:0"),
    ):
        super().__init__()
        self.remote_operator = RemoteOperator(remote_server)  # 新增RemoteOperator实例
        self.remote_root_dir = remote_root_dir
        self.local_tmp_folder = local_tmp_folder
        self.set_global_model_callback = set_global_model_callback
        self.set_fix_models_callback = set_fix_models_callback
        self.local_max_checkpoints = local_max_checkpoints
        self.specify_checkpoint = specify_checkpoint
        self.interval = interval
        self.dummy = dummy
        self.device = device

        self.checkpoint_prefix = "checkpoint-"
        os.makedirs(self.local_tmp_folder, exist_ok=True)

        mp_manager = mp.Manager()

        self.current_checkpoint = mp_manager.Value(int, -1)
        self.current_checkpoint_path = mp_manager.Value(str, "")
        self.current_checkpoint_lock = mp.Lock()

        self.daemon = True

        if specify_checkpoint is None:
            # find local available checkpoints
            local_checkpoints = [d for d in os.listdir(self.local_tmp_folder) if d.startswith(self.checkpoint_prefix)]
            if local_checkpoints:
                local_checkpoints.sort(key=lambda x: int(x.replace(self.checkpoint_prefix, "")))
                local_max_checkpoint = int(local_checkpoints[-1].replace(self.checkpoint_prefix, ""))
                print_green(f"Found local checkpoint: {local_checkpoints[-1]}")
                self.set_current_checkpoint(local_max_checkpoint, os.path.join(self.local_tmp_folder, local_checkpoints[-1]))
            else:
                print_yellow("No local checkpoint found, setting to -1")

    def set_current_checkpoint(self, tgt_checkpoint, tgt_checkpoint_path: str):
        with self.current_checkpoint_lock:
            self.current_checkpoint.value = tgt_checkpoint
            self.current_checkpoint_path.value = tgt_checkpoint_path

    def get_current_checkpoint(self):
        with self.current_checkpoint_lock:
            return self.current_checkpoint.value, self.current_checkpoint_path.value

    async def _async_download_file(self, remote_path, local_folder):
        """异步下载文件的辅助方法"""
        # 使用线程池执行器来运行阻塞的rsync操作
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.remote_operator.rsync_download_file(remote_path, local_folder)
        )

    def get_fix_models(self):
        vqvae_encoder_remote_path = os.path.join(self.remote_root_dir, "vq_encoder_jit.pt")
        vqvae_decoder_remote_path = os.path.join(self.remote_root_dir, "vq_decoder_jit.pt")
        dataset_config_remote_path = os.path.join(self.remote_root_dir, "dataset_config.yaml")

        # 确保本地临时目录存在
        os.makedirs(self.local_tmp_folder, exist_ok=True)

        # 构建完整的本地文件路径
        vqvae_encoder_path = os.path.join(self.local_tmp_folder, "vq_encoder_jit.pt")
        vqvae_decoder_path = os.path.join(self.local_tmp_folder, "vq_decoder_jit.pt")
        dataset_config_path = os.path.join(self.local_tmp_folder, "dataset_config.yaml")

        async def download_all_files():
            # 创建下载任务列表
            download_tasks = [
                self._async_download_file(vqvae_encoder_remote_path, self.local_tmp_folder),
                self._async_download_file(vqvae_decoder_remote_path, self.local_tmp_folder),
                self._async_download_file(dataset_config_remote_path, self.local_tmp_folder)
            ]
            # 并发执行所有下载任务
            await asyncio.gather(*download_tasks)

        # 运行协程
        asyncio.run(download_all_files())

        self.set_fix_models_callback(vqvae_encoder_path, vqvae_decoder_path, dataset_config_path)

    def try_to_update_a_new_model(self):
        # if user specified a checkpoint, grab it immediately and exit
        if self.specify_checkpoint is not None:
            curr_checkpoint, _ = self.get_current_checkpoint()
            if curr_checkpoint == self.specify_checkpoint:
                return

            cp = self.specify_checkpoint
            print_yellow(f"Downloading specified checkpoint: {cp}")
            ckpt_name = f"{self.checkpoint_prefix}{cp}"
            remote_path = os.path.join(self.remote_root_dir, ckpt_name)
            local_path  = os.path.join(self.local_tmp_folder, ckpt_name)
            # pull down only that one
            self.remote_operator.rsync_download_dir(
                remote_path, local_path, ["*.pth", "*.pt", "*.bin"]
            )
            self.set_current_checkpoint(cp, local_path)
            self.set_global_model_callback(local_path)
            return

        while True:
            current_checkpoint, current_checkpoint_path = self.get_current_checkpoint()
            if current_checkpoint != -1 and os.path.exists(current_checkpoint_path):
                print(f"Found valid checkpoint: {current_checkpoint}")
                break

            print("Waiting for a valid checkpoint...")
            time.sleep(self.interval)

        self.set_global_model_callback(current_checkpoint_path)

    def _check_for_new_checkpoint(self):
        # Get list of checkpoint folders on remote server with pattern matching
        pattern = f"^{self.checkpoint_prefix}[0-9]+"  # 匹配以checkpoint-开头后跟数字的格式
        remote_folders = self.remote_operator.ls(self.remote_root_dir, pattern=pattern)
        if not remote_folders:
            return

        # Find the maximum checkpoint number
        max_checkpoint = max(int(folder.replace(self.checkpoint_prefix, "")) for folder in remote_folders)

        current_checkpoint, _ = self.get_current_checkpoint()

        # If we found a newer checkpoint
        if max_checkpoint > current_checkpoint:
            print(f"Found new checkpoint: {max_checkpoint}")
            checkpoint_dir_name = f'{self.checkpoint_prefix}{max_checkpoint}'
            remote_path = os.path.join(self.remote_root_dir, checkpoint_dir_name)
            local_path = os.path.join(self.local_tmp_folder, checkpoint_dir_name)
            self.remote_operator.rsync_download_dir(remote_path, local_path, ["*.pth", "*.pt", "*.bin"])
            print(f"Downloaded checkpoint {max_checkpoint} successfully")

            self.set_current_checkpoint(max_checkpoint, local_path)

        print(f"No new checkpoint found. Current checkpoint: {current_checkpoint}")

    def _clean_checkpoints(self):
        """
        Delete the oldest local checkpoint if total local checkpoints > local_max_checkpoints
        """
        current_checkpoint, _ = self.get_current_checkpoint()
        if current_checkpoint == -1:
            return

        # List all checkpoint folders in the local tmp folder
        local_checkpoints = [d for d in os.listdir(self.local_tmp_folder) if d.startswith(self.checkpoint_prefix)]
        local_checkpoints.sort(key=lambda x: int(x.replace(self.checkpoint_prefix, "")))
        # If the number of local checkpoints exceeds the limit, delete the oldest one
        if len(local_checkpoints) > self.local_max_checkpoints:
            oldest_checkpoint = local_checkpoints[0]
            oldest_checkpoint_path = os.path.join(self.local_tmp_folder, oldest_checkpoint)
            print_blue(f"Deleting oldest checkpoint: {oldest_checkpoint_path}")
            shutil.rmtree(oldest_checkpoint_path, ignore_errors=True)

    def run(self):
        while True:
            try:
                if self.dummy:
                    print_yellow("Dummy mode: skipping model fetching")
                else:
                    self._check_for_new_checkpoint()
                    self._clean_checkpoints()
            except Exception as e:
                print(f"Error during model fetching: {str(e)}")
            time.sleep(self.interval)

class RemoteLocalMaintainerWait(mp.Process):
    def __init__(self, remote_server, remote_data_root_dir, remote_dir, remote_tmp_dir, local_folder, interval=5, delete_local=True):
        super().__init__()
        self.remote_operator = RemoteOperator(remote_server)

        self.remote_data_root_dir = remote_data_root_dir
        self.remote_dir = remote_dir
        self.remote_tmp_dir = remote_tmp_dir
        self.local_folder = local_folder
        self.interval = interval
        self.delete_local = delete_local

        self.daemon = True

        mp_manager = mp.Manager()

        self.cur_model_id_lock = mp.Lock()
        self.cur_model_id = mp_manager.Value(int, -1)

        # 第二级目录的可能值
        self.second_level_dirs = ["intervention", "success", "failure"]

    def set_cur_model_id(self, model_id):
        with self.cur_model_id_lock:
            self.cur_model_id.value = model_id

    def get_cur_model_id(self):
        with self.cur_model_id_lock:
            return self.cur_model_id.value

    def run(self):
        model_id_str = str(self.get_cur_model_id())

        assert model_id_str != "-1", "cur_model_id should be set before start"

        """进程主循环，定期检查本地文件夹并同步到远程服务器"""
        while True:
            try:
                self.sync_local_to_remote()
            except Exception as e:
                print(f"同步过程中发生错误: {str(e)}")

            time.sleep(self.interval)

    def sync_local_to_remote(self):
        model_id_str = str(self.get_cur_model_id())

        assert model_id_str != "-1", "cur_model_id should be set before start"

        """同步本地文件夹到远程服务器，处理三级目录结构"""
        # 检查是否存在 {self.cur_model_id}.ready 目录
        ready_dir = f"{model_id_str}"

        # 检查远程是否存在 .ready 目录
        if self.remote_operator.ls(self.remote_data_root_dir, pattern=ready_dir):
            print(f"发现 {ready_dir} 目录，跳过同步操作")
            return

        if not os.path.exists(self.local_folder):
            print(f"本地文件夹不存在: {self.local_folder}")
            return

        # 获取本地文件夹中的所有第一级子目录
        first_level_dirs = os.listdir(self.local_folder)
        if not first_level_dirs:
            return

        # 只处理与当前 model_id 匹配的目录

        # 遍历第一级目录，只处理与当前 model_id 匹配的目录
        if model_id_str in first_level_dirs:
            first_dir = model_id_str
            first_dir_path = os.path.join(self.local_folder, first_dir)
            if not os.path.isdir(first_dir_path):
                return

            # 遍历第二级目录 (intervention, success, failure)
            for second_dir in self.second_level_dirs:
                second_dir_path = os.path.join(first_dir_path, second_dir)
                if not os.path.exists(second_dir_path) or not os.path.isdir(second_dir_path):
                    continue

                # 遍历第三级目录（实际需要同步的目录）
                third_level_dirs = os.listdir(second_dir_path)
                for third_dir in third_level_dirs:
                    local_item_path = os.path.join(second_dir_path, third_dir)
                    if not os.path.isdir(local_item_path):
                        continue

                    # 构建远程对应的路径结构
                    remote_rel_path = os.path.join(first_dir, second_dir, third_dir)
                    remote_path = os.path.join(self.remote_dir, remote_rel_path)
                    remote_tmp_path = os.path.join(self.remote_tmp_dir, remote_rel_path)

                    # 检查远程是否已存在此目录
                    remote_parent_dir = os.path.join(self.remote_dir, first_dir, second_dir)
                    remote_items = self.remote_operator.ls(remote_parent_dir)

                    if third_dir in remote_items:
                        # 如果远程已存在且需要删除本地，则删除本地
                        if self.delete_local:
                            print(f"[SYNC] 远程已存在，删除本地: {local_item_path}")
                            shutil.rmtree(local_item_path, ignore_errors=True)
                        continue

                    # 同步到远程
                    print(f"[SYNC] 开始同步: {local_item_path} -> {remote_path}")

                    # 确保远程临时目录的父目录存在
                    remote_tmp_parent = os.path.dirname(remote_tmp_path)
                    self.remote_operator.mkdir(remote_tmp_parent)

                    # 先上传到临时目录
                    success = self.remote_operator.rsync_upload(
                        local_path=local_item_path,
                        remote_path=remote_tmp_path,
                        delete_local=False  # 不在这里删除，等重命名成功后再删除
                    )

                    if success:
                        # 确保远程目标目录的父目录存在
                        remote_parent = os.path.dirname(remote_path)
                        self.remote_operator.mkdir(remote_parent)

                        # 原子重命名操作
                        self.remote_operator.mv(remote_tmp_path, remote_path)
                        print(f"[SYNC] 同步成功: {local_item_path} -> {remote_path}")

                        # 如果需要删除本地，则删除
                        if self.delete_local:
                            shutil.rmtree(local_item_path, ignore_errors=True)
                            print(f"[SYNC] 本地文件已删除: {local_item_path}")
                    else:
                        print(f"[SYNC] 同步失败: {local_item_path} -> {remote_path}")

    def get_remote_data_counts(self):
        """获取当前 model_id 目录下三种类型数据的条数

        首先检查 {remote_data_root_dir}/{self.cur_model_id}.ready 目录是否存在，
        如果存在则统计该目录下的数据，否则统计 {remote_dir}/{self.cur_model_id} 目录下的数据。

        返回:
            tuple: 包含三个整数的元组 (intervention_count, success_count, failure_count)
        """
        model_id_str = str(self.get_cur_model_id())

        assert model_id_str != "-1", "cur_model_id should be set before start"

        intervention_count = 0
        success_count = 0
        failure_count = 0

        # 首先检查 {remote_data_root_dir}/{self.cur_model_id}.ready 目录是否存在
        ready_dir = f"{model_id_str}"
        ready_dir_path = os.path.join(self.remote_data_root_dir, ready_dir)

        # 检查远程是否存在 .ready 目录
        if self.remote_operator.ls(self.remote_data_root_dir, pattern=ready_dir):
            print(f"发现 {ready_dir} 目录，从该目录统计数据")
            first_dir_path = ready_dir_path
        else:
            # 使用当前的 model_id 作为第一级目录
            first_dir = str(model_id_str)
            first_dir_path = os.path.join(self.remote_dir, first_dir)

        if not self.remote_operator.ls(first_dir_path):  # 检查目录是否存在
            print(f"远程目录不存在: {first_dir_path}")
            return intervention_count, success_count, failure_count

        # 遍历第二级目录
        for second_dir in self.second_level_dirs:
            second_dir_path = os.path.join(first_dir_path, second_dir)

            # 获取该类型的数据条数
            try:
                items = self.remote_operator.ls(second_dir_path)
                count = len(items) if items else 0

                # 根据类型更新对应的计数
                if second_dir == "intervention":
                    intervention_count = count
                elif second_dir == "success":
                    success_count = count
                elif second_dir == "failure":
                    failure_count = count

            except Exception as e:
                print(f"获取 {second_dir_path} 数据条数时出错: {str(e)}")

        print_green(f"远程目录 {first_dir_path} 下的数据条数: intervention={intervention_count}, success={success_count}, failure={failure_count}")

        return intervention_count, success_count, failure_count



class RemoteLocalMaintainerNoWait(mp.Process):
    def __init__(self, remote_server, remote_data_root_dir, remote_tmp_dir, local_folder, interval=5, delete_local=True, sft=False):
        super().__init__()
        self.remote_operator = RemoteOperator(remote_server)

        self.remote_data_root_dir = remote_data_root_dir
        self.remote_tmp_dir = remote_tmp_dir
        self.local_folder = local_folder
        self.interval = interval
        self.delete_local = delete_local
        self.sft = sft

        self.daemon = True

        mp_manager = mp.Manager()

        self.cur_model_id_lock = mp.Lock()
        self.cur_model_id = mp_manager.Value(int, -1 if not sft else 0)

        self.second_level_dirs = ["intervention", "success", "failure"]

    def set_cur_model_id(self, model_id):
        with self.cur_model_id_lock:
            self.cur_model_id.value = model_id

    def get_cur_model_id(self):
        with self.cur_model_id_lock:
            return self.cur_model_id.value

    def run(self):
        model_id_str = str(self.get_cur_model_id())

        assert model_id_str != "-1", "cur_model_id should be set before start"

        """进程主循环，定期检查本地文件夹并同步到远程服务器"""
        while True:
            try:
                self.sync_local_to_remote()
            except Exception as e:
                print(f"同步过程中发生错误: {str(e)}")

            time.sleep(self.interval)

    def sync_local_to_remote(self):
        model_id_str = str(self.get_cur_model_id())

        assert model_id_str != "-1", "cur_model_id should be set before start"

        if not os.path.exists(self.local_folder):
            print(f"本地文件夹不存在: {self.local_folder}")
            return

        first_dir = model_id_str

        first_dir_path = os.path.join(self.local_folder, first_dir)
        if not os.path.exists(first_dir_path) or not os.path.isdir(first_dir_path):
            print(f"本地目录不存在: {first_dir_path}")
            return

        # 遍历第二级目录 (intervention, success, failure)
        for second_dir in self.second_level_dirs:
            second_dir_path = os.path.join(first_dir_path, second_dir)
            if not os.path.exists(second_dir_path) or not os.path.isdir(second_dir_path):
                continue

            # 遍历第三级目录（实际需要同步的目录）
            third_level_dirs = os.listdir(second_dir_path)
            for third_dir in third_level_dirs:
                local_item_path = os.path.join(second_dir_path, third_dir)
                if not os.path.isdir(local_item_path):
                    continue

                # 构建远程对应的路径结构
                remote_rel_path = os.path.join(first_dir, second_dir, third_dir)
                remote_path = os.path.join(self.remote_data_root_dir, remote_rel_path)
                remote_tmp_path = os.path.join(self.remote_tmp_dir, remote_rel_path)

                # 检查远程是否已存在此目录
                remote_parent_dir = os.path.join(self.remote_data_root_dir, first_dir, second_dir)
                remote_items = self.remote_operator.ls(remote_parent_dir)

                if third_dir in remote_items:
                    # 如果远程已存在且需要删除本地，则删除本地
                    if self.delete_local:
                        print(f"[SYNC] 远程已存在，删除本地: {local_item_path}")
                        shutil.rmtree(local_item_path, ignore_errors=True)
                    continue

                # 同步到远程
                print(f"[SYNC] 开始同步: {local_item_path} -> {remote_path}")

                # 确保远程临时目录的父目录存在
                remote_tmp_parent = os.path.dirname(remote_tmp_path)
                self.remote_operator.mkdir(remote_tmp_parent)

                # 先上传到临时目录
                success = self.remote_operator.rsync_upload(
                    local_path=local_item_path,
                    remote_path=remote_tmp_path,
                    delete_local=False  # 不在这里删除，等重命名成功后再删除
                )

                if success:
                    # 确保远程目标目录的父目录存在
                    remote_parent = os.path.dirname(remote_path)
                    self.remote_operator.mkdir(remote_parent)

                    # 原子重命名操作
                    self.remote_operator.mv(remote_tmp_path, remote_path)
                    print(f"[SYNC] 同步成功: {local_item_path} -> {remote_path}")

                    # 如果需要删除本地，则删除
                    if self.delete_local:
                        shutil.rmtree(local_item_path, ignore_errors=True)
                        print(f"[SYNC] 本地文件已删除: {local_item_path}")
                else:
                    print(f"[SYNC] 同步失败: {local_item_path} -> {remote_path}")
