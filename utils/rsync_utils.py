import threading
import subprocess
import os
import time
import shutil

from .remote_utils import RemoteOperator


class SyncFolderThread(threading.Thread):
    def __init__(self, local_path: str, remote_tmp_path: str, remote_server: str, delete_local: bool, max_retries=3, retry_delay=5):
        threading.Thread.__init__(self)
        self.remote_operator = RemoteOperator(
            remote_server=remote_server,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self.local_path = local_path
        self.remote_tmp_path = remote_tmp_path
        self.delete_local = delete_local

        self.remote_server = remote_server
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.daemon = True

    def run(self):
        self.remote_operator.rsync_upload(self.local_path, self.remote_tmp_path, delete_local=self.delete_local)

        print(f"同步完成: {self.local_path} -> {self.remote_tmp_path}")

class AtomicRenameThread(threading.Thread):
    def __init__(self, remote_tmp_path: str, remote_path: str, remote_server: str, suffix: str, max_retries=3, retry_delay=5):
        threading.Thread.__init__(self)
        self.remote_operator = RemoteOperator(
            remote_server=remote_server,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        self.remote_tmp_path = remote_tmp_path
        self.remote_path = remote_path.rstrip('/') + suffix

        self.remote_server = remote_server
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.daemon = True

    def run(self):
        # 检查目标路径是否已存在
        remote_parent_dir = os.path.dirname(self.remote_path)
        remote_basename = os.path.basename(self.remote_path)

        # 确保父目录存在
        self.remote_operator.mkdir(remote_parent_dir)

        # 检查目标文件是否已存在
        existing_files = self.remote_operator.ls(remote_parent_dir)
        if remote_basename in existing_files:
            print(f"目标路径已存在，跳过重命名: {self.remote_path}")
            return

        # 执行重命名操作
        self.remote_operator.mv(self.remote_tmp_path, self.remote_path)

        print(f"原子重命名完成: {self.remote_tmp_path} -> {self.remote_path}")

