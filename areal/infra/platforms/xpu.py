# SPDX-License-Identifier: Apache-2.0

import gc
import os

import torch

from .platform import Platform


class XpuPlatform(Platform):
    device_name: str = "Intel XPU"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    ray_device_key: str = "XPU"
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    ray_experimental_noset: str = ""
    communication_backend: str = "xccl"
    # Index of the worker's first device among the devices it sees.
    device_offset_env_var: str = "AREAL_XPU_DEVICE_OFFSET"

    def clear_memory(self) -> None:
        self.synchronize()
        gc.collect()
        torch.xpu.empty_cache()

    @classmethod
    def clear_cublas_workspaces(cls) -> None:
        pass

    @classmethod
    def set_allocator_settings(cls) -> None:
        pass

    @classmethod
    def get_custom_env_vars(cls) -> dict:
        return {}

    @classmethod
    def update_env_vars_for_visible_devices(
        cls, env_vars: dict, gpu_ranks: list
    ) -> None:
        env_vars[cls.device_control_env_var] = ",".join(map(str, gpu_ranks))

    @classmethod
    def get_worker_device_env_vars(
        cls, gpu_devices: list[int], node_gpu_devices: list[int]
    ) -> dict[str, str]:
        # oneCCL can only connect processes that each see every device of the
        # communicator: with one device per process, the XCCL weight update
        # between the trainer and the inference server never completes. So
        # every worker sees the whole node and picks its device by index.
        if not gpu_devices:
            return super().get_worker_device_env_vars(gpu_devices, node_gpu_devices)
        # Level Zero lists devices in ascending order whatever the mask order.
        visible = sorted(set(node_gpu_devices) | set(gpu_devices))
        indices = [visible.index(device) for device in gpu_devices]
        if indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError(
                f"XPU workers need consecutive devices, but got {gpu_devices} "
                f"out of {visible}."
            )
        return {
            cls.device_control_env_var: ",".join(map(str, visible)),
            cls.device_offset_env_var: str(indices[0]),
        }

    @classmethod
    def local_device_index(cls, local_rank: int) -> int:
        return int(os.environ.get(cls.device_offset_env_var, "0")) + local_rank

    @classmethod
    def synchronize(cls) -> None:
        torch.xpu.synchronize()
