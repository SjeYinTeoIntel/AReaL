# SPDX-License-Identifier: Apache-2.0

from vllm.v1.worker.xpu_worker import XPUWorker

from areal.infra.platforms import current_platform


class ArealXPUWorker(XPUWorker):
    """vLLM XPU worker that runs on the devices the scheduler assigned.

    AReaL workers on XPU see every device of the node, so the device
    visibility mask cannot steer vLLM to its own devices. Shift the worker's
    local rank to the first assigned device instead.
    """

    def init_device(self):
        self.local_rank = current_platform.local_device_index(self.local_rank)
        super().init_device()
