# SPDX-License-Identifier: Apache-2.0
"""Ray must request non-NVIDIA accelerators as custom resources, not ``num_gpus``."""

import pytest

from areal.infra.launcher.ray import accelerator_resource_kwargs
from areal.infra.platforms.cuda import CudaPlatform
from areal.infra.platforms.npu import NPUPlatform
from areal.infra.platforms.xpu import XpuPlatform


def _patch_platform(monkeypatch, platform_cls):
    """Point the ray launcher's ``current_platform`` proxy at ``platform_cls``."""
    import areal.infra.launcher.ray as ray_launcher

    monkeypatch.setattr(ray_launcher, "current_platform", platform_cls)


def test_cuda_uses_num_gpus(monkeypatch):
    _patch_platform(monkeypatch, CudaPlatform)

    assert accelerator_resource_kwargs(4) == {"num_gpus": 4}


def test_xpu_uses_custom_resource(monkeypatch):
    _patch_platform(monkeypatch, XpuPlatform)

    assert accelerator_resource_kwargs(2) == {"resources": {"XPU": 2}}


def test_npu_uses_custom_resource(monkeypatch):
    """Preserves the behaviour the previous hardcoded NPU branch provided."""
    _patch_platform(monkeypatch, NPUPlatform)

    assert accelerator_resource_kwargs(8) == {"resources": {"NPU": 8}}


@pytest.mark.parametrize("platform_cls", [CudaPlatform, XpuPlatform, NPUPlatform])
def test_zero_accelerators_is_representable(monkeypatch, platform_cls):
    """CPU-only jobs pass gpus=0; the request must stay well-formed."""
    _patch_platform(monkeypatch, platform_cls)

    kwargs = accelerator_resource_kwargs(0)

    assert kwargs in ({"num_gpus": 0}, {"resources": {platform_cls.ray_device_key: 0}})
