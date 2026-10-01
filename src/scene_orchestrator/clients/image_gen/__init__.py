"""Skyboxes from image-gen on Pegasus: prompt + seed in, skybox files out.

    from scene_orchestrator.clients.image_gen import SkyboxRequest, SlurmImageGenClient

    client = SlurmImageGenClient(config.image_gen)
    result = client.generate_skybox(
        SkyboxRequest(prompt="a calm forest", seed=1234, scene_id="p07"), out_dir
    )
"""

from .base import ImageGenError, ImageGenStatus, SkyboxRequest, SkyboxResult
from .slurm import PegasusShell, SlurmImageGenClient

__all__ = [
    "ImageGenError",
    "ImageGenStatus",
    "PegasusShell",
    "SkyboxRequest",
    "SkyboxResult",
    "SlurmImageGenClient",
]
