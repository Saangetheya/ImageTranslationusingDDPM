from typing import List, Optional, Tuple, Union

import torch
import numpy as np

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput


class MRIs3DDDPMPipeline(DiffusionPipeline):
    r"""
    Pipeline for 3D temporal MRI image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet3DModel`]):
            A `UNet3DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
            self,
            t1_volume: torch.Tensor,
            diagnosis_id: torch.Tensor = None,
            batch_size: int = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            num_inference_steps: int = 1000,
            output_type: Optional[str] = "np",
            return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            t1_volume (`torch.Tensor`):
                The input T1 MRI volume with shape [B, C, T, H, W] where T is the temporal dimension.
            batch_size (`int`, *optional*, defaults to 1):
                The number of volumes to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `np.array` or `torch.Tensor`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated volumes.
        """
        # Extract input shape information
        _, _, temporal_length, height, width = t1_volume.shape

        # Create initial noise
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels // 2,  # Subtract conditional input channels
                temporal_length,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (
                batch_size,
                self.unet.config.in_channels // 2,
                temporal_length,
                *self.unet.config.sample_size
            )

        if self.device.type == "mps":
            # randn is not reproducible on MPS
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)

        # Set denoising steps
        self.scheduler.set_timesteps(num_inference_steps)

        # Denoising process
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. Concatenate conditional input and noisy image
            model_input = torch.cat([image, t1_volume], dim=1)

            # 2. Predict noise using the model
            # model_output = self.unet(model_input, t, class_labels=diagnosis_id).sample
            if diagnosis_id is not None:
                diagnosis_id = diagnosis_id.long().to(self.device)
            model_output = self.unet(model_input, t, class_labels=diagnosis_id).sample

            # 3. Compute the previous timestep image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # Post-process the generated image
        image = (image / 2 + 0.5).clamp(0, 1)

        if output_type == "np":
            image = image.cpu().numpy()

        if not return_dict:
            return (image,)

        # Still using ImagePipelineOutput, though it now contains volumetric data
        return ImagePipelineOutput(images=image)

    def save_volume(self, volume, save_path, file_prefix="dwi_slice"):
        """
        Save the generated 3D volume as a series of 2D slice images.

        Args:
            volume (np.ndarray): 3D volume data with shape [C, T, H, W]
            save_path (str): Directory to save images
            file_prefix (str): Filename prefix
        """
        import os
        import cv2

        os.makedirs(save_path, exist_ok=True)

        # Handle volume data in numpy array format
        if isinstance(volume, np.ndarray):
            if volume.ndim == 5:  # [B, C, T, H, W]
                volume = volume[0]  # Use the first batch
            # Convert [C, T, H, W] -> [T, H, W, C] for image saving
            volume = np.transpose(volume, (1, 2, 3, 0))
        else:  # Handle tensor volume data
            if volume.dim() == 5:  # [B, C, T, H, W]
                volume = volume[0]  # Use the first batch
            volume = volume.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]

        # Save each temporal slice
        for t in range(volume.shape[0]):
            slice_img = volume[t]  # [H, W, C]

            # Ensure values are in 0-255 range
            if slice_img.max() <= 1.0:
                slice_img = (slice_img * 255).astype(np.uint8)

            # Convert to 3 channels if single channel
            if slice_img.shape[-1] == 1:
                slice_img = np.repeat(slice_img, 3, axis=-1)

            # Save the image
            cv2.imwrite(os.path.join(save_path, f"{file_prefix}_{t:03d}.png"), slice_img)
