# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from typing import List
from omni_zero import OmniZeroCouple
from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.omni_zero = OmniZeroCouple(
            base_model="frankjoshua/albedobaseXL_v13",
        )
    def predict(
        self,
        base_image: Path = Input(description="Base image for the model", default=None),
        base_image_strength: float = Input(description="Base image strength for the model", default=0.2, ge=0.0, le=1.0),
        style_image: Path = Input(description="Style image for the model", default=None),
        style_image_strength: float = Input(description="Style image strength for the model", default=1.0, ge=0.0, le=1.0),
        identity_image_1: Path = Input(description="First identity image for the model", default=None),
        identity_image_strength_1: float = Input(description="First identity image strength for the model", default=1.0, ge=0.0, le=1.0),
        identity_image_2: Path = Input(description="Second identity image for the model", default=None),
        identity_image_strength_2: float = Input(description="Second identity image strength for the model", default=1.0, ge=0.0, le=1.0),
        seed: int = Input(description="Random seed for the model. Use -1 for random", default=-1),
        prompt: str = Input(description="Prompt for the model", default="Cinematic still photo of a couple. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy"),
        negative_prompt: str = Input(description="Negative prompt for the model", default="anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured"),
        guidance_scale: float = Input(description="Guidance scale for the model", default=3.0, ge=0.0, le=14.0),
        number_of_images: int = Input(description="Number of images to generate", default=1, ge=1, le=4),
        number_of_steps: int = Input(description="Number of steps for the model", default=10, ge=1, le=50),
        depth_image: Path = Input(description="Depth image for the model", default=None),
        depth_image_strength: float = Input(description="Depth image strength for the model", default=0.2, ge=0.0, le=1.0),
        mask_guidance_start: float = Input(description="Mask guidance start value", default=0.0, ge=0.0, le=1.0),
        mask_guidance_end: float = Input(description="Mask guidance end value", default=1.0, ge=0.0, le=1.0),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        base_image = Image.open(base_image) if base_image else None
        style_image = Image.open(style_image) if style_image else None
        identity_image_1 = Image.open(identity_image_1) if identity_image_1 else None
        identity_image_2 = Image.open(identity_image_2) if identity_image_2 else None
        depth_image = Image.open(depth_image) if depth_image else None

        images = self.omni_zero.generate(
            seed=seed,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            number_of_images=number_of_images,
            number_of_steps=number_of_steps,
            base_image=base_image,
            base_image_strength=base_image_strength,
            style_image=style_image,
            style_image_strength=style_image_strength,
            identity_image_1=identity_image_1,
            identity_image_strength_1=identity_image_strength_1,
            identity_image_2=identity_image_2,
            identity_image_strength_2=identity_image_strength_2,
            depth_image=depth_image,
            depth_image_strength=depth_image_strength,
            mask_guidance_start=mask_guidance_start,
            mask_guidance_end=mask_guidance_end,
        )
        
        outputs = []
        for i, image in enumerate(images):
            output_path = f"oz_output_{i}.jpg"
            image.save(output_path)
            outputs.append(Path(output_path))
        
        return outputs