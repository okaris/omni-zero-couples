import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import sys

sys.path.insert(0, './diffusers/src')

import cv2
import numpy as np
import PIL
import torch
from controlnet_aux import ZoeDetector
from diffusers import DPMSolverMultistepScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models import ControlNetModel
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from pipeline import OmniZeroPipeline
from transformers import CLIPVisionModelWithProjection
from utils import align_images, draw_kps, load_and_resize_image
import random

class OmniZeroSingle():
    def __init__(self,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
    ):
        snapshot_download("okaris/antelopev2", local_dir="./models/antelopev2")
        self.face_analysis = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        dtype = torch.float16

        ip_adapter_plus_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        ).to(device)

        zoedepthnet_path = "okaris/zoe-depth-controlnet-xl"
        zoedepthnet = ControlNetModel.from_pretrained(zoedepthnet_path,torch_dtype=dtype).to(device)

        identitiynet_path = "okaris/face-controlnet-xl"
        identitynet = ControlNetModel.from_pretrained(identitiynet_path, torch_dtype=dtype).to(device)

        self.zoe_depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to(device)

        self.pipeline = OmniZeroPipeline.from_pretrained(
            base_model,
            controlnet=[identitynet, zoedepthnet],
            torch_dtype=dtype,
            image_encoder=ip_adapter_plus_image_encoder,
        ).to(device)

        config = self.pipeline.scheduler.config
        config["timestep_spacing"] = "trailing"
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", final_sigmas_type="zero")

        self.pipeline.load_ip_adapter(["okaris/ip-adapter-instantid", "h94/IP-Adapter", "h94/IP-Adapter"], subfolder=[None, "sdxl_models", "sdxl_models"], weight_name=["ip-adapter-instantid.bin", "ip-adapter-plus_sdxl_vit-h.safetensors", "ip-adapter-plus_sdxl_vit-h.safetensors"])
   
    def get_largest_face_embedding_and_kps(self, image, target_image=None):
        face_info = self.face_analysis.get(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if len(face_info) == 0:
            return None, None
        largest_face = sorted(face_info, key=lambda x: x['bbox'][2] * x['bbox'][3], reverse=True)[0]
        face_embedding = torch.tensor(largest_face['embedding']).to("cuda")
        if target_image is None:
            target_image = image
        zeros = np.zeros((target_image.size[1], target_image.size[0], 3), dtype=np.uint8)
        face_kps_image = draw_kps(zeros, largest_face['kps'])
        return face_embedding, face_kps_image
    
    def generate(self,
        seed=42,
        prompt="A person",
        negative_prompt="blurry, out of focus",
        guidance_scale=3.0,
        number_of_images=1,
        number_of_steps=10,
        base_image=None,
        base_image_strength=0.15,
        composition_image=None,
        composition_image_strength=1.0,
        style_image=None,
        style_image_strength=1.0,
        identity_image=None,
        identity_image_strength=1.0,
        depth_image=None,
        depth_image_strength=0.5,        
    ):
        resolution = 1024

        if base_image is not None:
            base_image = load_and_resize_image(base_image, resolution, resolution)
        else:
            if composition_image is not None:
                base_image = load_and_resize_image(composition_image, resolution, resolution)
            else:
                raise ValueError("You must provide a base image or a composition image")

        if depth_image is None:
            depth_image = self.zoe_depth_detector(base_image, detect_resolution=resolution, image_resolution=resolution)
        else:
            depth_image = load_and_resize_image(depth_image, resolution, resolution)

        base_image, depth_image = align_images(base_image, depth_image)

        if composition_image is not None:
            composition_image = load_and_resize_image(composition_image, resolution, resolution)
        else: 
            composition_image = base_image

        if style_image is not None:
            style_image = load_and_resize_image(style_image, resolution, resolution)
        else:
            raise ValueError("You must provide a style image")
        
        if identity_image is not None:
            identity_image = load_and_resize_image(identity_image, resolution, resolution)
        else:
            raise ValueError("You must provide an identity image")
        
        face_embedding_identity_image, target_kps = self.get_largest_face_embedding_and_kps(identity_image, base_image)
        if face_embedding_identity_image is None:
            raise ValueError("No face found in the identity image, the image might be cropped too tightly or the face is too small")
        
        face_embedding_base_image, face_kps_base_image = self.get_largest_face_embedding_and_kps(base_image)
        if face_embedding_base_image is not None:
            target_kps = face_kps_base_image

        self.pipeline.set_ip_adapter_scale([identity_image_strength,
            {
                "down": { "block_2": [0.0, 0.0] },
                "up": { "block_0": [0.0, style_image_strength, 0.0] }
            },
            {
                "down": { "block_2": [0.0, composition_image_strength] },
                "up": { "block_0": [0.0, 0.0, 0.0] }
            }
        ])

        generator = torch.Generator(device="cpu").manual_seed(seed)

        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            ip_adapter_image=[face_embedding_identity_image, style_image, composition_image],
            image=base_image,
            control_image=[target_kps, depth_image],
            controlnet_conditioning_scale=[identity_image_strength, depth_image_strength],
            identity_control_indices=[(0,0)],
            num_inference_steps=number_of_steps, 
            num_images_per_prompt=number_of_images,
            strength=(1-base_image_strength),
            generator=generator,
            seed=seed,
        ).images

        return images
    
class OmniZeroCouple():
    def __init__(self,
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        device="cuda",
    ):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        self.patch_onnx_runtime()

        snapshot_download("okaris/antelopev2", local_dir="./models/antelopev2")
        self.face_analysis = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.face_analysis.prepare(ctx_id=0, det_size=(640, 640))

        self.dtype = dtype = torch.float16

        ip_adapter_plus_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter", 
            subfolder="models/image_encoder",
            torch_dtype=dtype,
        ).to(device)

        zoedepthnet_path = "okaris/zoe-depth-controlnet-xl"
        zoedepthnet = ControlNetModel.from_pretrained(zoedepthnet_path,torch_dtype=dtype).to(device)

        identitiynet_path = "okaris/face-controlnet-xl"
        identitynet = ControlNetModel.from_pretrained(identitiynet_path, torch_dtype=dtype).to(device)

        self.zoe_depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to(device)
        self.ip_adapter_mask_processor = IPAdapterMaskProcessor()

        self.pipeline = OmniZeroPipeline.from_pretrained(
            base_model,
            controlnet=[identitynet, identitynet, zoedepthnet],
            torch_dtype=dtype,
            image_encoder=ip_adapter_plus_image_encoder,
        ).to(device)

        config = self.pipeline.scheduler.config
        config["timestep_spacing"] = "trailing"
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", final_sigmas_type="zero")

        self.pipeline.load_ip_adapter(["okaris/ip-adapter-instantid", "okaris/ip-adapter-instantid", "h94/IP-Adapter"], subfolder=[None, None, "sdxl_models"], weight_name=["ip-adapter-instantid.bin", "ip-adapter-instantid.bin", "ip-adapter-plus_sdxl_vit-h.safetensors"])
   
    def generate(self,
        seed=42,
        prompt="A person",
        negative_prompt="blurry, out of focus",
        guidance_scale=3.0,
        number_of_images=1,
        number_of_steps=10,
        base_image=None,
        base_image_strength=0.2,
        style_image=None,
        style_image_strength=1.0,
        identity_image_1=None,
        identity_image_strength_1=1.0,
        identity_image_2=None,
        identity_image_strength_2=1.0,
        depth_image=None,
        depth_image_strength=0.5,
        mask_guidance_start=0.0,
        mask_guidance_end=1.0,      
    ):

        if seed == -1:
            seed = random.randint(0, 1000000)

        resolution = 1024

        if base_image is not None:
            base_image = load_and_resize_image(base_image, resolution, resolution)

        if depth_image is None:
            depth_image = self.zoe_depth_detector(base_image, detect_resolution=resolution, image_resolution=resolution)
        else:
            depth_image = load_and_resize_image(depth_image, resolution, resolution)

        base_image, depth_image = align_images(base_image, depth_image)

        if style_image is not None:
            style_image = load_and_resize_image(style_image, resolution, resolution)
        else:
            raise ValueError("You must provide a style image")
        
        if identity_image_1 is not None:
            identity_image_1 = load_and_resize_image(identity_image_1, resolution, resolution)
        else:
            raise ValueError("You must provide an identity image")
        
        if identity_image_2 is not None:
            identity_image_2 = load_and_resize_image(identity_image_2, resolution, resolution)
        else:
            raise ValueError("You must provide an identity image 2")

        height, width = base_image.size

        face_info_1 = self.face_analysis.get(cv2.cvtColor(np.array(identity_image_1), cv2.COLOR_RGB2BGR))
        for i, face in enumerate(face_info_1):
            print(f"Face 1 -{i}: Age: {face['age']}, Gender: {face['gender']}")
        face_info_1 = sorted(face_info_1, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb_1 = torch.tensor(face_info_1['embedding']).to("cuda", dtype=self.dtype)

        face_info_2 = self.face_analysis.get(cv2.cvtColor(np.array(identity_image_2), cv2.COLOR_RGB2BGR))
        for i, face in enumerate(face_info_2):
            print(f"Face 2 -{i}: Age: {face['age']}, Gender: {face['gender']}")
        face_info_2 = sorted(face_info_2, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
        face_emb_2 = torch.tensor(face_info_2['embedding']).to("cuda", dtype=self.dtype)

        zero = np.zeros((width, height, 3), dtype=np.uint8)
        # face_kps_identity_image_1 = self.draw_kps(zero, face_info_1['kps'])
        # face_kps_identity_image_2 = self.draw_kps(zero, face_info_2['kps'])

        face_info_img2img = self.face_analysis.get(cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR))
        faces_info_img2img = sorted(face_info_img2img, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])
        face_info_a = faces_info_img2img[-1]
        face_info_b = faces_info_img2img[-2]
        # face_emb_a = torch.tensor(face_info_a['embedding']).to("cuda", dtype=self.dtype)
        # face_emb_b = torch.tensor(face_info_b['embedding']).to("cuda", dtype=self.dtype)
        face_kps_identity_image_a = draw_kps(zero, face_info_a['kps'])
        face_kps_identity_image_b = draw_kps(zero, face_info_b['kps'])

        general_mask = PIL.Image.fromarray(np.ones((width, height, 3), dtype=np.uint8))

        control_mask_1 = zero.copy()
        x1, y1, x2, y2 = face_info_a["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask_1[y1:y2, x1:x2] = 255
        control_mask_1 = PIL.Image.fromarray(control_mask_1.astype(np.uint8))

        control_mask_2 = zero.copy()
        x1, y1, x2, y2 = face_info_b["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask_2[y1:y2, x1:x2] = 255
        control_mask_2 = PIL.Image.fromarray(control_mask_2.astype(np.uint8))

        controlnet_masks = [control_mask_1, control_mask_2, general_mask]
        ip_adapter_images = [face_emb_1, face_emb_2, style_image, ]

        masks = self.ip_adapter_mask_processor.preprocess([control_mask_1, control_mask_2, general_mask], height=height, width=width)
        ip_adapter_masks = [mask.unsqueeze(0) for mask in masks]

        inpaint_mask = torch.logical_or(torch.tensor(np.array(control_mask_1)), torch.tensor(np.array(control_mask_2))).float()
        inpaint_mask = PIL.Image.fromarray((inpaint_mask.numpy() * 255).astype(np.uint8)).convert("RGB")

        new_ip_adapter_masks = []
        for ip_img, mask in zip(ip_adapter_images, controlnet_masks):
            if isinstance(ip_img, list):
                num_images = len(ip_img)
                mask = mask.repeat(1, num_images, 1, 1)

            new_ip_adapter_masks.append(mask)
            
        generator = torch.Generator(device="cpu").manual_seed(seed)

        self.pipeline.set_ip_adapter_scale([identity_image_strength_1, identity_image_strength_2,
            {
                "down": { "block_2": [0.0, 0.0] }, #Composition
                "up": { "block_0": [0.0, style_image_strength, 0.0] } #Style
            }
        ])

        images = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt, 
            guidance_scale=guidance_scale,
            num_inference_steps=number_of_steps,
            num_images_per_prompt=number_of_images,
            ip_adapter_image=ip_adapter_images,
            cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
            image=base_image,
            mask_image=inpaint_mask,
            i2i_mask_guidance_start=mask_guidance_start,
            i2i_mask_guidance_end=mask_guidance_end,
            control_image=[face_kps_identity_image_a, face_kps_identity_image_b, depth_image],
            control_mask=controlnet_masks,
            identity_control_indices=[(0,0), (1,1)],
            controlnet_conditioning_scale=[identity_image_strength_1, identity_image_strength_2, depth_image_strength],
            strength=1-base_image_strength,
            generator=generator,
            seed=seed,
        ).images

        return images

    def patch_onnx_runtime(
        self,
        inter_op_num_threads: int = 16,
        intra_op_num_threads: int = 16,
        omp_num_threads: int = 16,
    ):
        import os

        import onnxruntime as ort

        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

        _default_session_options = ort.capi._pybind_state.get_default_session_options()

        def get_default_session_options_new():
            _default_session_options.inter_op_num_threads = inter_op_num_threads
            _default_session_options.intra_op_num_threads = intra_op_num_threads
            return _default_session_options

        ort.capi._pybind_state.get_default_session_options = get_default_session_options_new
        