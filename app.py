import os

import gradio as gr
import spaces

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch

#Hack for ZeroGPU
torch.jit.script = lambda f: f
####

import cv2
import numpy as np
import PIL
from controlnet_aux import ZoeDetector
from diffusers import DPMSolverMultistepScheduler
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.models import ControlNetModel
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis
from pipeline import OmniZeroPipeline
from transformers import CLIPVisionModelWithProjection
from utils import align_images, draw_kps, load_and_resize_image


def patch_onnx_runtime(
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
    

base_model = "frankjoshua/albedobaseXL_v13"

patch_onnx_runtime()

snapshot_download("okaris/antelopev2", local_dir="./models/antelopev2")
face_analysis = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
face_analysis.prepare(ctx_id=0, det_size=(640, 640))

dtype = torch.float16

ip_adapter_plus_image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", 
    subfolder="models/image_encoder",
    torch_dtype=dtype,
).to("cuda")

zoedepthnet_path = "okaris/zoe-depth-controlnet-xl"
zoedepthnet = ControlNetModel.from_pretrained(zoedepthnet_path,torch_dtype=dtype).to("cuda")

identitiynet_path = "okaris/face-controlnet-xl"
identitynet = ControlNetModel.from_pretrained(identitiynet_path, torch_dtype=dtype).to("cuda")

zoe_depth_detector = ZoeDetector.from_pretrained("lllyasviel/Annotators").to("cuda")
ip_adapter_mask_processor = IPAdapterMaskProcessor()

pipeline = OmniZeroPipeline.from_pretrained(
    base_model,
    controlnet=[identitynet, identitynet, zoedepthnet],
    torch_dtype=dtype,
    image_encoder=ip_adapter_plus_image_encoder,
).to("cuda")

config = pipeline.scheduler.config
config["timestep_spacing"] = "trailing"
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++", final_sigmas_type="zero")

pipeline.load_ip_adapter(["okaris/ip-adapter-instantid", "okaris/ip-adapter-instantid", "h94/IP-Adapter"], subfolder=[None, None, "sdxl_models"], weight_name=["ip-adapter-instantid.bin", "ip-adapter-instantid.bin", "ip-adapter-plus_sdxl_vit-h.safetensors"])

@spaces.GPU()
def generate(
    base_image="https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg",
    style_image="https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg",
    identity_image_1="https://cdn-prod.styleof.com/inferences/cm1hp4lea14oz14jeoghnex7g/dlgc5xwo0qzey7qaixy45i1o-medium.jpeg",
    identity_image_2="https://cdn-prod.styleof.com/inferences/cm1ho69ha14np14jesnusqiep/mp3aaktzqz20ujco5i3bi5s1-medium.jpeg",
    seed=42,
    prompt="Cinematic still photo of a couple. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy",
    negative_prompt="anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
    guidance_scale=3.0,
    number_of_images=1,
    number_of_steps=10,
    base_image_strength=0.3,
    style_image_strength=1.0,
    identity_image_strength_1=1.0,
    identity_image_strength_2=1.0,
    depth_image=None,
    depth_image_strength=0.2,
    mask_guidance_start=0.0,
    mask_guidance_end=1.0,
    progress=gr.Progress(track_tqdm=True)
):
    resolution = 1024

    if base_image is not None:
        base_image = load_and_resize_image(base_image, resolution, resolution)

    if depth_image is None:
        depth_image = zoe_depth_detector(base_image, detect_resolution=resolution, image_resolution=resolution)
    else:
        depth_image = load_and_resize_image(depth_image, resolution, resolution)

    base_image, depth_image = align_images(base_image, depth_image)

    if style_image is not None:
        style_image = load_and_resize_image(style_image, resolution, resolution)
    else:
        style_image = base_image 
        # raise ValueError("You must provide a style image")
    
    if identity_image_1 is not None:
        identity_image_1 = load_and_resize_image(identity_image_1, resolution, resolution)
    else:
        raise ValueError("You must provide an identity image")
    
    if identity_image_2 is not None:
        identity_image_2 = load_and_resize_image(identity_image_2, resolution, resolution)
    else:
        raise ValueError("You must provide an identity image 2")

    height, width = base_image.size

    face_info_1 = face_analysis.get(cv2.cvtColor(np.array(identity_image_1), cv2.COLOR_RGB2BGR))
    for i, face in enumerate(face_info_1):
        print(f"Face 1 -{i}: Age: {face['age']}, Gender: {face['gender']}")
    face_info_1 = sorted(face_info_1, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb_1 = torch.tensor(face_info_1['embedding']).to("cuda", dtype=dtype)

    face_info_2 = face_analysis.get(cv2.cvtColor(np.array(identity_image_2), cv2.COLOR_RGB2BGR))
    for i, face in enumerate(face_info_2):
        print(f"Face 2 -{i}: Age: {face['age']}, Gender: {face['gender']}")
    face_info_2 = sorted(face_info_2, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])[-1] # only use the maximum face
    face_emb_2 = torch.tensor(face_info_2['embedding']).to("cuda", dtype=dtype)

    zero = np.zeros((width, height, 3), dtype=np.uint8)
    # face_kps_identity_image_1 = draw_kps(zero, face_info_1['kps'])
    # face_kps_identity_image_2 = draw_kps(zero, face_info_2['kps'])

    face_info_img2img = face_analysis.get(cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR))
    faces_info_img2img = sorted(face_info_img2img, key=lambda x:(x['bbox'][2]-x['bbox'][0])*x['bbox'][3]-x['bbox'][1])
    face_info_a = faces_info_img2img[-1]
    face_info_b = faces_info_img2img[-2]
    # face_emb_a = torch.tensor(face_info_a['embedding']).to("cuda", dtype=dtype)
    # face_emb_b = torch.tensor(face_info_b['embedding']).to("cuda", dtype=dtype)
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

    masks = ip_adapter_mask_processor.preprocess([control_mask_1, control_mask_2, general_mask], height=height, width=width)
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

    pipeline.set_ip_adapter_scale([identity_image_strength_1, identity_image_strength_2,
        {
            "down": { "block_2": [0.0, 0.0] }, #Composition
            "up": { "block_0": [0.0, style_image_strength, 0.0] } #Style
        }
    ])

    images = pipeline(
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

#Move the components in the example fields outside so they are available when gr.Examples is instantiated
buy_me_a_coffee_button = """
[![Buy me a coffee](https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=vk654cf2pv8&button_colour=BD5FFF&font_colour=ffffff&font_family=Bree&outline_colour=000000&coffee_colour=FFDD00)](https://www.buymeacoffee.com/vk654cf2pv8)
"""

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center'>Omni Zero Couples</h1>")
    gr.Markdown("<h4 style='text-align: center'>A diffusion pipeline for zero-shot stylized portrait creation [<a href='https://github.com/okaris/omni-zero-couples' target='_blank'>GitHub</a>]")#, [<a href='https://styleof.com/s/remix-yourself' target='_blank'>StyleOf Remix Yourself</a>]</h4>")
    gr.Markdown(buy_me_a_coffee_button)

    with gr.Row():
        with gr.Column():
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", value="Cinematic still photo of a couple. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy")
            with gr.Row():
                negative_prompt = gr.Textbox(label="Negative Prompt", value="anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured")
            with gr.Row():
                with gr.Column(min_width=140):
                    with gr.Row():
                        base_image = gr.Image(label="Base Image")
                    with gr.Row():
                        base_image_strength = gr.Slider(label="Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
                with gr.Column(min_width=140):
                    with gr.Row():
                        identity_image = gr.Image(label="Identity Image")
                    with gr.Row():
                        identity_image_strength = gr.Slider(label="Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
                with gr.Column(min_width=140):
                    with gr.Row():
                        identity_image_2 = gr.Image(label="Identity Image 2")
                    with gr.Row():
                        identity_image_strength_2 = gr.Slider(label="Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
            with gr.Accordion("Advanced options", open=False):    
                with gr.Row():
                    with gr.Column():
                        style_image = gr.Image(label="Style Image")
                        style_image_strength = gr.Slider(label="Style Strength",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
                    with gr.Column():
                        depth_image = gr.Image(label="Depth Image")
                        depth_image_strength = gr.Slider(label="Depth Strength",step=0.01, minimum=0.0, maximum=1.0, value=0.5)
                with gr.Row():
                    seed = gr.Slider(label="Seed",step=1, minimum=0, maximum=10000000, value=42)
                    number_of_images = gr.Slider(label="Number of Outputs",step=1, minimum=1, maximum=4, value=1)
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance Scale",step=0.1, minimum=0.0, maximum=14.0, value=3.0)
                    number_of_steps = gr.Slider(label="Number of Steps",step=1, minimum=1, maximum=50, value=10)
                with gr.Row():
                    mask_guidance_start = gr.Slider(label="Mask Guidance Start",step=0.01, minimum=0.0, maximum=1.0, value=0.0)
                    mask_guidance_end = gr.Slider(label="Mask Guidance End",step=0.01, minimum=0.0, maximum=1.0, value=1.0)
            
        with gr.Column():
            with gr.Row():
                out = gr.Gallery(label="Output(s)")
            with gr.Row():
                # clear = gr.Button("Clear")
                submit = gr.Button("Generate")
        
                submit.click(generate, inputs=[
                    base_image,
                    style_image if style_image is not None else bas,
                    identity_image,
                    identity_image_2,
                    seed,
                    prompt,
                    negative_prompt,
                    guidance_scale,
                    number_of_images,
                    number_of_steps,
                    base_image_strength,
                    style_image_strength,
                    identity_image_strength,
                    identity_image_strength_2,
                    depth_image,
                    depth_image_strength,
                    mask_guidance_start,
                    mask_guidance_end,
                    ],
                    outputs=[out]
                )
        # clear.click(lambda: None, None, chatbot, queue=False)
    gr.Examples(
        examples=[
            [
                "https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg",
                "https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg",
                "https://cdn-prod.styleof.com/inferences/cm1hp4lea14oz14jeoghnex7g/dlgc5xwo0qzey7qaixy45i1o-medium.jpeg",
                "https://cdn-prod.styleof.com/inferences/cm1ho69ha14np14jesnusqiep/mp3aaktzqz20ujco5i3bi5s1-medium.jpeg",
                42,
                "Cinematic still photo of a couple. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy",
            ]
        ],
        inputs=[base_image, style_image, identity_image, identity_image_2, seed, prompt],
        outputs=[out],
        fn=generate,
        cache_examples="lazy",
    )
if __name__ == "__main__":
    demo.launch()