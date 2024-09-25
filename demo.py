from omni_zero import OmniZeroCouple

def demo():
    omni_zero = OmniZeroCouple(
        base_model="frankjoshua/albedobaseXL_v13",
        device="cuda",
    )
        
    base_image="https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg"
    style_image="https://cdn-prod.styleof.com/inferences/cm1ho5cjl14nh14jec6phg2h8/i6k59e7gpsr45ufc7l8kun0g-medium.jpeg"
    identity_image_1="https://ichef.bbci.co.uk/images/ic/1040x1040/p0f5vv8q.jpg"#"https://cdn-prod.styleof.com/inferences/cm1hp4lea14oz14jeoghnex7g/dlgc5xwo0qzey7qaixy45i1o-medium.jpeg"
    identity_image_2="https://www.judentum-projekt.de/images/meitner22-2_640.jpg"#"https://cdn-prod.styleof.com/inferences/cm1ho69ha14np14jesnusqiep/mp3aaktzqz20ujco5i3bi5s1-medium.jpeg"

    images = omni_zero.generate(
        seed=42,
        prompt="Cinematic still photo of a couple. emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous, film grain, grainy",
        negative_prompt="anime, cartoon, graphic, (blur, blurry, bokeh), text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured",
        guidance_scale=3.0,
        number_of_images=1,
        number_of_steps=10,
        base_image=base_image,
        base_image_strength=0.3,
        style_image=style_image,
        style_image_strength=1.0,
        identity_image_1=identity_image_1,
        identity_image_strength_1=1.0,
        identity_image_2=identity_image_2,
        identity_image_strength_2=1.0,
        depth_image=None,
        depth_image_strength=0.2, 
        mask_guidance_start=0.0,
        mask_guidance_end=1.0,
    )

    for i, image in enumerate(images):
        image.save(f"oz_output_{i}.jpg")

if __name__ == "__main__":
    demo()