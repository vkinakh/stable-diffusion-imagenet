from pathlib import Path
import random
from tqdm import tqdm
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

def main():
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",
                                                   revision="fp16", torch_dtype=torch.float16,
                                                   use_auth_token=False)
    pipe = pipe.to(torch.device("cuda"))
    n = 200

    out_dir = Path('./caltech_256')
    out_dir.mkdir(exist_ok=True, parents=True)

    with open('caltech256_idx_to_class.txt', 'r') as f:
        classes = f.read().splitlines()

    for cls_id, lemma in tqdm(enumerate(classes)):
        curr_dir = out_dir / f'{cls_id:03d}_{lemma}'
        curr_dir.mkdir(exist_ok=True, parents=True)

        lemma = lemma.replace('_', ' ')
        prompt = f'a photo of {lemma}, realistic, high quality'

        for img_idx in range(n):
            seed = random.randint(0, 10000)

            with autocast('cuda'):
                generator = torch.Generator("cuda").manual_seed(seed)
                out = pipe(prompt, generator=generator)
                image = out["sample"][0]
                out_path = curr_dir / f'{img_idx:03d}_{seed}.jpg'
                image.save(out_path)


if __name__ == '__main__':
    main()
