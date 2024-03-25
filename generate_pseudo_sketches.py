import yaml
import cv2

import os
import argparse

from util.pseudo_sketches_webui_runner import WebuiAPI


def main(args):
    # Open the file and load its contents
    with open(args.input_yaml, 'r') as file:
        nouns = yaml.safe_load(file).split()
    
    # Initiate webui runner to run Stable Diffusion
    api = WebuiAPI()
    
    for noun in nouns:
        imgs = api.generate_image(
            prompt=noun, 
            steps=5, cfg_scale=2, width=1024, height=1024, 
            batch_size=args.sketch_batch_size, seed=args.seed)

        for i, img in enumerate(imgs):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, sketch = cv2.threshold(gray_img, 50, 255, 0)
            cv2.imwrite(os.path.join(args.output_dir, "sketches", f"{noun}_{i}.png"), sketch)

            sketch_to_imgs = api.generate_image(
                prompt=noun, 
                steps=5, cfg_scale=2, width=1024, height=1024, 
                batch_size=args.sktch2img_batch_size, sketch=sketch
            )
            for j, sk2img in enumerate(sketch_to_imgs):
                if j == args.sktch2img_batch_size: break
                cv2.imwrite(os.path.join(args.output_dir, "imgs", f"{noun}_{i}_{j}_img.png"), sk2img)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Generate pseudo sketch-image dataset.')

    # Add arguments
    parser.add_argument("--input_yaml", type=str, default="nouns_to_generate.yaml")
    parser.add_argument("--output_dir", type=str, default="pseudo_img_sketch_dataset", 
                            help="Spedify directory path to save results")
    parser.add_argument('--sketch_batch_size', type=int, default=4, help='sketch batch size')
    parser.add_argument('--sktch2img_batch_size', type=int, default=1, help='sketch-image batch size')
    parser.add_argument('--seed', type=int, default=-1)

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments as variables in your program
    print(f"input_yaml: {args.input_yaml}")
    print(f"output_dir: {args.output_dir}")
    print(f"sketch_batch_size: {args.sketch_batch_size}")
    print(f"sktch2img_batch_size: {args.sktch2img_batch_size}")
    print(f"seed: {args.seed}")
    
    main(args)
    