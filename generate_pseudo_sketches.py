import yaml
import cv2

import os
import argparse

from pathlib import Path
from PIL import Image

from util.pseudo_sketches_webui_runner import WebuiAPI
from util.util import read_bmp_np


def main(args):
    # Open the file and load its contents
    with open(args.input_txt, 'r') as file:
        nouns = list(map(lambda x: x.strip(), file.readlines()))
    
    # Initiate webui runner to run Stable Diffusion
    api = WebuiAPI(args.url)
    
    for noun in nouns:
        imgs = api.generate_image(
            prompt=noun, 
            steps=7, cfg_scale=2, width=1024, height=1024, 
            batch_size=args.sketch_batch_size, seed=args.seed)

        output_dir_sketch_basepath = os.path.join(args.output_dir, "sketches")

        for i, img in enumerate(imgs):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_img_3c = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
            
            noun_for_c = noun.replace(" ", "_")
            sketch_filename = f"{noun_for_c}_{i}.bmp"
            sketch_path = os.path.join(output_dir_sketch_basepath, sketch_filename)
            
            print(f"{sketch_path}")
            cv2.imwrite(sketch_path, gray_img_3c)

            # vectorize generated sketch
            # it save <sketch_path>.bmp -> <sketch_path>_shifted.bmp
            os.system(f"./FDoG_GUI {sketch_path}")
            
            vectorized_sketch_filename = f"{noun_for_c}_{i}_shifted.bmp"
            vectorized_sketch_path = os.path.join(output_dir_sketch_basepath, vectorized_sketch_filename)
            
            vectorized_sketch_filename_png = f"{noun_for_c}_{i}_shifted.png"
            vectorized_sketch_png_path = os.path.join(output_dir_sketch_basepath, vectorized_sketch_filename_png)
            
            # bmp 파일로 읽으면 에러가 남. 직접 정의한 함수로 읽어와야함.
            sketch = Image.fromarray(read_bmp_np(vectorized_sketch_path))
            
            rsc_path = os.path.join(output_dir_sketch_basepath, f"*_rsc")
            print(f"$ rm -rf {rsc_path}")
            os.system(f"rm -rf {rsc_path}")
            
            sketch_to_imgs = api.generate_image(
                prompt=noun, 
                steps=7, cfg_scale=2, width=1024, height=1024, 
                batch_size=args.sktch2img_batch_size, sketch=sketch
            )
            for j, sk2img in enumerate(sketch_to_imgs):
                if j == args.sktch2img_batch_size: break
                print(os.path.join(args.output_dir, "imgs", f"{noun}_{i}_{j}_img.png"))
                cv2.imwrite(os.path.join(args.output_dir, "imgs", f"{noun}_{i}_{j}_img.png"), sk2img)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Generate pseudo sketch-image dataset.')

    # Add arguments
    parser.add_argument("--input_txt", type=str, default="nouns_to_generate.txt")
    parser.add_argument("--output_dir", type=str, default="pseudo_img_sketch_dataset", 
                            help="Spedify directory path to save results")
    parser.add_argument('--sketch_batch_size', type=int, default=1, help='sketch batch size')
    parser.add_argument('--sktch2img_batch_size', type=int, default=1, help='sketch-image batch size')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--url', type=str, default="http://127.0.0.1:7860", help="url for running webui of AUTOMATIC1111")

    # Parse the arguments
    args = parser.parse_args()

    # create directories
    Path(os.path.join(args.output_dir, "sketches")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "imgs")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, "depths")).mkdir(parents=True, exist_ok=True)

    # Set absolute path for paths
    args.output_dir = os.path.abspath(args.output_dir)

    # Now you can use the arguments as variables in your program
    print(f"input_txt: {args.input_txt}")
    print(f"output_dir: {args.output_dir}")
    print(f"sketch_batch_size: {args.sketch_batch_size}")
    print(f"sktch2img_batch_size: {args.sktch2img_batch_size}")
    print(f"seed: {args.seed}")
    print(f"url: {args.url}")
    
    main(args)
    