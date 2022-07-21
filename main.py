import argparse
from CLIP import CLIP

def parse_args():
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument("--train", "-t", action="store_true", default=False, help="training")
    parser.add_argument("--inference", "-i", action="store_true", default=False, help="training")
    parser.add_argument("--wandb", "-w", action="store_true", default=False, help="wand loggings")
    parser.add_argument("--save_weight", "-s", action="store_true", default=True, help="save model weight")
    return parser.parse_args()

def main():
    args = parse_args()
    clip = CLIP(args)

    #build model
    clip.build_model()

    if args.train:
        clip.train()
    
    if args.inference:
        clip.inference()

if __name__=="__main__":
    main()