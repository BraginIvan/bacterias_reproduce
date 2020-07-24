from segmentation.train_dice import run as run_dice
from argparse import ArgumentParser
from utils.constants import dataset_path

parser = ArgumentParser()
parser.add_argument("-id")
args = parser.parse_args()

config_dice = {
    'dataset_path': dataset_path
}

run_dice(config_dice, int(args.id))
