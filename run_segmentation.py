from segmentation.train_dice import run as run_dice
from pathlib import Path

dataset_path = Path('/home/ivan/projects/its/dataset_its/')
config_dice = {
    'dataset_path': dataset_path
}
for version in range(10):
    run_dice(config_dice, version)
