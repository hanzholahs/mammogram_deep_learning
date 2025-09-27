from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


def augment_images(
    paths: Sequence[str | Path],
    output_folder: str | Path,
    n_augment: int,
    augmentation_fn: Callable,
) -> None:
    """
    Apply an augmentation function to a list of images and save results.

    Parameters
    ----------
    paths : Sequence[str | Path]
        List of image file paths to process.
    output_folder : str | Path
        Destination directory where augmented images are saved.
    n_augment : int
        Number of augmented versions to generate per image.
    augmentation_fn : Callable
        Augmentation function returning a dict with key "image",
        e.g. from Albumentations.

    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for p in tqdm(paths, desc="Augmenting images"):
        p = Path(p)
        filename = p.stem

        img = Image.open(p)
        img_array = np.array(img)

        for i in range(n_augment):
            augmented = augmentation_fn(image=img_array)["image"]
            Image.fromarray(augmented).save(
                output_folder / f"{filename}-augmented-{i + 1}.png"
            )
