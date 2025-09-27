from pathlib import Path
import pandas as pd
import re


def create_manifest(
    image_dir: str | Path, 
    patient_id_pattern: str = r"P_\d+"
) -> pd.DataFrame:
    """Create a manifest DataFrame for images in a directory structured by label.

    Parameters
    ----------
    image_dir : str | Path
        Path to the root folder containing subfolders for each label. 
        Each subfolder should contain PNG images.
    patient_id_pattern : str, optional
        Regex pattern to extract patient ID from the filename. 
        Default is 'P_\\d+' from CBIS-DDSM dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - 'filepath': full path to the image
        - 'patientID': extracted from filename using the provided regex pattern
        - 'label': name of the parent folder (assumed to be class label)

    Notes
    -----
    - Expects images to be in subfolders named after their labels.
    - Filenames not matching the patient ID pattern will raise an error.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    image_dir = Path(image_dir)
    all_image_paths = list(image_dir.rglob("*.png"))

    filepaths = []
    patient_ids = []
    labels = []

    for path in all_image_paths:
        match = re.search(patient_id_pattern, path.name)
        if match is None:
            raise ValueError(f"Filename {path.name} does not match pattern {patient_id_pattern}")
        filepaths.append(str(path))
        patient_ids.append(match.group(0))
        labels.append(path.parent.name)

    manifest = pd.DataFrame({
        "filepath": filepaths,
        "patientID": patient_ids,
        "label": labels,
    })

    return manifest
