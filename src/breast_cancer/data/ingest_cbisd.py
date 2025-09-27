import glob
import os
from typing import List, Tuple
from pathlib import Path

import pandas as pd
import pydicom as dicom
from numpy.typing import NDArray
from tqdm import tqdm

from breast_cancer.utils.calcs import to_uint8

LABEL_MAPPING = {
    "BENIGN": "benign",
    "BENIGN_WITHOUT_CALLBACK": "benign",
    "MALIGNANT": "malignant",
}


def cbisd_extract_pixels(
    raw_dir: str, convert_to_uint8: bool = False
) -> Tuple[List[str], List[NDArray]]:
    """Load cropped CBIS-DDSM DICOM ROI images from a directory.

    Parameters
    ----------
    raw_dir : str
        Path to the root directory containing CBIS-DDSM DICOM files. This is the folder named `CBIS-DDSM` downloaded from https://www.cancerimagingarchive.net/collection/cbis-ddsm/.

    Returns
    -------
    image_paths : list of str
        List of cropped image identifiers (from DICOM PatientID).
    image_pixels : list of numpy.ndarray
        List of pixel arrays for the cropped ROI images.

    Notes
    -----
    - Only images with SeriesDescription == "cropped images" are included.
    - Patient ID in the CBIS-DDSM refers to single instance of ROI. One subject can have multiple Patient IDs.

    Data Citation
    -------------
    Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
    Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set].
    The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    dicom_files = glob.glob(os.path.join(raw_dir, "**", "*.dcm"), recursive=True)

    image_pixels = []
    image_paths = []
    for file in tqdm(dicom_files, desc="Reading CBIS-DDSM DICOMs"):
        dcm = dicom.dcmread(file)
        try:
            if dcm.SeriesDescription != "cropped images":
                continue
            image_paths.append(dcm.PatientID)
            if convert_to_uint8:
                image_pixels.append(to_uint8(dcm.pixel_array))
            else:
                image_pixels.append(dcm.pixel_array)
        except AttributeError:
            pass
    return image_paths, image_pixels


def cbisd_import_description(csv_files: List[str]) -> pd.DataFrame:
    """[AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.

    Import CBIS-DDSM description CSV files into a DataFrame.

    Parameters
    ----------
    csv_files : list of str
        Paths to CBIS-DDSM CSV description files downloaded from https://www.cancerimagingarchive.net/collection/cbis-ddsm/.
        There are four files, which include:
        - calc_case_description_test_set.csv
        - calc_case_description_train_set.csv
        - mass_case_description_test_set.csv
        - mass_case_description_train_set.csv

    Returns
    -------
    description : pandas.DataFrame
        DataFrame with additional columns:
        - 'image paths': first path component from 'cropped image file path'
        - 'image labels': mapped labels ('benign' or 'malignant')

    Data Citation
    -------------
    Sawyer-Lee, R., Gimenez, F., Hoogi, A., & Rubin, D. (2016).
    Curated Breast Imaging Subset of Digital Database for Screening Mammography (CBIS-DDSM) [Data set].
    The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY

    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    cbisd_columns = [
        "source",
        "patient_id",
        "breast density",
        "left or right breast",
        "image view",
        "abnormality id",
        "abnormality type",
        "calc type",
        "calc distribution",
        "mass shape",
        "mass margins",
        "assessment",
        "pathology",
        "subtlety",
        "image file path",
        "cropped image file path",
        "ROI mask file path",
        "breast_density",
        "image paths",
        "image labels",
    ]

    description = pd.DataFrame()
    for path in [Path(f) for f in csv_files]:
        df = pd.read_csv(path)
        df.insert(0, "source", path.name)
        description = pd.concat([description, df])

    description["image paths"] = description["cropped image file path"].apply(
        lambda x: x.split("/")[0]
    )

    description["image labels"] = description["pathology"].map(
        lambda x: LABEL_MAPPING[x]
    )

    description = description[cbisd_columns]

    return description
