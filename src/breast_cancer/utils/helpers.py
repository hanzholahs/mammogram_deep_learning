import os

def get_save_path(base_dir: str, model_name: str, suffix: str = ".h5") -> str:
    """Create a path like: <base_dir>/<model_name>/<model_name><suffix>"""
    path = os.path.join(base_dir, model_name, model_name + suffix)
    return os.path.normcase(path)