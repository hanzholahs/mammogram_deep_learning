import tensorflow as tf

CLASS_LABELS = ["benign", "malignant"]


def concatenate_dataset(ds1: tf.data.Dataset, ds2: tf.data.Dataset) -> tf.data.Dataset:
    """Unbatch and concatenate two TensorFlow datasets.
    
    Args:
        ds1 (tf.data.Dataset): First dataset.
        ds2 (tf.data.Dataset): Second dataset.
    Returns:
        tf.data.Dataset: Concatenated dataset.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    return ds1.unbatch().concatenate(ds2.unbatch())


def prepare_dataset(image_paths, image_lables, image_resize):
    """Create a dataset from image paths and labels with preprocessing.
    
    Args:
        image_paths (list[str]): Paths to image files.
        image_lables (list[str]): Corresponding class labels.
        image_resize (tuple[int, int]): Target resize dimensions.
    Returns:
        tf.data.Dataset: Preprocessed dataset of (image, label) pairs.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_lables))
    dataset = dataset.map(lambda path, label: load_image(path, label, image_resize))
    return dataset


def optimize_dataset(dataset, batch_size, shuffle_buffer_size, prefetch_buffer_size):
    """Apply caching, shuffling, batching, and prefetching to optimize dataset pipeline.
    
    Args:
        dataset (tf.data.Dataset): Input dataset.
        batch_size (int): Number of samples per batch.
        shuffle_buffer_size (int): Buffer size for shuffling.
        prefetch_buffer_size (int): Buffer size for prefetching.
    Returns:
        tf.data.Dataset: Optimized dataset pipeline.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    dataset = dataset.cache()
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer_size)
    return dataset


def load_image(image_path, image_label, image_resize, class_labels=CLASS_LABELS):
    """Load and preprocess an image, returning normalized pixels and encoded label.
    
    Args:
        image_path (str): Path to the image file.
        image_label (str): Class label for the image.
        image_resize (tuple[int, int]): Target resize dimensions.
        class_labels (list[str], optional): List of class names. Defaults to CLASS_LABELS.
    Returns:
        tuple[tf.Tensor, tf.Tensor]: Normalized image tensor and integer label.
    
    [AI-assisted] This docstring and some other documentation-related tasks generated with assistance from generative AI and reviewed by a human.
    """
    label = tf.argmax(image_label == class_labels)

    pixels = tf.io.read_file(image_path)
    pixels = tf.io.decode_png(pixels, dtype=tf.uint16, channels=1)
    pixels = tf.image.resize(pixels, image_resize)
    pixels = tf.cast(pixels, tf.float32) / 65535.0

    return pixels, label
