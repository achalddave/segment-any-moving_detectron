import cv2
import logging

from pathlib import Path


def load_flow_png(png_path):
    # R channel contains angle, G channel contains magnitude. Note that
    # this image is loaded in BGR format because of OpenCV.
    try:
        image = cv2.imread(png_path).astype(float)
    except Exception:
        logging.exception('Failed to load image at %s' % png_path)
        raise
    image_path = Path(png_path)
    minmax_path = image_path.parent / (
        image_path.stem + '_magnitude_minmax.txt')
    assert minmax_path.exists(), (
        'Magnitude min max path %s does not exist for image %s' %
        (image_path, minmax_path))
    with open(minmax_path, 'r') as f:
        magnitude_min, magnitude_max = f.read().strip().split()
        magnitude_min = float(magnitude_min)
        magnitude_max = float(magnitude_max)

    image[:, :, 1] = ((image[:, :, 1] / 255) * (magnitude_max - magnitude_min)
                      + magnitude_min)
    height, width = image.shape[:2]

    # Largest possible flow vector
    diagonal = (height ** 2 + width ** 2) ** 0.5

    # A few frames in FlyingThings3D have extremely large values, which leads
    # to NaNs in the output. We clip them arbitrarily.
    image[:, :, 1] = image[:, :, 1].clip(max=diagonal)

    # 6 is a magic constant Pavel used for normalization
    image[:, :, 1] /= (diagonal / 6)

    return image
