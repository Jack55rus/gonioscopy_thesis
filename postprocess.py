import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import measure


def postprocess_class_mask(
    mask: np.ndarray,
    min_size: int,
    max_distance: float,
) -> np.ndarray:
    """
    Post-process one binary class mask.

    Steps:
    1. Split mask into 8-connected objects.
    2. Remove objects smaller than min_size.
    3. Find the largest remaining object.
    4. Keep the largest object.
    5. Keep any other object only if its minimum Euclidean pixel distance
       to the largest object is <= max_distance.

    Parameters
    ----------
    mask:
        Boolean or binary HxW array.
    min_size:
        Minimum component area in pixels.
    max_distance:
        Maximum allowed Euclidean distance in pixels from the largest object.

    Returns
    -------
    np.ndarray
        Boolean HxW mask after post-processing.
    """
    mask = mask.astype(bool)

    labeled = measure.label(
        mask,
        connectivity=2,  # 8-connectivity in 2D
    )

    regions = measure.regionprops(labeled)

    # First remove small components.
    valid_regions = [
        region
        for region in regions
        if region.area >= min_size
    ]

    if not valid_regions:
        return np.zeros_like(mask, dtype=bool)

    # Largest surviving object is the reference object.
    largest_region = max(
        valid_regions,
        key=lambda region: region.area,
    )

    largest_mask = labeled == largest_region.label

    # distance_transform_edt gives, for every non-largest pixel,
    # distance to the nearest pixel belonging to largest_mask.
    distance_to_largest = distance_transform_edt(
        ~largest_mask
    )

    output = largest_mask.copy()

    for region in valid_regions:
        if region.label == largest_region.label:
            continue

        component_mask = labeled == region.label

        min_component_distance = float(
            distance_to_largest[component_mask].min()
        )

        if min_component_distance <= max_distance:
            output[component_mask] = True

    return output


def postprocess_prediction(
    prediction: np.ndarray,
    class_ids,
    min_size: int,
    max_distance: float,
) -> np.ndarray:
    """
    Apply object filtering independently to selected segmentation classes.

    Parameters
    ----------
    prediction:
        HxW integer semantic-segmentation prediction.
    class_ids:
        Iterable of integer class IDs to post-process.
    min_size:
        Minimum connected-component size in pixels.
    max_distance:
        Maximum component-to-largest-component distance in pixels.

    Returns
    -------
    np.ndarray
        Post-processed HxW integer prediction.
    """
    result = prediction.copy()

    for class_id in class_ids:
        class_mask = prediction == class_id

        filtered_mask = postprocess_class_mask(
            class_mask,
            min_size=min_size,
            max_distance=max_distance,
        )

        # Remove the old prediction for this class.
        result[result == class_id] = 0

        # Put surviving objects back.
        result[filtered_mask] = class_id

    return result
