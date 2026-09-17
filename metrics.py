# import torch
#
# import config
#
#
# def update_confusion_matrix(confusion_matrix, prediction, target):
#     prediction = prediction.reshape(-1).long()
#     target = target.reshape(-1).long()
#
#     valid = target != config.IGNORE_INDEX
#
#     prediction = prediction[valid]
#     target = target[valid]
#
#     indices = target * config.NUM_CLASSES + prediction
#
#     current = torch.bincount(
#         indices,
#         minlength=config.NUM_CLASSES ** 2,
#     )
#
#     current = current.reshape(
#         config.NUM_CLASSES,
#         config.NUM_CLASSES,
#     )
#
#     confusion_matrix += current.cpu()
#
#
# def metrics_from_confusion_matrix(confusion_matrix):
#     cm = confusion_matrix.double()
#     total = cm.sum().clamp_min(1)
#
#     class_names = ["background", "ptm", "nptm"]
#     result = {
#         "confusion_matrix": cm.long().tolist()
#     }
#
#     foreground_dice = []
#     foreground_iou = []
#
#     for class_index, class_name in enumerate(class_names):
#         true_positive = cm[class_index, class_index]
#         false_positive = cm[:, class_index].sum() - true_positive
#         false_negative = cm[class_index, :].sum() - true_positive
#         true_negative = (
#             total
#             - true_positive
#             - false_positive
#             - false_negative
#         )
#
#         dice = (
#             2.0 * true_positive
#             / (
#                 2.0 * true_positive
#                 + false_positive
#                 + false_negative
#             ).clamp_min(1)
#         ).item()
#
#         iou = (
#             true_positive
#             / (
#                 true_positive
#                 + false_positive
#                 + false_negative
#             ).clamp_min(1)
#         ).item()
#
#         precision = (
#             true_positive
#             / (
#                 true_positive
#                 + false_positive
#             ).clamp_min(1)
#         ).item()
#
#         recall = (
#             true_positive
#             / (
#                 true_positive
#                 + false_negative
#             ).clamp_min(1)
#         ).item()
#
#         specificity = (
#             true_negative
#             / (
#                 true_negative
#                 + false_positive
#             ).clamp_min(1)
#         ).item()
#
#         result[class_name] = {
#             "dice": dice,
#             "iou": iou,
#             "precision": precision,
#             "recall": recall,
#             "specificity": specificity,
#         }
#
#         if class_index > 0:
#             foreground_dice.append(dice)
#             foreground_iou.append(iou)
#
#     result["macro_foreground_dice"] = (
#         sum(foreground_dice) / len(foreground_dice)
#     )
#
#     result["macro_foreground_iou"] = (
#         sum(foreground_iou) / len(foreground_iou)
#     )
#
#     result["pixel_accuracy"] = (
#         cm.diag().sum() / total
#     ).item()
#
#     # Merge PTM and NPTM into one TM foreground class.
#     merged_true_positive = cm[1:, 1:].sum()
#     merged_false_positive = cm[0, 1:].sum()
#     merged_false_negative = cm[1:, 0].sum()
#
#     result["merged_tm_dice"] = (
#         2.0 * merged_true_positive
#         / (
#             2.0 * merged_true_positive
#             + merged_false_positive
#             + merged_false_negative
#         ).clamp_min(1)
#     ).item()
#
#     return result


import torch

import config


def update_confusion_matrix(
    confusion_matrix,
    prediction,
    target,
):
    prediction = prediction.reshape(-1).long()
    target = target.reshape(-1).long()

    valid = target != config.IGNORE_INDEX

    prediction = prediction[valid]
    target = target[valid]

    indices = (
        target * config.NUM_CLASSES
        + prediction
    )

    current = torch.bincount(
        indices,
        minlength=config.NUM_CLASSES ** 2,
    )

    current = current.reshape(
        config.NUM_CLASSES,
        config.NUM_CLASSES,
    )

    confusion_matrix += current.cpu()


def metrics_from_confusion_matrix(
    confusion_matrix,
):
    cm = confusion_matrix.double()
    total = cm.sum().clamp_min(1)

    result = {
        "confusion_matrix": cm.long().tolist()
    }

    all_foreground_dice = []
    all_foreground_iou = []
    primary_dice = []
    primary_iou = []

    for class_index, class_name in enumerate(
        config.CLASS_NAMES
    ):
        true_positive = cm[
            class_index,
            class_index,
        ]

        false_positive = (
            cm[:, class_index].sum()
            - true_positive
        )

        false_negative = (
            cm[class_index, :].sum()
            - true_positive
        )

        true_negative = (
            total
            - true_positive
            - false_positive
            - false_negative
        )

        dice = (
            2.0 * true_positive
            / (
                2.0 * true_positive
                + false_positive
                + false_negative
            ).clamp_min(1)
        ).item()

        iou = (
            true_positive
            / (
                true_positive
                + false_positive
                + false_negative
            ).clamp_min(1)
        ).item()

        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            ).clamp_min(1)
        ).item()

        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            ).clamp_min(1)
        ).item()

        specificity = (
            true_negative
            / (
                true_negative
                + false_positive
            ).clamp_min(1)
        ).item()

        result[class_name] = {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
        }

        if class_name != "background":
            all_foreground_dice.append(dice)
            all_foreground_iou.append(iou)

        if class_name in config.PRIMARY_CLASS_NAMES:
            primary_dice.append(dice)
            primary_iou.append(iou)

    result["all_classes_mean_dice"] = (
        sum(all_foreground_dice)
        / len(all_foreground_dice)
    )

    result["all_classes_mean_iou"] = (
        sum(all_foreground_iou)
        / len(all_foreground_iou)
    )

    result["primary_mean_dice"] = (
        sum(primary_dice)
        / len(primary_dice)
    )

    result["primary_mean_iou"] = (
        sum(primary_iou)
        / len(primary_iou)
    )

    result["pixel_accuracy"] = (
        cm.diag().sum() / total
    ).item()

    ptm_index = config.CLASS_NAMES.index("ptm")
    nptm_index = config.CLASS_NAMES.index("nptm")

    tm_indices = [ptm_index, nptm_index]

    merged_true_positive = cm[
        tm_indices,
        :,
    ][:, tm_indices].sum()

    non_tm_indices = [
        index
        for index in range(config.NUM_CLASSES)
        if index not in tm_indices
    ]

    merged_false_positive = cm[
        non_tm_indices,
        :,
    ][:, tm_indices].sum()

    merged_false_negative = cm[
        tm_indices,
        :,
    ][:, non_tm_indices].sum()

    result["merged_tm_dice"] = (
        2.0 * merged_true_positive
        / (
            2.0 * merged_true_positive
            + merged_false_positive
            + merged_false_negative
        ).clamp_min(1)
    ).item()

    result["merged_tm_iou"] = (
        merged_true_positive
        / (
            merged_true_positive
            + merged_false_positive
            + merged_false_negative
        ).clamp_min(1)
    ).item()

    return result
