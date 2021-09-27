from typing import List

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels


def get_npcer(false_negative: int, true_positive: int):
    return false_negative / (false_negative + true_positive)


def get_apcer(false_positive: int, true_negative: int):
    return false_positive / (true_negative + false_positive)


def get_acer(apcer: float, npcer: float):
    return (apcer + npcer) / 2.0


def get_metrics(pred: ndarray, targets: ndarray):
    negative_indices = targets == 0
    positive_indices = targets == 1

    false_positive = (pred[negative_indices] == 1).sum()
    false_negative = (pred[positive_indices] == 0).sum()

    true_positive = (pred[positive_indices] == 1).sum()
    true_negative = (pred[negative_indices] == 0).sum()

    npcer = get_npcer(false_negative, true_positive)
    apcer = get_apcer(false_positive, true_negative)

    acer = get_acer(apcer, npcer)

    return acer, apcer, npcer


def get_threshold(probs: ndarray, grid_density: int = 10):
    min_, max_ = min(probs), max(probs)
    thresholds = [min_]
    for i in range(grid_density + 1):
        thresholds.append(min_ + (i * (max_ - min_)) / float(grid_density))
    thresholds.append(1.1)
    return thresholds


def eval_from_scores(scores: ndarray, targets: ndarray):
    thrs = get_threshold(scores)
    acc = 0.0
    best_thr = -1
    for thr in thrs:
        acc_new = accuracy_score(targets, scores >= thr)
        if acc_new > acc:
            best_thr = thr
            acc = acc_new
    return get_metrics(scores >= best_thr, targets), best_thr, acc


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    classes: list,
    normalize: bool = False,
    title: str = None,
    cmap: object = plt.cm.Blues,
):
    """This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        classes (list): Class names
        normalize (bool, optional): Whether to normalize entries. Defaults to False.
        title (str, optional): Custom title of the plot. Defaults to None.
        cmap (object, optional): Custom matplotlib colormap. Defaults to plt.cm.Blues.

    Returns:
        object: An matplotlib axes object
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = np.array(classes)[np.int32(unique_labels(y_true, y_pred))]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def plot_error_distribution(
    errors: np.ndarray, title: str, show: bool = False
) -> plt.Figure:
    """
    Plot an error distribution as histogram

    Args:
        errors: an array of errors
        title: title for the error plot, i.e. which error is plotted
        show: flag indicating if plot should be displayed

    Returns:
        plt.Figure: the histogram figure
    """
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlabel("Error range")
    ax.set_ylabel("Frequency")

    plt.title(f"Error range distribution: {title}")

    ax.hist(errors)

    ax.legend()
    if show:
        plt.show()

    return fig


def draw_probability_density_distribution(
    results: List[List[float]], legend: List[str], range: List[float] = [0.0, 1.0]
):
    plt.title("Scores distribution")
    plt.xlabel("Score")
    plt.ylabel("Probability Density")
    plt.hist(results, range=range, bins=100, density=True, log=True)

    plt.legend(legend)


def get_tps_fps_for_class(y_true: list, softmax_values: list, class_id: int):
    """Calculate True Positives and False Positives in relation to the sorted
       thresholds of confidence values. Required for plotting a precision recall curve.

    Args:
        y_true (list): Groundtruth true labels
        softmax_values (list): Confidence softmax values of the predictions
        class_id (int): Class id

    Returns:
        tuple: Cumulative True Positives, False Positives and sorted thresholds
               according to confidence values.
    """

    # TODO: any particular reason for mergesort here? Only really useful for on-disk sorting
    sorted_softmax_indices_by_score = np.argsort(
        softmax_values[:, class_id], kind="mergesort"
    )[::-1]
    y_true = y_true[sorted_softmax_indices_by_score, class_id]
    y_pred = softmax_values[sorted_softmax_indices_by_score, class_id]

    distinct_indices = np.where(np.diff(y_pred))[0]
    threshold_inidices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_inidices]
    fps = np.cumsum(1 - y_true)[threshold_inidices]

    return tps, fps, y_pred[threshold_inidices]


def plot_pr_curve(pr_dict: dict, show: bool = False):
    """Plot a precision recall curve of multiple classes

    Args:
        pr_dict (dict): Dictionary containing True Positives,
                        False Positives and sorted confidence thresholds for each class
                        (entries should be generated with get_tps_fps_for_class(...))
        show (bool, optional): Whether to display the generated plot. Defaults to False.

    Returns:
        object: A matplotlib figure object
    """
    fig = plt.gcf()
    ax = fig.gca()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    plt.title("Precision Recall curve")

    for key, value in pr_dict.items():
        precision = value[0] / (value[0] + value[1])
        precision[np.isnan(precision)] = 0
        recall = value[0] / (value[0][-1])
        (line,) = ax.plot(recall, precision)
        line.set_label(key)

    ax.legend()
    if show:
        plt.show()

    return fig


def draw_roc(pr_dict: dict):
    plt.switch_backend("agg")
    plt.rcParams["figure.figsize"] = (6.0, 6.0)
    plt.title("ROC")
    results_dict = []
    for key, value in pr_dict.items():
        TPR = value[0] / np.max(value[0])
        FPR = value[1] / np.max(value[1])
        plt.plot(FPR, TPR, "b", label=f"{key} ROC curve")

        dict = {}
        dict["TPR"] = TPR
        dict["FPR"] = FPR
        results_dict.append(dict)

    plt.legend(loc="upper right")
    plt.plot([0, 1], [1, 0], "r--")
    plt.grid(ls="--")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    return results_dict


def to_one_hot(labels: list, max_value: int = None) -> np.ndarray:
    """Converts categorical class labels to one hot representation.

    Args:
        labels (list): Categorical labels as a list of integers

    Returns:
        numpy.ndarray: A numpy array of the converted class labels.
    """
    labels = np.asarray(labels).astype(int)
    if not max_value:
        shape = (labels.size, labels.max() + 1)
    else:
        shape = (labels.size, max_value)
    one_hot = np.zeros(shape)
    rows = np.arange(labels.size)

    one_hot[rows, labels] = 1
    return one_hot