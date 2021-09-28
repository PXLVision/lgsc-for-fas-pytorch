from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from metrics import plot_confusion_matrix, to_one_hot, draw_probability_density_distribution,get_tps_fps_for_class, plot_pr_curve, draw_roc


def create_graphs_and_metrics(results_df: pd.DataFrame, output_folder: Path):
    predictions = results_df["Score"].to_numpy()
    y_true = results_df["Label"].to_numpy().astype(int)
    results = results_df

    classes = ["Attack", "Real"]

    scheme_results_path = output_folder
    # Metrics
    histogram(results, scheme_results_path)
    prec_rec_dict = pr_curve(y_true, predictions, scheme_results_path)
    statistics_dict = ROC(prec_rec_dict, scheme_results_path)

    EER, EER_threshold, FNR, FPR = get_EER(statistics_dict, prec_rec_dict)
    HTER, HTER_threshold = get_HTER(FNR, FPR, prec_rec_dict)

    confusion_matrix(
        y_true, predictions, classes, EER_threshold, scheme_results_path
    )
    confusion_matrix(
        y_true, predictions, classes, HTER_threshold, scheme_results_path
    )

    # Save results
    results.to_csv(scheme_results_path / "full_results.csv", index=False)
    with open(scheme_results_path / "report.txt", "w+") as report_file:
        report_file.write(f"EER: {EER} at threshold: {EER_threshold}\n")
        report_file.write(f"HTER: {HTER} at threshold:  {HTER_threshold}\n")

        # BPCER at different APCERs
        APCERS = [0.1, 0.05, 0.01, 0.005, 0.001, 0]
        for apcer in APCERS:
            tmp_FPR = FPR - apcer
            apcer_arg = np.argmin(np.abs(tmp_FPR))
            FNR_apcer = FNR[apcer_arg]
            apcer_threshold = prec_rec_dict["Real class"][2][apcer_arg]
            report_file.write(
                f"BPCER: {FNR_apcer} at APCER: {apcer} using threshold: {apcer_threshold}\n"
            )


def histogram(results: pd.DataFrame, scheme_results_path: Path):
    pos_results = list(results.loc[results["Label"] == 1]["Score"].to_numpy())
    neg_results = list(results.loc[results["Label"] == 0]["Score"].to_numpy())
    legend = ["Real videos", "Attack videos"]
    draw_probability_density_distribution(
        [pos_results, neg_results], legend=legend, range=[0.9, 1.0]
    )
    plt.savefig(scheme_results_path / "Histogram.png")
    plt.clf()
    plt.cla()


def pr_curve(
        y_true: List[int], predictions: List[float], scheme_results_path: Path
):
    prec_rec_dict = {}
    y_true_one_hot = to_one_hot(y_true, 2)
    y_pred = np.zeros((len(predictions), 2))
    y_pred[:, 1] = predictions
    y_pred[:, 0] = 1 - y_pred[:, 1]
    prec_rec_dict["Real class"] = get_tps_fps_for_class(y_true_one_hot, y_pred, 1)

    plot_pr_curve(prec_rec_dict, False)
    plt.savefig(scheme_results_path / "precision_recall.png")
    plt.clf()
    plt.cla()
    return prec_rec_dict


def ROC(prec_rec_dict: Dict, scheme_results_path: Path):
    statistics_dict = draw_roc(prec_rec_dict)
    plt.savefig(scheme_results_path / "ROC.png")
    plt.clf()
    plt.cla()
    return statistics_dict


def get_EER(statistics_dict: Dict, prec_rec_dict: Dict):
    result_dict = statistics_dict[0]
    FNR = 1 - result_dict["TPR"]
    FPR = result_dict["FPR"]
    Error_dif = np.abs(FNR - FPR)
    EER_arg = np.argmin(Error_dif)
    EER = FPR[EER_arg]
    EER_Threshold = prec_rec_dict["Real class"][2][EER_arg + 1]

    return EER, EER_Threshold, FNR, FPR


def get_HTER(FNR, FPR, prec_rec_dict):
    HTER = (FNR + FPR) / 2
    Best_HTER = np.min(HTER)
    Best_HTER_arg = np.argmin(HTER)
    BEST_HTER_Threshold = prec_rec_dict["Real class"][2][Best_HTER_arg + 1]
    return Best_HTER, BEST_HTER_Threshold


def confusion_matrix(
        y_true, predictions, classes, threshold, scheme_results_path
):
    y_pred = list(1 * (predictions > threshold))
    plot_confusion_matrix(y_true, y_pred, classes, normalize=True)
    plt.savefig(
        scheme_results_path / "Confusion_matrix_{:.4f}.png".format(threshold)
    )
    plt.clf()
    plt.cla()
