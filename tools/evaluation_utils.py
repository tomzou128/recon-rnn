import torch

import numpy as np
import json

def eval_depth(depth_pred, depth_trgt, near=0.0, far=10.0):
    """ Computes 2d metrics between two depth maps

    Args:
        depth_pred: mxn np.array containing prediction
        depth_trgt: mxn np.array containing ground truth

    Returns:
        Dict of metrics
    """

    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < far) * (depth_trgt > near) * mask1

    depth_pred = depth_pred[mask]
    depth_trgt = depth_trgt[mask]
    abs_diff = np.abs(depth_pred - depth_trgt)
    abs_rel = abs_diff / depth_trgt
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_trgt
    sq_log_diff = (np.log(depth_pred) - np.log(depth_trgt)) ** 2
    thresh = np.maximum((depth_trgt / depth_pred), (depth_pred / depth_trgt))
    r5 = (thresh < 1.05).astype('float')
    r10 = (thresh < 1.10).astype('float')
    r1 = (thresh < 1.25).astype('float')
    r2 = (thresh < 1.25 ** 2).astype('float')
    r3 = (thresh < 1.25 ** 3).astype('float')

    metrics = {}
    metrics['AbsRel'] = np.mean(abs_rel)
    metrics['AbsDiff'] = np.mean(abs_diff)
    metrics['SqRel'] = np.mean(sq_rel)
    metrics['RMSE'] = np.sqrt(np.mean(sq_diff))
    metrics['LogRMSE'] = np.sqrt(np.mean(sq_log_diff))
    metrics['r5'] = np.mean(r5)
    metrics['r10'] = np.mean(r10)
    metrics['r1'] = np.mean(r1)
    metrics['r2'] = np.mean(r2)
    metrics['r3'] = np.mean(r3)
    metrics['complete'] = np.mean(mask1.astype('float'))

    return metrics


def eval_depth_batched(depth_pred, depth_trgt, near=0, far=10.0):
    depth_pred = torch.stack(depth_pred).squeeze(2).permute(1, 0, 2, 3)
    mask1 = depth_pred > 0  # ignore values where prediction is 0 (% complete)
    mask = (depth_trgt < far) * (depth_trgt > near) * mask1

    depth_gt_bN = depth_trgt.flatten(start_dim=1).float()
    depth_pred_bN = depth_pred.flatten(start_dim=1).float()
    mask_bN = mask.flatten(start_dim=1)

    depth_gt_bN[~mask_bN] = torch.nan
    depth_pred_bN[~mask_bN] = torch.nan

    abs_diff = torch.abs(depth_pred_bN - depth_gt_bN)
    abs_rel = abs_diff / depth_gt_bN
    sq_diff = abs_diff ** 2
    sq_rel = sq_diff / depth_gt_bN
    sq_log_diff = (torch.log(depth_pred_bN) - torch.log(depth_gt_bN)) ** 2

    thresh = torch.max(torch.stack([(depth_gt_bN / depth_pred_bN), (depth_pred_bN / depth_gt_bN)], dim=2), dim=2)[0]
    r5 = (thresh < 1.05).float()
    r5[~mask_bN] = torch.nan
    r10 = (thresh < 1.10).float()
    r10[~mask_bN] = torch.nan
    r1 = (thresh < 1.25).float()
    r1[~mask_bN] = torch.nan
    r2 = (thresh < 1.25 ** 2).float()
    r2[~mask_bN] = torch.nan
    r3 = (thresh < 1.25 ** 3).float()
    r3[~mask_bN] = torch.nan

    metrics = {}
    metrics['AbsRel'] = torch.nanmean(abs_rel, dim=1)
    metrics['AbsDiff'] = torch.nanmean(abs_diff, dim=1)
    metrics['SqRel'] = torch.nanmean(sq_rel, dim=1)
    metrics['RMSE'] = torch.sqrt(torch.nanmean(sq_diff, dim=1))
    metrics['LogRMSE'] = torch.sqrt(torch.nanmean(sq_log_diff, dim=1))
    metrics['r5'] = torch.nanmean(r5, dim=1)
    metrics['r10'] = torch.nanmean(r10, dim=1)
    metrics['r1'] = torch.nanmean(r1, dim=1)
    metrics['r2'] = torch.nanmean(r2, dim=1)
    metrics['r3'] = torch.nanmean(r3, dim=1)
    metrics['complete'] = torch.mean(mask1.float(), dim=1)

    return metrics


class ResultsAverager():
    """
    Helper class for stable averaging of metrics across frames and scenes.
    """

    def __init__(self, exp_name, metrics_name):
        """
            Args:
                exp_name: name of the specific experiment.
                metrics_name: type of metrics.
        """
        self.exp_name = exp_name
        self.metrics_name = metrics_name

        self.elem_metrics_list = []
        self.running_metrics = None
        self.running_count = 0

        self.final_computed_average = None

    def update_results(self, elem_metrics):
        """
        Adds elem_matrix to elem_metrics_list. Updates running_metrics with
        incomming metrics to keep a running average.

        running_metrics are cheap to compute but not totally stable.
        """

        self.elem_metrics_list.append(elem_metrics.copy())

        if self.running_metrics is None:
            self.running_metrics = elem_metrics.copy()
        else:
            for key in list(elem_metrics.keys()):
                self.running_metrics[key] = (
                                                    self.running_metrics[key] *
                                                    self.running_count
                                                    + elem_metrics[key]
                                            ) / (self.running_count + 1)

        self.running_count += 1

    def print_sheets_friendly(
            self, print_exp_name=True,
            include_metrics_names=False,
            print_running_metrics=True,
    ):
        """
        Print for easy sheets copy/paste.
        Args:
            print_exp_name: should we print the experiment name?
            include_metrics_names: should we print a row for metric names?
            print_running_metrics: should we print running metrics or the
                final average?
        """

        if print_exp_name:
            print(f"{self.exp_name}, {self.metrics_name}")

        if print_running_metrics:
            metrics_to_print = self.running_metrics
        else:
            metrics_to_print = self.final_metrics

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics to print.")
            return

        metric_names_row = ""
        metrics_row = ""
        for k, v in metrics_to_print.items():
            metric_names_row += f"{k:8} "
            metric_string = f"{v:.4f},"
            metrics_row += f"{metric_string:8} "

        if include_metrics_names:
            print(metric_names_row)
        print(metrics_row)

    def output_json(self, filepath, print_running_metrics=False):
        """
        Outputs metrics to a json file.
        Args:
            filepath: file path where we should save the file.
            print_running_metrics: should we print running metrics or the
                final average?
        """
        scores_dict = {}
        scores_dict["exp_name"] = self.exp_name
        scores_dict["metrics_type"] = self.metrics_name

        scores_dict["scores"] = {}

        if print_running_metrics:
            metrics_to_use = self.running_metrics
        else:
            metrics_to_use = self.final_metrics

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics will be output.")

        metric_names_row = ""
        metrics_row = ""
        for k, v in metrics_to_use.items():
            metric_names_row += f"{k:8} "
            metric_string = f"{v:.4f},"
            metrics_row += f"{metric_string:8} "
            scores_dict["scores"][k] = float(v)

        scores_dict["metrics_string"] = metric_names_row
        scores_dict["scores_string"] = metrics_row

        with open(filepath, "w") as file:
            json.dump(scores_dict, file, indent=4)

    def pretty_print_results(
            self,
            print_exp_name=True,
            print_running_metrics=True
    ):
        """
        Pretty print for easy(ier) reading
        Args:
            print_exp_name: should we print the experiment name?
            include_metrics_names: should we print a row for metric names?
            print_running_metrics: should we print running metrics or the
                final average?
        """
        if print_running_metrics:
            metrics_to_print = self.running_metrics
        else:
            metrics_to_print = self.final_metrics

        if len(self.elem_metrics_list) == 0:
            print("WARNING: No valid metrics to print.")
            return

        if print_exp_name:
            print(f"{self.exp_name}, {self.metrics_name}")
        for k, v in metrics_to_print.items():
            print(f"{k:8}: {v:.4f}")

    def compute_final_average(self, ignore_nans=False):
        """
        Computes final a final average on the metrics element list using
        numpy.

        This should be more accurate than running metrics as it's a single
        average vs multiple high level multiplications and divisions.

        Args:
            ignore_nans: ignore nans in the results and run using nanmean.
        """

        self.final_metrics = {}

        if len(self.elem_metrics_list) == 0:
            print("WARNING: no valid entry to average!")
            return

        for key in list(self.running_metrics.keys()):
            values = []
            for element in self.elem_metrics_list:
                if torch.is_tensor(element[key]):
                    values.append(element[key].cpu().numpy())
                else:
                    values.append(element[key])

            if ignore_nans:
                mean_value = np.nanmean(np.array(values))
            else:
                mean_value = np.array(values).mean()
            self.final_metrics[key] = mean_value
