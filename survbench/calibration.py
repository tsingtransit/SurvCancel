"""Fixed-window discrimination and validation-only probability calibration."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from survbench.metrics import window_c_index


def equal_mass_bins(y, p, n_bins=10):
    rows, ece = [], 0.0
    for chunk in np.array_split(np.argsort(p, kind="mergesort"), n_bins):
        if not len(chunk):
            continue
        predicted, observed = float(np.mean(p[chunk])), float(np.mean(y[chunk]))
        gap = abs(predicted - observed)
        ece += len(chunk) / len(y) * gap
        rows.append({"n": len(chunk), "mean_predicted": predicted,
                     "observed_rate": observed, "absolute_gap": gap})
    return rows, float(ece)


def probability_metrics(arrays, probability=None):
    y = np.asarray(arrays["y_cancel_window"], dtype=np.int8)
    p = np.asarray(arrays["p_cancel_window"] if probability is None else probability, dtype=np.float64)
    if not len(y) or p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected nonempty, finite cancellation probabilities in [0, 1].")
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    bins, ece = equal_mass_bins(y, clipped)
    both = len(np.unique(y)) == 2
    ci = window_c_index(p, arrays["time_to_first_event_in_window"], arrays["first_event_type"])
    return {"n": len(y), "c_index_cancel": float(ci) if np.isfinite(ci) else None,
            "brier": float(np.mean((p - y) ** 2)),
            "auroc": float(roc_auc_score(y, p)) if both else None,
            "average_precision": float(average_precision_score(y, p)) if np.any(y) else None,
            "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
            "ece_10_equal_mass": ece, "reliability_bins": bins}


def validation_platt(validation, test):
    """Fit only validation labels, separately by stage; test labels are unused."""
    val_p = np.asarray(validation["p_cancel_window"], dtype=np.float64)
    val_y = np.asarray(validation["y_cancel_window"], dtype=np.int8)
    test_p = np.asarray(test["p_cancel_window"], dtype=np.float64)
    calibrated, parameters = test_p.copy(), {}
    for stage in (0, 1):
        vm, tm = validation["stage"] == stage, test["stage"] == stage
        if len(np.unique(val_y[vm])) < 2:
            parameters[str(stage)] = {"status": "unchanged_insufficient_validation_classes", "n_validation": int(vm.sum())}
            continue
        vp = np.clip(val_p[vm], 1e-6, 1 - 1e-6)
        tp = np.clip(test_p[tm], 1e-6, 1 - 1e-6)
        fit = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        fit.fit(np.log(vp / (1 - vp)).reshape(-1, 1), val_y[vm])
        if tm.any():
            calibrated[tm] = fit.predict_proba(np.log(tp / (1 - tp)).reshape(-1, 1))[:, 1]
        parameters[str(stage)] = {"status": "fitted_on_validation", "intercept": float(fit.intercept_[0]),
                                  "slope": float(fit.coef_[0, 0]), "n_validation": int(vm.sum())}
    return calibrated, parameters
