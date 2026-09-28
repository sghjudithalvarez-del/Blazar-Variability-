import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, least_squares
import statsmodels.formula.api as smf
from scipy.optimize import OptimizeWarning
import warnings
#from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import chi2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def load_data(filename):
    """
    Lädt Daten aus einer .dat Datei
    und gibt ein pandas DataFrame zurück.
    """

    data = np.genfromtxt(
        filename,
        skip_header=19,
        delimiter='\t',
        dtype=float
    )

    df = pd.DataFrame({
        "MJD": data[:, 0],
        "MJD_err": data[:, 9],
        "CU": data[:, 7],
        "CU_err": data[:, 15],
        "Flux": data[:, 8],
        "Flux_err": data[:, 16]
    })

    return df


def load_data(filename):
    """
    Loads compact FACT .dat files and returns a pandas DataFrame.

    Expected columns:
    0 MJDstart
    1 MJDstop
    2 corrected_flux[ph/cm2/s]
    3 corrected_flux_error[ph/cm2/s]
    4 corrected_flux[CU]
    5 corrected_flux_error[CU]
    6 ontime
    7-10 quality / cut flags
    """

    data = np.genfromtxt(
        filename,
        comments="#",
        delimiter=None,
        dtype=float
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    mjd_start = data[:, 0]
    mjd_stop = data[:, 1]

    df = pd.DataFrame({
        "MJD": mjd_start,
        "MJD_err": (mjd_stop - mjd_start) / 2,
        "CU": data[:, 4],
        "CU_err": data[:, 5],
        "Flux": data[:, 2],
        "Flux_err": data[:, 3],
        "MJDstop": mjd_stop,
        "ontime": data[:, 6],
    })

    return df


# ---------------------------------------------
# Nichtlineare Modellfunktionen
# ---------------------------------------------

def gauss(x, A, mu, sigma, offset):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

def lorentz(x, A, x0, gamma, offset):
    return A * gamma**2 / ((x - x0) ** 2 + gamma**2) + offset


# -------------------------------------------
# Datenpunkte Gruppieren
# -------------------------------------------

def gruppiere_datenpunkte(df, abstand, spalte):
    """
    Gruppiert Datenpunkte anhand einer Spalte und eines Abstandes.

    Fuer FACT-Lightcurves mit MJD/MJDstop wird der Abstand als
    current MJD start - previous MJD stop berechnet. Fuer andere Daten faellt
    die Funktion auf die alte start-to-start Differenz zurueck.
    """
    df = df.copy()
    df[spalte] = pd.to_numeric(df[spalte], errors="coerce")
    df = df.sort_values(by=spalte).reset_index(drop=True)

    if spalte == "MJD" and "MJDstop" in df.columns:
        df["MJDstop"] = pd.to_numeric(df["MJDstop"], errors="coerce")
        gaps = df[spalte] - df["MJDstop"].shift(1)
    else:
        gaps = df[spalte].diff()

    gruppe = 0
    gruppen_liste = []

    for i, gap in enumerate(gaps):
        if i == 0:
            gruppen_liste.append(gruppe)
        elif gap > abstand:
            gruppe += 1
            gruppen_liste.append(gruppe)
        else:
            gruppen_liste.append(gruppe)

    df["Gruppe"] = gruppen_liste

    return df

# bis hier funktioniert es und macht sinn


def gruppiere_beobachtungsfenster(
    df,
    max_gap_days=0.5,
    start_col="MJD",
    stop_col="MJDstop",
    group_col="Gruppe",
):
    """
    Gruppiert Beobachtungsfenster nach der Luecke zwischen vorherigem Stop
    und aktuellem Start.

    Eine neue Gruppe beginnt, wenn:
        current_start - previous_stop > max_gap_days
    """
    if start_col not in df.columns:
        raise ValueError(f"start_col '{start_col}' fehlt im DataFrame.")
    if stop_col not in df.columns:
        raise ValueError(f"stop_col '{stop_col}' fehlt im DataFrame.")

    df = df.copy()
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    df[stop_col] = pd.to_numeric(df[stop_col], errors="coerce")
    df = df.sort_values(start_col).reset_index(drop=True)

    df["gap_days_from_previous_stop"] = df[start_col] - df[stop_col].shift(1)

    group = 0
    groups = []

    for i, gap in enumerate(df["gap_days_from_previous_stop"]):
        if i == 0:
            groups.append(group)
        elif gap > max_gap_days:
            group += 1
            groups.append(group)
        else:
            groups.append(group)

    df[group_col] = groups
    return df


def gruppiere_nach_mjd_nacht(
    df,
    spalte="MJD",
    mjd_offset=0.5,
    group_col="Gruppe",
    night_col="MJD_Nacht",
    start_at=1,
):
    """
    Gruppiert Datenpunkte nach MJD-Nacht.

    mjd_offset=0.5 entspricht dem astronomisch ueblichen Tageswechsel bei
    Mittag UTC: floor(MJD + 0.5). Das ist eine kalenderbasierte Alternative
    zur Luecken-Gruppierung mit abstand=0.5.

    Die MJD-Nacht wird in night_col behalten. group_col bekommt eine
    fortlaufende, natuerliche Nummerierung, damit Plot-Titel nicht die
    grossen MJD-Werte verwenden muessen.
    """
    df = df.copy()
    df[spalte] = pd.to_numeric(df[spalte], errors="coerce")
    df = df.sort_values(by=spalte).reset_index(drop=True)
    df[night_col] = np.floor(df[spalte] + mjd_offset).astype(int)
    df[group_col] = pd.factorize(df[night_col], sort=True)[0] + start_at
    return df


def _candidate_criteria_labels(row):
    labels = []

    q_const = row.get("q_const", np.nan)
    if np.isfinite(q_const):
        if q_const < 0.01:
            labels.append("q_const < 0.01")
        elif bool(row.get("passes_q_const", False)):
            labels.append("q_const < candidate threshold")
        elif bool(row.get("passes_possible_q_const", False)):
            labels.append("q_const < possible threshold")

    range_p_mc = row.get("range_p_mc", np.nan)
    if np.isfinite(range_p_mc):
        if range_p_mc < 0.01:
            labels.append("range MC p < 0.01")
        elif bool(row.get("support_range_mc", False)):
            labels.append("range MC support")

    if bool(row.get("support_fvar", False)):
        labels.append("F_var support")
    if bool(row.get("support_adjacent", False)):
        labels.append("adjacent-point support")

    if bool(row.get("passes_min_strong_points", False)):
        labels.append("n >= strong minimum")

    return "; ".join(labels) if labels else "no candidate criteria passed"


def diagnose_grouping_gaps(
    df,
    start_col="MJD",
    stop_col="MJDstop",
    threshold_days=0.5,
    spalte=None,
):
    """
    Gibt die Luecken zwischen vorherigem Stop und aktuellem Start zurueck.

    Falls stop_col fehlt, faellt die Funktion auf start-to-start Luecken
    zurueck und meldet dies per Warnung. Der Parameter spalte bleibt als
    Rueckwaertskompatibilitaet fuer alte Notebook-Zellen erhalten.
    """
    if spalte is not None:
        start_col = spalte

    if start_col not in df.columns:
        raise ValueError(f"start_col '{start_col}' fehlt im DataFrame.")

    df = df.copy()
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    df = df.sort_values(start_col).reset_index(drop=True)

    if stop_col in df.columns:
        df[stop_col] = pd.to_numeric(df[stop_col], errors="coerce")
        previous_stop = df[stop_col].shift(1)
    else:
        warnings.warn(
            f"stop_col '{stop_col}' fehlt; verwende start-to-start Luecken.",
            RuntimeWarning,
            stacklevel=2,
        )
        previous_stop = df[start_col].shift(1)

    diagnostics = pd.DataFrame({
        "previous_stop": previous_stop,
        "current_start": df[start_col],
    })
    diagnostics["gap_days"] = (
        diagnostics["current_start"] - diagnostics["previous_stop"]
    )
    diagnostics["gap_hours"] = diagnostics["gap_days"] * 24.0
    diagnostics["is_gap_gt_threshold"] = (
        diagnostics["gap_days"] > threshold_days
    )

    return diagnostics.iloc[1:].reset_index(drop=True)


def _benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR-Korrektur.
    Gibt q-Werte in der Originalreihenfolge zurueck.
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan, dtype=float)
    finite = np.isfinite(p_values)

    if not np.any(finite):
        return q_values

    p = p_values[finite]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    raw_q = ranked * m / np.arange(1, m + 1)
    monotonic_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    monotonic_q = np.clip(monotonic_q, 0.0, 1.0)

    q_finite = np.empty_like(monotonic_q)
    q_finite[order] = monotonic_q
    q_values[finite] = q_finite
    return q_values


def _weighted_mean(y, yerr):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    positive = yerr > 0
    if not np.any(positive):
        return np.nan
    safe_yerr = np.where(positive, yerr, np.min(yerr[positive]))
    weights = 1.0 / safe_yerr**2
    return np.average(y, weights=weights)


def _fractional_variability(y, yerr):
    """
    Berechnet F_var und eine konservative 1-sigma Unsicherheit.

    Grundlage ist die normalisierte Excess Variance nach Vaughan et al. 2003.
    Fuer negative Excess Variance wird F_var auf 0 gesetzt; der untere Rand ist
    dann nicht positiv und kann keinen Kandidaten stuetzen.
    """
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    n = len(y)

    if n < 2:
        return {
            "excess_variance": np.nan,
            "normalized_excess_variance": np.nan,
            "normalized_excess_variance_err": np.nan,
            "F_var": np.nan,
            "F_var_err": np.nan,
            "F_var_lower": np.nan,
        }

    mean_flux = np.mean(y)
    if mean_flux == 0 or not np.isfinite(mean_flux):
        return {
            "excess_variance": np.nan,
            "normalized_excess_variance": np.nan,
            "normalized_excess_variance_err": np.nan,
            "F_var": np.nan,
            "F_var_err": np.nan,
            "F_var_lower": np.nan,
        }

    sample_variance = np.var(y, ddof=1)
    mean_err_sq = np.mean(yerr**2)
    excess_variance = sample_variance - mean_err_sq
    normalized_excess_variance = excess_variance / mean_flux**2
    f_var = np.sqrt(max(normalized_excess_variance, 0.0))

    nxs_err = np.sqrt(
        (np.sqrt(2.0 / n) * mean_err_sq / mean_flux**2) ** 2
        + (np.sqrt(mean_err_sq / n) * 2.0 * f_var / mean_flux) ** 2
    )

    if f_var > 0:
        f_var_err = np.sqrt(f_var**2 + nxs_err) - f_var
    else:
        f_var_err = np.sqrt(nxs_err)

    return {
        "excess_variance": excess_variance,
        "normalized_excess_variance": normalized_excess_variance,
        "normalized_excess_variance_err": nxs_err,
        "F_var": f_var,
        "F_var_err": f_var_err,
        "F_var_lower": f_var - f_var_err,
    }


def _max_adjacent_significance(x, y, yerr):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if len(y) < 2:
        return np.nan

    order = np.argsort(x)
    y = y[order]
    yerr = yerr[order]
    denom = np.sqrt(yerr[:-1]**2 + yerr[1:]**2)
    valid = denom > 0

    if not np.any(valid):
        return np.nan

    sig = np.full(len(denom), np.nan)
    sig[valid] = np.abs(np.diff(y)[valid]) / denom[valid]
    return np.nanmax(sig)


def _monte_carlo_range_p_value(y, yerr, n_sim=10000, random_state=None):
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    if len(y) < 2:
        return np.nan

    mean_const = _weighted_mean(y, yerr)
    if not np.isfinite(mean_const):
        return np.nan

    obs_range = np.max(y) - np.min(y)
    rng = np.random.default_rng(random_state)
    sims = rng.normal(loc=mean_const, scale=yerr, size=(int(n_sim), len(y)))
    sim_ranges = np.ptp(sims, axis=1)

    # Add-one correction avoids returning an impossible exact zero p-value.
    return (np.count_nonzero(sim_ranges >= obs_range) + 1.0) / (int(n_sim) + 1.0)


def flag_single_candidate_group(
    x,
    y,
    yerr,
    n_sim=10000,
    random_state=None,
    min_points=3,
    min_strong_points=5,
    fvar_lower_tol=0.03,
    adjacent_sigma_threshold=2.5,
    range_mc_threshold=0.05,
):
    """
    Bewertet eine einzelne Nacht/Gruppe als IDV-Kandidat.

    Kandidatenlogik:
    q_const < 0.05 muss spaeter in flag_candidate_groups ergänzt werden.
    Diese Funktion berechnet alle gruppenlokalen Metriken inklusive p_const.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    n = len(y)

    result = {
        "n": n,
        "duration_hours": (np.nanmax(x) - np.nanmin(x)) * 24.0 if n else np.nan,
        "weighted_mean": np.nan,
        "chi2_const": np.nan,
        "dof_const": np.nan,
        "p_const": np.nan,
        "q_const": np.nan,
        "F_var": np.nan,
        "F_var_err": np.nan,
        "F_var_lower": np.nan,
        "max_adjacent_sigma": np.nan,
        "range_CU": np.nan,
        "range_p_mc": np.nan,
        "support_fvar": False,
        "support_adjacent": False,
        "support_range_mc": False,
        "is_candidate": False,
        "candidate_class": "not_testable" if n < min_points else "pending_q_value",
    }

    if n < min_points:
        return result

    positive = yerr > 0
    if not np.any(positive):
        return result
    yerr = np.where(positive, yerr, np.min(yerr[positive]))

    weighted_mean = _weighted_mean(y, yerr)
    chi2_const = np.sum(((y - weighted_mean) / yerr) ** 2)
    dof_const = n - 1
    p_const = 1.0 - chi2.cdf(chi2_const, dof_const)

    fvar_result = _fractional_variability(y, yerr)
    max_adjacent_sigma = _max_adjacent_significance(x, y, yerr)
    range_p_mc = _monte_carlo_range_p_value(
        y, yerr, n_sim=n_sim, random_state=random_state
    )

    result.update({
        "weighted_mean": weighted_mean,
        "chi2_const": chi2_const,
        "dof_const": dof_const,
        "p_const": p_const,
        "F_var": fvar_result["F_var"],
        "F_var_err": fvar_result["F_var_err"],
        "F_var_lower": fvar_result["F_var_lower"],
        "max_adjacent_sigma": max_adjacent_sigma,
        "range_CU": np.max(y) - np.min(y),
        "range_p_mc": range_p_mc,
        "support_fvar": fvar_result["F_var_lower"] > -fvar_lower_tol,
        "support_adjacent": max_adjacent_sigma > adjacent_sigma_threshold,
        "support_range_mc": range_p_mc < range_mc_threshold,
        "candidate_class": "pending_q_value",
    })

    return result


def flag_candidate_groups(
    daten_gruppiert,
    n_sim=10000,
    random_state=42,
    min_points=3,
    min_strong_points=5,
    q_const_threshold=0.30,
    possible_q_const_threshold=0.50,
    fvar_lower_tol=0.03,
    adjacent_sigma_threshold=2.5,
    range_mc_threshold=0.05,
):
    """
    Bewertet alle Gruppen und fuegt Benjamini-Hochberg q-Werte hinzu.

    daten_gruppiert kann entweder ein Dict {gruppe: DataFrame} oder ein
    DataFrame mit Spalte 'Gruppe' sein.

    Candidate tiers:
        strong_candidate:
            q_const < 0.01
            range_p_mc < 0.01
            n >= min_strong_points
        candidate:
            q_const < q_const_threshold
            and at least one support metric
        possible_candidate:
            q_const < possible_q_const_threshold
            and at least one support metric

    Support metrics:
        q_const < q_const_threshold
        F_var_lower > -fvar_lower_tol
        max_adjacent_sigma > adjacent_sigma_threshold
        range_p_mc < range_mc_threshold
    """
    if isinstance(daten_gruppiert, pd.DataFrame):
        grouped_items = list(daten_gruppiert.groupby("Gruppe"))
    else:
        grouped_items = list(daten_gruppiert.items())

    rng = np.random.default_rng(random_state)
    rows = []

    for gruppe, daten in grouped_items:
        x = np.asarray(daten["MJD"], dtype=float)
        y = np.asarray(daten["CU"], dtype=float)
        yerr = np.asarray(daten.get("CU_err", np.ones_like(y)), dtype=float)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        metrics = flag_single_candidate_group(
            x,
            y,
            yerr,
            n_sim=n_sim,
            random_state=seed,
            min_points=min_points,
            min_strong_points=min_strong_points,
            fvar_lower_tol=fvar_lower_tol,
            adjacent_sigma_threshold=adjacent_sigma_threshold,
            range_mc_threshold=range_mc_threshold,
        )
        metrics.update({
            "gruppe": gruppe,
            "x_data": x,
            "y_data": y,
            "y_err_data": yerr,
        })
        if "MJD_Nacht" in daten.columns:
            mjd_nights = pd.unique(daten["MJD_Nacht"].dropna())
            metrics["mjd_nacht"] = (
                int(mjd_nights[0]) if len(mjd_nights) == 1 else tuple(mjd_nights)
            )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    if results.empty:
        return results

    results["q_const"] = _benjamini_hochberg(results["p_const"].to_numpy())
    results["passes_possible_q_const"] = (
        results["q_const"] < possible_q_const_threshold
    )
    results["passes_q_const"] = results["q_const"] < q_const_threshold
    results["passes_strong_q_const"] = results["q_const"] < 0.01
    results["passes_strong_range_mc"] = results["range_p_mc"] < 0.01
    results["passes_min_strong_points"] = results["n"] >= min_strong_points
    support = (
        results["support_fvar"]
        | results["support_adjacent"]
        | results["support_range_mc"]
    )
    testable = results["n"] >= min_points
    possible_mask = (
        (results["q_const"] < possible_q_const_threshold)
        & support
        & testable
    )
    candidate_mask = (
        (results["q_const"] < q_const_threshold)
        & support
        & testable
    )
    strong_mask = (
        (results["q_const"] < 0.01)
        & (results["range_p_mc"] < 0.01)
        & (results["n"] >= min_strong_points)
        & testable
    )
    results["is_candidate"] = possible_mask

    results["candidate_class"] = "not_candidate"
    results.loc[~testable, "candidate_class"] = "not_testable"
    results.loc[
        (results["p_const"] < 0.003) & testable & ~possible_mask,
        "candidate_class"
    ] = "review"
    results.loc[possible_mask, "candidate_class"] = "possible_candidate"
    results.loc[candidate_mask, "candidate_class"] = "candidate"
    results.loc[strong_mask, "candidate_class"] = "strong_candidate"
    results["candidate_criteria"] = results.apply(
        _candidate_criteria_labels, axis=1
    )

    ordered_columns = [
        "gruppe", "candidate_class", "is_candidate", "n", "duration_hours",
        "mjd_nacht",
        "weighted_mean", "chi2_const", "dof_const", "p_const", "q_const",
        "F_var", "F_var_err", "F_var_lower", "max_adjacent_sigma",
        "range_CU", "range_p_mc", "support_fvar", "support_adjacent",
        "support_range_mc", "passes_possible_q_const", "passes_q_const",
        "passes_strong_q_const", "passes_strong_range_mc",
        "passes_min_strong_points", "candidate_criteria", "x_data", "y_data",
        "y_err_data",
    ]
    ordered_columns = [col for col in ordered_columns if col in results.columns]
    return results[ordered_columns].sort_values(
        ["is_candidate", "q_const", "range_p_mc"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


# -------------------------------
# Erzeugt AICc Tabelle
# -------------------------------

def akaike_table(aicc_dict):
    """
    Erzeugt eine Tabelle mit AICc, ΔAICc und Akaike-Gewichten.
    aicc_dict = {modellname: AICc}
    """

    df = pd.DataFrame.from_dict(aicc_dict, orient="index", columns=["AICc"])

    # Bestes AICc
    min_aicc = df["AICc"].min()
    df["Delta"] = df["AICc"] - min_aicc

    # Relative Likelihood
    rel = np.exp(-0.5 * df["Delta"])
    rel = rel.replace([np.inf, -np.inf], 0).fillna(0)

    s = rel.sum()
    if s == 0:
        df["weight"] = rel * 0
    else:
        df["weight"] = rel / s

    return df.sort_values("weight", ascending=False)


# ---------------------------------------------
# Hilfsfunktionen: AIC/AICc/BIC aus Log-Likelihood
# ---------------------------------------------

def compute_info_criteria_with_errors(resid, yerr, k):
    """
    Berechnet AIC, AICc und BIC unter Berücksichtigung der Messfehler yerr.
    resid = y - f(x)
    yerr  = Fehler der Daten
    k     = Anzahl freier Parameter
    """

    n = len(resid)
    yerr = np.asarray(yerr)

    # Verhindern, dass Fehler = 0 ist
    positive_yerr = yerr[yerr > 0]
    if len(positive_yerr) == 0:
        return np.inf, np.inf, np.inf, -np.inf
    yerr = np.where(yerr <= 0, np.min(positive_yerr), yerr)

    # Log-Likelihood für bekannte Fehler
    logL = -0.5 * np.sum(
        (resid**2) / (yerr**2) + np.log(2 * np.pi * (yerr**2))
    )

    AIC = 2 * k - 2 * logL
    BIC = np.log(n) * k - 2 * logL

    # Korrektur für kleine Stichproben
    if n > k + 1:
        AICc = AIC + (2 * k * (k + 1)) / (n - k - 1)
    else:
        AICc = np.inf

    return AIC, AICc, BIC, logL


# ---------------------------------------------
# Robuste Fit-Funktion: zuerst curve_fit, fallback least_squares
# ---------------------------------------------

def fit_nonlinear_robust(model_func, x, y, yerr, p0, bounds=(-np.inf, np.inf)):
    """
    Robuste Fit-Funktion mit Fehlergewichtung.
    yerr MUST NOT be None.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)

    # Fehler dürfen nicht 0 sein
    if np.any(yerr <= 0):
        min_pos = np.min(yerr[yerr > 0])
        yerr = np.where(yerr <= 0, min_pos, yerr)

    n = len(y)

    # ---------------------------------------------------
    # 1) Versuch: curve_fit mit Fehlergewichten
    # ---------------------------------------------------
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(
                model_func, x, y,
                sigma=yerr,              # <- FEHLER BERÜCKSICHTIGT
                absolute_sigma=True,      # <- sigma wird ernst genommen
                p0=p0,
                bounds=bounds,
                maxfev=50000
            )

        y_pred = model_func(x, *popt)
        resid = y - y_pred

        AIC, AICc, BIC, logL = compute_info_criteria_with_errors(resid, yerr, k=len(popt))

        # Fehlerabschätzung
        try:
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            perr = None

        return {
            "params": popt,
            "errors": perr,
            "AIC": AIC,
            "AICc": AICc,
            "BIC": BIC,
            "logL": logL
        }

    except Exception:
        pass     # gehe zum Fallback


    # ---------------------------------------------------
    # 2) Fallback: least_squares mit robustem Loss
    # ---------------------------------------------------
    try:
        def weighted_residuals(p):
            return (y - model_func(x, *p)) / yerr   # <- Fehler hier auch berücksichtigt!!!

        ls = least_squares(
            weighted_residuals,
            x0=p0,
            bounds=bounds,
            max_nfev=50000,
            loss="soft_l1",
            f_scale=1.0
        )

        if not ls.success:
            return {
                "params": None,
                "errors": None,
                "AIC": np.inf,
                "AICc": np.inf,
                "BIC": np.inf,
                "logL": -np.inf
            }

        popt = ls.x
        resid = (y - model_func(x, *popt))

        # AIC erneut mit Fehlern
        AIC, AICc, BIC, logL = compute_info_criteria_with_errors(resid, yerr, k=len(popt))

        # Kovarianz approx
        try:
            J = ls.jac
            JTJ = J.T @ J
            cov = np.linalg.pinv(JTJ)
            perr = np.sqrt(np.abs(np.diag(cov)))
        except Exception:
            perr = None

        return {
            "params": popt,
            "errors": perr,
            "AIC": AIC,
            "AICc": AICc,
            "BIC": BIC,
            "logL": logL
        }

    except Exception:
        return {
            "params": None,
            "errors": None,
            "AIC": np.inf,
            "AICc": np.inf,
            "BIC": np.inf,
            "logL": -np.inf
        }


# -------------------------------------------------------------
# Klassifikation MIT Goodness-of-Fit (χ²-Test)
# -------------------------------------------------------------
def classify_model(table, results, x, y, yerr):
    """
    Klassifiziert das beste Modell nach Akaike und prüft zusätzlich
    mit einem χ²-Test, ob es die Daten überhaupt beschreibt.
    Gibt eine String-Klassifikation zurück.
    """

    # --- Bestes Modell per Akaike ---
    best = table["Akaike_weight"].idxmax()
    w = table.loc[best, "Akaike_weight"]

    # Schwaches Akaike-Gewicht → kein Modell klar
    if w < 0.5:
        return "unklar/rauschen"

    # Parameter des besten Modells
    params = results[best]["params"]
    if params is None:
        return "unklar/rauschen"

    # Modellfunktionen
    model_funcs = {
        "konstant": lambda x,p: np.full_like(x, p[0], dtype=float),
        "linear":   lambda x,p: p[0] + p[1]*x,
        "quadratisch": lambda x,p: p[0] + p[1]*x + p[2]*x**2,
        "gauss":    lambda x,p: p[0] * np.exp(-(x-p[1])**2/(2*p[2]**2)),
        "lorentz":  lambda x,p: p[0] / (1 + ((x-p[1])/p[2])**2)
    }

    if best not in model_funcs:
        return "unklar"

    # --- χ²-Goodness-of-Fit ---
    yfit = model_funcs[best](x, params)
    chi2_val = np.sum(((y - yfit) / yerr)**2)

    k = len(params)
    n = len(y)
    dof = n - k

    # Zu kleine Freiheitsgrade → unsicher
    if dof < 1:
        return "unklar/rauschen"

    # p-Wert
    p_value = 1 - chi2.cdf(chi2_val, dof)

    # Modell beschreibt Daten nicht gut
    if p_value < 0.05:
        return "unklar/rauschen"

    # --- Klassifikation nach Modelltyp ---
    if best in ["gauss", "lorentz"]:
        amplitude = params[0]
        width = params[2] if len(params) > 2 else np.nan
        if amplitude <= 0 or width <= 0 or not np.isfinite(width):
            return "unklar/rauschen"
        return "peak"

    if best == "quadratisch":
        return "gekrümmt"
    if best == "linear":
        return "linearer trend"
    if best == "konstant":
        return "konstant"

    return "unklar"



# ---------------------------------------------
# Fit aller Modelle: lineare (WLS) + gauss (robust)
# ---------------------------------------------

#models_formula = {
#    "konstant": "y ~ 1",
#    "linear": "y ~ x"
#}

# def fit_all_models(x, y, yerr):
#     """
#     Fitet alle Modelle (linear + gauss) und berücksichtigt Messfehler yerr.
#     Berechnet AIC/AICc/BIC, Akaike-Gewichte UND Chi²-Test.
#     """
#     results = {}
#     df = pd.DataFrame({"x": x, "y": y, "yerr": yerr})
#     n = len(y)
    

#     # ---------------------------------------------------------
#     # LINEARE MODELLE (WLS mit Gewichten = 1 / yerr^2)
#     # ---------------------------------------------------------
#     for name, formula in models_formula.items():
#         try:
#             weights = 1.0 / (yerr ** 2)

#             fit = smf.wls(formula=formula, data=df, weights=weights).fit()

#             k = int(fit.df_model + 1)
#             AIC = fit.aic
#             BIC = fit.bic
#             AICc = AIC + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.inf

            

#             # Modell speichern
#             results[name] = {
#                 "params": fit.params.values,
#                 "errors": fit.bse.values,
#                 "AIC": AIC,
#                 "AICc": AICc,
#                 "BIC": BIC
#             }

            


#             # ---- Chi² ----
#             yfit = fit.fittedvalues.values
#             residuals = (y - yfit)
#             chi2_val = np.sum((residuals / yerr)**2)
#             dof = n - len(fit.params)
#             p_val = 1 - chi2.cdf(chi2_val, dof)

#             results[name]["chi2"] = chi2_val
#             results[name]["chi2_dof"] = dof
#             results[name]["chi2_p"] = p_val

#         except Exception:
#             results[name] = {
#                 "params": None, "errors": None,
#                 "AIC": np.inf, "AICc": np.inf, "BIC": np.inf,
#                 "chi2": np.inf, "chi2_dof": 0, "chi2_p": 0
#             }

        

#     # ---------------------------------------------------------
#     # NICHTLINEARE MODELLE → robustes curve_fit
#     # ---------------------------------------------------------
#     amp_guess = np.max(y) - np.min(y)
#     offset_guess = np.median(y)
#     mu_guess = np.mean(x)
#     sigma_guess = np.std(x) if np.std(x) > 0 else 1.0

#     lower = [-np.inf, np.min(x) - 10*np.ptp(x), 1e-6, -np.inf]
#     upper = [ np.inf, np.max(x) + 10*np.ptp(x),  np.ptp(x)*10, np.inf]
#     bounds = (lower, upper)

#     # ---- Gauss-Modell fitten ----
#     res = fit_nonlinear_robust(
#         gauss, x, y, p0=[amp_guess, mu_guess, sigma_guess, offset_guess],
#         bounds=bounds, yerr=yerr
#     )
#     results["gauss"] = res

#     # Chi² auch für Gauss
#     if res["params"] is not None:
#         yfit = gauss(x, *res["params"])
#         chi2_val = np.sum(((y - yfit) / yerr)**2)
#         dof = n - len(res["params"])
#         p_val = 1 - chi2.cdf(chi2_val, dof)

#         results["gauss"]["chi2"] = chi2_val
#         results["gauss"]["chi2_dof"] = dof
#         results["gauss"]["chi2_p"] = p_val
#     else:
#         results["gauss"]["chi2"] = np.inf
#         results["gauss"]["chi2_dof"] = 0
#         results["gauss"]["chi2_p"] = 0

#     # ---------------------------------------------------------
#     # Akaike-Gewichte (AICc)
#     # ---------------------------------------------------------
#     AICc_vals = np.array([v["AICc"] for v in results.values()], dtype=float)

#     if np.all(~np.isfinite(AICc_vals)):
#         # alle fits kaputt
#         for name in results.keys():
#             results[name]["Akaike_weight"] = 0.0

#         table = pd.DataFrame({
#             m: {"AIC": results[m]["AIC"],
#                 "AICc": results[m]["AICc"],
#                 "BIC": results[m]["BIC"],
#                 "Akaike_weight": results[m].get("Akaike_weight", 0.0)}
#             for m in results.keys()
#         }).T
#         return results, table

#     min_AICc = np.nanmin(AICc_vals)
#     rel = np.exp(-0.5 * (AICc_vals - min_AICc))
#     rel[~np.isfinite(rel)] = 0.0
#     sum_rel = np.sum(rel)
#     weights = rel / sum_rel if sum_rel > 0 else np.zeros_like(rel)

#     for (name, w) in zip(results.keys(), weights):
#         results[name]["Akaike_weight"] = float(w)

#     # Zusammenfassung als Tabelle
#     table = pd.DataFrame({
#         m: {"AIC": results[m]["AIC"],
#             "AICc": results[m]["AICc"],
#             "BIC": results[m]["BIC"],
#             "Akaike_weight": results[m]["Akaike_weight"],
#             "chi2": results[m]["chi2"],
#             "chi2_dof": results[m]["chi2_dof"],
#             "chi2_p": results[m]["chi2_p"]}
#         for m in results.keys()
#     }).T

#     return results, table


# ---------------------------------------------
# Fit aller Modelle: lineare (WLS) + gauss (robust)
# ---------------------------------------------

# models_formula = {
#     "konstant": "y ~ 1",
#     "linear": "y ~ x"
# }


def fit_all_models(x, y, yerr):
    """
    Fit linear and Gaussian models to data (x, y) with errors yerr.
    Returns fit results and a summary table with AIC, BIC, AICc, and Akaike weights.
    Information criteria are computed with the same Gaussian error likelihood
    for all models, so AIC/AICc/BIC are comparable across model families.
    """
    results = {}
    n = len(y)
    df = pd.DataFrame({"x": x, "y": y, "yerr": yerr})

    # ------------------------------
    # 1) Lineare Modelle (WLS)
    # ------------------------------
    models_formula = {
        "konstant": "y ~ 1",
        "linear": "y ~ x"
    }

    for name, formula in models_formula.items():
        try:
            fit = smf.wls(formula=formula, data=df, weights=1/yerr**2).fit()
            params = fit.params.values
            errors = fit.bse.values
            k = len(params)

            y_pred = fit.predict(df)
            resid = y - y_pred
            AIC, AICc, BIC, logL = compute_info_criteria_with_errors(
                resid, yerr, k
            )

            results[name] = {
                "params": params,
                "errors": errors,
                "AIC": AIC,
                "AICc": AICc,
                "BIC": BIC,
                "logL": logL,
            }

        except Exception as e:
            print(f"[WARN] WLS für Modell '{name}' fehlgeschlagen:", e)
            results[name] = {
                "params": None, "errors": None,
                "AIC": np.inf, "AICc": np.inf, "BIC": np.inf, "logL": -np.inf,
            }

    # ------------------------------
    # 2) Gaussian Fit
    # ------------------------------
    # Startwerte für den Fit
    p0 = [np.ptp(y), np.mean(x), np.std(x), np.median(y)]
    bounds = (
        [-np.inf, np.min(x)-5*np.ptp(x), 1e-6, -np.inf],
        [ np.inf, np.max(x)+5*np.ptp(x), 5*np.ptp(x), np.inf]
    )

    gauss_result = fit_nonlinear_robust(gauss, x, y, yerr, p0=p0, bounds=bounds)
    results["gauss"] = gauss_result

    # ------------------------------
    # Akaike-Gewichte
    # ------------------------------
    AICc_vals = np.array([v.get("AICc", np.inf) for v in results.values()])
    finite = np.isfinite(AICc_vals)

    if np.any(finite):
        min_AICc = np.min(AICc_vals[finite])
        rel = np.zeros_like(AICc_vals, dtype=float)
        rel[finite] = np.exp(-0.5 * (AICc_vals[finite] - min_AICc))
        weights = rel / np.sum(rel) if np.sum(rel) > 0 else np.zeros_like(rel)
    else:
        weights = np.zeros_like(AICc_vals, dtype=float)

    for name, w in zip(results.keys(), weights):
        results[name]["Akaike_weight"] = w

    table = pd.DataFrame(results).T
    return results, table


# -------------------------------------------------------------
# Klassifikation mit Goodness-of-Fit NUR für das konstante Modell
# -------------------------------------------------------------
def classify_model2(table, results, x, y, yerr):
    """
    Klassifiziert das beste Modell nach Akaike.
    Prüft aber per χ²-Test ausschließlich, ob das konstante Modell
    die Daten beschreibt (Nullhypothesen-Test).
    Gibt eine String-Klassifikation zurück.
    """


    # --- Akaike: bestes Modell bestimmen ---
    best = table["Akaike_weight"].idxmax()
    w = table.loc[best, "Akaike_weight"]

    if w < 0.5:
        return "unklar/rauschen (w)"

    # ------------------------------------------------------------------
    # χ²-Goodness-of-Fit: TESTE NUR DAS KONSTANTE MODELL
    # ------------------------------------------------------------------
    # Nullmodell-Wert = Mittelwert ist typischerweise sinnvoll
    # (alternativ kannst du auch results["konstant"]["params"] nehmen)
    const_param = np.array([np.average(y, weights=1/yerr**2)])
    yfit_const = np.full_like(y, const_param[0])

    chi2_const = np.sum(((y - yfit_const) / yerr)**2)

    k0 = 1  # nur ein Parameter im konstanten Modell
    n = len(y)
    dof0 = n - k0

    if dof0 < 1:
        return "unklar/rauschen (Freiheitsgrade)"

    # p-Wert NUR für das konstante Modell
    p_value = 1 - chi2.cdf(chi2_const, dof0)

    # Falls selbst das konstante Modell deutlich verworfen wird:
    if p_value < 0.003:
        # → Die Daten sind eindeutig nicht konstant → reales Signal
        # Danach klassifizieren wir ganz normal nach Akaike
        pass
    else:
        # Daten sind kompatibel mit Konstanz → KEIN Signal
        return "konstant/rauschen (Model)"

    # ------------------------------------------------------------------
    # AB HIER: Klassifikation nach dem bestgewählten Modell (Akaike)
    # ------------------------------------------------------------------
    params = results[best]["params"]
    if params is None:
        return "unklar/rauschen"

    if best == "gauss" or best == "lorentz":
        amp = params[0]
        width = params[2] if len(params) > 2 else np.nan
        if amp <= 0 or not np.isfinite(width) or width <= 0:
            return "unklar/rauschen"
        return "peak"

    if best == "quadratisch":
        return "gekrümmt"

    if best == "linear":
        return "linearer trend"

    if best == "konstant":
        return "konstant"

    return "unklar"




def process_single_group(args):
    gruppe, daten = args

    x = np.asarray(daten["MJD"])
    y = np.asarray(daten["CU"])
    yerr = np.asarray(daten.get("CU_err", np.ones_like(y)))

    if len(y) < 3:
        return {
            "gruppe": gruppe,
            "n": len(y),
            "x_data": x,
            "y_data": y,
            "y_err_data": yerr,
            "best_model": None,
            "classification": "zu wenige daten",
            "AICc": np.nan,
            "Akaike_weight": np.nan,
            "params": None,
            "all_params": {},
            "errors": None,
            "akaike_table": None
        }

    results, table = fit_all_models(x, y, yerr)

    best_model = table["Akaike_weight"].idxmax()
    classification = classify_model2(
        table, results, x, y, yerr
    )

    best_params = results[best_model]["params"]
    all_params = {
    model_name: res["params"]
    for model_name, res in results.items()
    if res.get("params") is not None
}
    best_errors = results[best_model].get("errors", None)

    aicc_dict = {
        m: res["AICc"]
        for m, res in results.items()
        if res["AICc"] is not None
    }

    akaike_df = akaike_table(aicc_dict)

    return {
        "gruppe": gruppe,
        "n": len(y),
        "x_data": x,
        "y_data": y,
        "y_err_data": yerr,
        "best_model": best_model,
        "classification": classification,
        "AICc": table.loc[best_model, "AICc"],
        "Akaike_weight": table.loc[best_model, "Akaike_weight"],
        "params": best_params,
        "all_params": all_params, 
        "errors": best_errors,
        "akaike_table": akaike_df
    }



def process_all_groups_parallel(daten_gruppiert, max_workers=None):

    if max_workers is None:
        max_workers = os.cpu_count()

    args_list = [
        (gruppe, daten)
        for gruppe, daten in daten_gruppiert.items()
    ]

    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in tqdm(
            executor.map(process_single_group, args_list),
            total=len(args_list),
            desc="Fitting Gruppen"
        ):
            results.append(res)

    return pd.DataFrame(results)





def create_group_plots_pdf(
    alle_results,
    pdf_filename="alle_gruppen_plots_default.pdf"
):
    """
    Erstellt für jede Gruppe einen Plot mit allen gespeicherten Fit-Modellen
    und speichert alles in einer mehrseitigen PDF.
    Es werden KEINE Modelle neu gefittet (schnell).
    """

    # ---------------------------------------------------------
    # Modellfunktionen
    # ---------------------------------------------------------
    def f_konstant(x, p):
        return np.full_like(x, p[0], dtype=float)

    def f_linear(x, p):
        return p[0] + p[1] * x

    def f_gauss(x, p):
        A, mu, sigma, offset = p
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset

    fit_functions = {
        "konstant": f_konstant,
        "linear": f_linear,
        "gauss": f_gauss
    }

    needed_params = {
        "konstant": 1,
        "linear": 2,
        "gauss": 4,
    }

    # ---------------------------------------------------------
    # PDF erstellen
    # ---------------------------------------------------------
    with PdfPages(pdf_filename) as pdf:

        for _, eintrag in alle_results.iterrows():

            gruppe = eintrag["gruppe"]
            best_model = eintrag["best_model"]
            classification = eintrag["classification"]
            params = eintrag["params"]

            if best_model is None or params is None:
                continue

            x = np.asarray(eintrag["x_data"])
            y = np.asarray(eintrag["y_data"])
            yerr = np.asarray(eintrag.get("y_err_data", np.ones_like(y)))

            plt.figure(figsize=(10, 6))
            plt.errorbar(x, y, yerr=yerr,
                         fmt='o', color="black", label="Daten")

            xfine = np.linspace(min(x), max(x), 500)

            # Nur BESTES Modell plotten (optional schneller & übersichtlicher)
            params = np.array(params).flatten()

            if best_model in fit_functions:
                if len(params) >= needed_params[best_model]:
                    try:
                        yfit = fit_functions[best_model](xfine, params)
                        plt.plot(
                            xfine,
                            yfit,
                            linewidth=3,
                            label=f"{best_model} (BEST)"
                        )
                    except Exception:
                        continue

            plt.title(f"Gruppe {gruppe} – Classification: {classification}")
            plt.xlabel("MJD")
            plt.ylabel("CU")
            plt.legend()
            plt.grid(alpha=0.3)

            pdf.savefig()
            plt.close()

    print(f"Alle Plots wurden in '{pdf_filename}' gespeichert.")


    

def f_konstant(x, p):
    return np.full_like(x, p[0], dtype=float)

def f_linear(x, p):
    return p[0] + p[1] * x

def f_gauss(x, p):
    A, mu, sigma, offset = p
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset


fit_functions = {
    "konstant": f_konstant,
    "linear": f_linear,
    "gauss": f_gauss
}

needed_params = {
    "konstant": 1,
    "linear": 2,
    "gauss": 4,
}



def create_single_plot(eintrag):

    if eintrag["best_model"] is None:
        return None

    x = np.asarray(eintrag["x_data"])
    y = np.asarray(eintrag["y_data"])
    yerr = np.asarray(eintrag["y_err_data"])

    if len(x) == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, yerr=yerr, fmt='o', color="black", label="Daten")

    xfine = np.linspace(min(x), max(x), 500)

    best_model = eintrag["best_model"]
    all_params = eintrag.get("all_params", {})

    if not isinstance(all_params, dict) or not all_params:
        params = eintrag.get("params", None)
        if best_model in fit_functions and params is not None:
            try:
                params_array = np.array(params, dtype=float).flatten()
                if np.all(np.isfinite(params_array)):
                    all_params = {best_model: params_array}
                else:
                    all_params = {}
            except Exception:
                all_params = {}
        else:
            all_params = {}

    if not all_params:
        plt.close(fig)
        return None

    for model_name, params in all_params.items():

        if model_name not in fit_functions:
            continue

        params = np.array(params).flatten()

        if len(params) < needed_params[model_name]:
            continue

        try:
            yfit = fit_functions[model_name](xfine, params)

            if model_name == best_model:
                ax.plot(
                    xfine,
                    yfit,
                    linewidth=3,
                    label=f"{model_name} (BEST)"
                )
            else:
                ax.plot(
                    xfine,
                    yfit,
                    linewidth=1.5,
                    alpha=0.5,
                    label=model_name
                )

        except Exception:
            continue

    ax.set_title(
        f"Gruppe {eintrag['gruppe']} – "
        f"Classification: {eintrag['classification']}"
    )
    ax.set_xlabel("MJD")
    ax.set_ylabel("CU")
    #ax.legend()
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    return fig





def create_group_plots_pdf_parallel(
    df_results,
    pdf_filename="alle_gruppen_plots_parallel.pdf",
    max_workers=None
):

    if max_workers is None:
        max_workers = os.cpu_count()

    records = df_results.to_dict("records")

    with PdfPages(pdf_filename) as pdf:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            for fig in tqdm(
                executor.map(create_single_plot, records),
                total=len(records),
                desc="Erstelle & speichere Plots"
            ):
                if fig is not None:
                    pdf.savefig(fig)
                    plt.close(fig)

    print(f"PDF gespeichert: {pdf_filename}")



def calculate_nightly_delta(df_results, daten_nightly):
    """
    Berechnet für jede Gruppe und jeden nightly-Eintrag die Differenzen
    zwischen Max und Min der y-Daten der Gruppe sowie die Fehler.

    Args:
        df_results (pd.DataFrame): DataFrame mit Spalten 'gruppe', 'x_data', 'y_data', 'y_err_data', 'n'
        daten_nightly (pd.DataFrame): DataFrame mit nightly Werten, mindestens Spalten 'MJD' und 'CU'

    Returns:
        pd.DataFrame: DataFrame mit allen nightly-Deltas
    """
    nightly_delta = []

    for gruppe, df_gruppe in df_results.groupby("gruppe"):

        for _, nightly in daten_nightly.iterrows():

            # Arrays aus der Gruppe zusammenfügen
            alle_werte = df_gruppe["y_data"].iloc[0]
            x_werte = df_gruppe["x_data"].iloc[0]
            number = df_gruppe["n"].iloc[0]
            alle_werte_err = df_gruppe["y_err_data"].iloc[0]

            max_idx = np.argmax(alle_werte)
            min_idx = np.argmin(alle_werte)

            max_err = alle_werte_err[max_idx]
            min_err = alle_werte_err[min_idx]

            delta_err = np.sqrt(max_err**2 + min_err**2)

            if number != 1 and nightly["MJD"] >= min(x_werte) and nightly["MJD"] <= max(x_werte):

                nightly_delta.append({
                    "Gruppe": gruppe,
                    "n": number,
                    "nightly_x": nightly["MJD"],
                    "binning_x": x_werte,
                    "nightly_CU": nightly["CU"],
                    "binning_CU": alle_werte,
                    "binning_CU_err": alle_werte_err,
                    "binning_CU_avg": np.mean(alle_werte),
                    "Min": np.min(alle_werte),
                    "Max": np.max(alle_werte),
                    "Delta": np.max(alle_werte) - np.min(alle_werte),
                    "Delta_err": delta_err
                })

    return pd.DataFrame(nightly_delta)
