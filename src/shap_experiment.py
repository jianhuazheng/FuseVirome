#!/usr/bin/env python3
"""FuseVirome - SHAP Interpretability Experiment

This standalone script explains the final 29-feature Random Forest model
using SHAP (SHapley Additive exPlanations). It generates global summary
plots, beeswarm plots, importance-metric comparisons, dependence plots,
and local waterfall plots for representative samples.

Run:
    python src/shap_experiment.py

Output:
    outputs/figures/Figure8_SHAP_Global_Summary.png
    outputs/figures/Figure9_SHAP_Beeswarm.png
    outputs/figures/Figure10_SHAP_Importance_Comparison.png
    outputs/figures/Figure11_SHAP_Dependence_TopFeatures.png
    outputs/figures/Figure12_SHAP_Local_Waterfall.png
    outputs/figures/shap_importance.csv
    outputs/figures/shap_values.npy
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow importing fusevirome.py regardless of the working directory.
_SYS_PATH_INSERTED = False
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _SYS_PATH_INSERTED = True

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance

from fusevirome import (
    load_data,
    stage1_null_removal,
    stage2_fused_exclusion,
    RANDOM_SEED,
    PLOT_FIG_SIZE,
    PLOT_DPI,
    PLOT_FONT_FAMILY,
    PLOT_TITLE_SIZE,
    PLOT_PANEL_TITLE_SIZE,
    PLOT_LABEL_SIZE,
    PLOT_LEGEND_SIZE,
    PLOT_SMALL_SIZE,
    configure_plot_style,
    save_plot,
)

import shap

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures', '')

np.random.seed(RANDOM_SEED)
configure_plot_style()

# Reproducible colour palette consistent with the existing manuscript figures.
COLOR_PALETTE = {
    'RF': '#E74C3C',
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'neutral': '#6C757D',
}


# =============================================================================
# 1. DATA PREPARATION
# =============================================================================

def get_final_datasets():
    """Reproduce the final 29-feature train/test split from the main pipeline."""
    X_train_raw, y_train, X_test_raw, y_test = load_data()

    r1 = stage1_null_removal(X_train_raw)
    X1 = X_train_raw.drop(columns=r1)
    r2 = stage2_fused_exclusion(X1)
    X2 = X1.drop(columns=r2)
    final_features = X2.columns.tolist()

    train_medians = X2.median()
    X_train = X2.fillna(train_medians)
    X_test = X_test_raw[final_features].fillna(train_medians)

    print(f'  Feature refinement: {X_train_raw.shape[1]} -> {X1.shape[1]} -> {X2.shape[1]}')
    print(f'  Final features ({len(final_features)}): {final_features}')
    return X_train, y_train, X_test, y_test, final_features


def train_final_rf(X_train, y_train):
    """Train the final Random Forest model matching fusevirome.py."""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


# =============================================================================
# 2. SHAP VALUE COMPUTATION
# =============================================================================

def compute_shap_values(rf_model, X_train, X_test, final_features):
    """Compute TreeExplainer SHAP values for the positive class."""
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    # TreeExplainer returns different shapes across SHAP versions:
    #   - list [class_0, class_1] for binary (older versions)
    #   - 3D array (n_samples, n_features, n_classes) for newer versions
    #   - 2D array (n_samples, n_features) for single-output models
    if isinstance(shap_values, list):
        shap_values_positive = shap_values[1]
        expected_value = explainer.expected_value[1]
    elif shap_values.ndim == 3:
        shap_values_positive = shap_values[..., 1]
        if np.ndim(explainer.expected_value) > 0:
            expected_value = float(explainer.expected_value[..., 1])
        else:
            expected_value = float(explainer.expected_value)
    else:
        shap_values_positive = shap_values
        expected_value = float(explainer.expected_value)

    # Sanity checks.
    assert shap_values_positive.shape == X_test.shape, (
        f'SHAP value shape mismatch: {shap_values_positive.shape} vs {X_test.shape}'
    )
    assert len(final_features) == X_test.shape[1], 'Feature list mismatch.'

    # Sum-to-prediction sanity check on first 5 samples.
    margins = expected_value + shap_values_positive.sum(axis=1)
    probs = rf_model.predict_proba(X_test)[:5, 1]
    # Convert probability margin via logit for comparison.
    prob_margins = np.log(probs / (1 - probs + 1e-12) + 1e-12)
    print(f'  SHAP expected value (positive class): {expected_value:.4f}')
    print(f'  Sum-to-prediction check (first 5):')
    for i in range(5):
        print(f'    Sample {i}: SHAP margin={margins[i]:.4f}, logit(prob)={prob_margins[i]:.4f}')

    explanation = shap.Explanation(
        values=shap_values_positive,
        base_values=np.full(len(X_test), expected_value, dtype=float),
        data=X_test.values,
        feature_names=final_features,
    )
    return explainer, shap_values_positive, explanation, expected_value


def compute_importance_metrics(rf_model, X_test, y_test, shap_values_positive, final_features):
    """Compute Gini, permutation, and SHAP importances for comparison."""
    # Gini importance.
    gini = pd.Series(rf_model.feature_importances_, index=final_features)

    # Permutation importance (30 repeats, AUC scoring).
    pi = permutation_importance(
        rf_model, X_test, y_test,
        n_repeats=30,
        random_state=RANDOM_SEED,
        scoring='roc_auc',
        n_jobs=-1,
    )
    pi_mean = pd.Series(pi.importances_mean, index=final_features)

    # SHAP importance.
    shap_imp = pd.Series(np.abs(shap_values_positive).mean(axis=0), index=final_features)

    imp_df = pd.DataFrame({
        'Feature': final_features,
        'Gini': gini.values,
        'Permutation': pi_mean.values,
        'SHAP': shap_imp.values,
    })

    # Normalise each metric to [0, 1] for side-by-side comparison.
    for col in ['Gini', 'Permutation', 'SHAP']:
        col_max = imp_df[col].max()
        if col_max > 0:
            imp_df[f'{col}_norm'] = imp_df[col] / col_max
        else:
            imp_df[f'{col}_norm'] = 0.0

    imp_df = imp_df.sort_values('SHAP', ascending=False).reset_index(drop=True)
    return imp_df


# =============================================================================
# 3. FIGURE GENERATION
# =============================================================================

def plot_global_summary(explanation, out_path, n_top=15):
    """Figure 8: Global mean absolute SHAP value bar plot."""
    fig = plt.figure(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    shap.plots.bar(explanation, max_display=n_top, show=False)
    fig = plt.gcf()
    fig.set_size_inches(*PLOT_FIG_SIZE, forward=True)
    plt.title('SHAP Feature Importance (Mean |SHAP Value|)', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, out_path)
    plt.close()
    print(f'  Figure 8 saved: {out_path}')


def plot_beeswarm(explanation, out_path, n_top=15):
    """Figure 9: Beeswarm plot showing feature impact direction and magnitude."""
    fig = plt.figure(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    shap.plots.beeswarm(explanation, max_display=n_top, show=False)
    fig = plt.gcf()
    fig.set_size_inches(*PLOT_FIG_SIZE, forward=True)
    plt.title('SHAP Beeswarm Plot (Feature Value Impact)', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, out_path)
    plt.close()
    print(f'  Figure 9 saved: {out_path}')


def plot_importance_comparison(imp_df, out_path, n_top=10):
    """Figure 10: Side-by-side comparison of Gini, permutation, and SHAP importance."""
    top_df = imp_df.head(n_top).sort_values('SHAP_norm', ascending=True)
    features = top_df['Feature'].values

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    y_pos = np.arange(len(features))
    bar_height = 0.25

    ax.barh(
        y_pos - bar_height,
        top_df['Gini_norm'].values,
        height=bar_height,
        color='#E74C3C',
        label='Gini Importance',
    )
    ax.barh(
        y_pos,
        top_df['Permutation_norm'].values,
        height=bar_height,
        color='#3498DB',
        label='Permutation Importance',
    )
    ax.barh(
        y_pos + bar_height,
        top_df['SHAP_norm'].values,
        height=bar_height,
        color='#1ABC9C',
        label='Mean |SHAP|',
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=PLOT_SMALL_SIZE)
    ax.set_xlabel('Normalised Importance', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Comparison of Feature Importance Metrics', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.set_xlim(0, 1.15)
    ax.legend(loc='lower right', fontsize=PLOT_LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, out_path)
    plt.close()
    print(f'  Figure 10 saved: {out_path}')


def plot_dependence_top_features(explanation, X_test, final_features, out_path, n_top=4):
    """Figure 11: 2x2 dependence plots for the top 4 SHAP features."""
    top_features = list(pd.Series(
        np.abs(explanation.values).mean(axis=0),
        index=final_features,
    ).sort_values(ascending=False).head(n_top).index)

    fig, axes = plt.subplots(2, 2, figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    axes = axes.ravel()

    for idx, (ax, feature) in enumerate(zip(axes, top_features)):
        plt.sca(ax)
        shap.plots.scatter(explanation[:, feature], ax=ax, show=False)
        ax.set_title(f'Dependence: {feature}', fontsize=PLOT_PANEL_TITLE_SIZE, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('SHAP Dependence Plots for Top Features',
                 fontsize=PLOT_TITLE_SIZE, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, out_path)
    plt.close()
    print(f'  Figure 11 saved: {out_path}')


def select_example_indices(y_test, y_proba, y_pred):
    """Select representative samples for local explanation."""
    y_test = np.asarray(y_test)
    y_proba = np.asarray(y_proba)
    y_pred = np.asarray(y_pred)

    indices = {}

    # High-confidence true positive.
    tp_mask = (y_test == 1) & (y_pred == 1)
    if tp_mask.any():
        indices['True Positive (high confidence)'] = int(np.argmax(y_proba * tp_mask))

    # High-confidence true negative.
    tn_mask = (y_test == 0) & (y_pred == 0)
    if tn_mask.any():
        indices['True Negative (high confidence)'] = int(np.argmin(y_proba + (1 - tn_mask)))

    # False negative (if any).
    fn_mask = (y_test == 1) & (y_pred == 0)
    if fn_mask.any():
        indices['False Negative'] = int(np.argmax(fn_mask))

    # False positive (if any).
    fp_mask = (y_test == 0) & (y_pred == 1)
    if fp_mask.any():
        indices['False Positive'] = int(np.argmax(fp_mask))

    # Near-threshold case.
    near_idx = int(np.argmin(np.abs(y_proba - 0.5)))
    indices['Near-threshold'] = near_idx

    return indices


def _plot_waterfall_examples_legacy(explanation, indices, out_path):
    """Figure 12: Waterfall plots for selected individual samples.

    Each waterfall is rendered in its own figure and then stacked vertically
    using PIL to guarantee that text/annotations from different subplots do
    not overlap.
    """
    import io
    from matplotlib import font_manager
    from PIL import Image, ImageDraw, ImageFont

    gap = 40          # vertical gap between waterfalls (pixels at 300 dpi)
    top_margin = 80   # space for the main title
    images = []
    subtitles = []

    for label, idx in indices.items():
        prob = 1 / (1 + np.exp(-explanation[idx].base_values - explanation[idx].values.sum()))
        subtitle = f'{label} (sample {idx}, predicted prob ≈ {prob:.3f})'
        subtitle = f'{label} (sample {idx}, predicted prob ~= {prob:.3f})'
        subtitles.append(subtitle)

        fig, ax = plt.subplots(figsize=(11, 3.5), dpi=PLOT_DPI, facecolor='white')
        plt.sca(ax)
        shap.plots.waterfall(explanation[idx], max_display=8, show=False)
        ax.set_title(subtitle, fontsize=PLOT_PANEL_TITLE_SIZE, fontweight='bold', pad=10)
        ax.set_facecolor('white')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=PLOT_DPI, bbox_inches='tight',
                    pad_inches=0.05, facecolor='white')
        plt.close(fig)
        buf.seek(0)
        images.append(Image.open(buf).convert('RGB'))

    # Compute combined canvas size.
    max_width = max(img.width for img in images)
    total_height = sum(img.height for img in images) + gap * (len(images) - 1) + top_margin

    combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined)

    # Main title.
    try:
        font_path = font_manager.findfont(PLOT_FONT_FAMILY, fallback_to_default=True)
        title_font_size_px = int(PLOT_TITLE_SIZE * PLOT_DPI / 72)
        font = ImageFont.truetype(font_path, title_font_size_px)
    except Exception:
        font = ImageFont.load_default()
    main_title = "SHAP Waterfall Plots for Representative Samples"
    bbox = draw.textbbox((0, 0), main_title, font=font)
    title_width = bbox[2] - bbox[0]
    draw.text(((max_width - title_width) // 2, 20), main_title, fill=(0, 0, 0), font=font)

    # Stack individual waterfalls.
    y_offset = top_margin
    for img in images:
        x_offset = (max_width - img.width) // 2
        combined.paste(img, (x_offset, y_offset))
        y_offset += img.height + gap

    target_size = (
        int(PLOT_FIG_SIZE[0] * PLOT_DPI),
        int(PLOT_FIG_SIZE[1] * PLOT_DPI),
    )
    final_canvas = Image.new('RGB', target_size, (255, 255, 255))
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        resample = Image.LANCZOS
    combined.thumbnail(target_size, resample)
    final_canvas.paste(
        combined,
        ((target_size[0] - combined.width) // 2, (target_size[1] - combined.height) // 2),
    )
    final_canvas.save(out_path, dpi=(PLOT_DPI, PLOT_DPI))
    print(f'  Figure 12 saved: {out_path}')


def plot_waterfall_examples(explanation, indices, out_path):
    """Figure 12: Bioinformatics-ready local SHAP waterfall multipanel figure.

    The final figure is formatted as double-column artwork: 178 mm wide,
    white background, readable panel labels, and a TIFF copy for submission.
    """
    import string
    from matplotlib.patches import Patch
    from PIL import Image

    bio_width_in = 178 / 25.4
    fig_height_in = max(6.8, 1.65 * len(indices) + 0.45)
    pos_color = '#E64B35'
    neg_color = '#3182BD'
    max_display = 6

    def format_feature_value(value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 'NA'
        value = float(value)
        abs_value = abs(value)
        if abs_value >= 100:
            return f'{value:.0f}'
        if abs_value >= 10:
            return f'{value:.1f}'
        if abs_value >= 1:
            return f'{value:.2f}'
        return f'{value:.3f}'

    fig, axes = plt.subplots(
        len(indices), 1,
        figsize=(bio_width_in, fig_height_in),
        dpi=PLOT_DPI,
        facecolor='white',
    )
    if len(indices) == 1:
        axes = [axes]

    feature_names = np.asarray(explanation.feature_names)
    compact_labels = {
        'True Positive (high confidence)': 'TP (high confidence)',
        'True Negative (high confidence)': 'TN (high confidence)',
        'False Negative': 'FN',
        'False Positive': 'FP',
        'Near-threshold': 'Near threshold',
    }

    for panel_idx, (ax, (label, idx)) in enumerate(zip(axes, indices.items())):
        sample_exp = explanation[idx]
        values = np.asarray(sample_exp.values, dtype=float)
        data = np.asarray(sample_exp.data, dtype=float)
        base_value = float(np.asarray(sample_exp.base_values).reshape(-1)[0])
        final_value = float(base_value + values.sum())

        ranked_idx = np.argsort(np.abs(values))[::-1]
        top_idx = ranked_idx[:max_display - 1]
        other_idx = ranked_idx[max_display - 1:]
        plot_values = list(values[top_idx])
        plot_labels = [
            f'{format_feature_value(data[i])} = {feature_names[i]}'
            for i in top_idx
        ]
        if len(other_idx) > 0:
            plot_values.append(float(values[other_idx].sum()))
            plot_labels.append(f'{len(other_idx)} other features')

        running = [base_value]
        current = base_value
        y_pos = np.arange(len(plot_values))
        for y, shap_value in zip(y_pos, plot_values):
            next_value = current + shap_value
            left = min(current, next_value)
            width = abs(shap_value)
            color = pos_color if shap_value >= 0 else neg_color
            ax.barh(
                y, width, left=left, height=0.66,
                color=color, edgecolor='white', linewidth=0.45,
            )
            label_x = next_value + (0.01 if shap_value >= 0 else -0.01)
            ax.text(
                label_x, y, f'{shap_value:+.2f}',
                ha='left' if shap_value >= 0 else 'right',
                va='center',
                fontsize=7,
                color=color,
            )
            current = next_value
            running.append(current)

        x_min = min(running)
        x_max = max(running)
        x_pad = max((x_max - x_min) * 0.18, 0.04)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.axvline(base_value, color='#7F7F7F', linestyle='--', linewidth=0.7)
        ax.axvline(final_value, color='#222222', linestyle='-', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_labels, fontsize=7.2)
        ax.invert_yaxis()
        ax.tick_params(axis='x', labelsize=7.5, width=0.7, length=3)
        ax.tick_params(axis='y', width=0.7, length=2)
        ax.grid(axis='x', color='#E6E6E6', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_facecolor('white')

        panel_label = string.ascii_uppercase[panel_idx]
        probability = np.clip(final_value, 0, 1)
        compact_label = compact_labels.get(label, label)
        ax.set_title(
            f'{panel_label} {compact_label}, sample {idx}, $p$ = {probability:.3f}',
            fontsize=8.2,
            fontweight='bold',
            loc='left',
            pad=3,
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.7)

    axes[-1].set_xlabel('Model output probability', fontsize=8.5)
    fig.legend(
        handles=[
            Patch(facecolor=pos_color, edgecolor='white', label='Increases prediction'),
            Patch(facecolor=neg_color, edgecolor='white', label='Decreases prediction'),
        ],
        loc='upper right',
        bbox_to_anchor=(0.985, 0.995),
        ncol=2,
        frameon=False,
        fontsize=7.4,
        handlelength=1.0,
        handleheight=0.8,
        columnspacing=1.0,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.36, right=0.985, top=0.965, bottom=0.065, hspace=0.82)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor='white')
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    fig.savefig(pdf_path, facecolor='white')
    tif_path = os.path.splitext(out_path)[0] + '.tif'
    plt.close(fig)
    with Image.open(out_path) as image:
        image.save(tif_path, dpi=(PLOT_DPI, PLOT_DPI))
    print(f'  Figure 12 saved: {out_path}')
    print(f'  Figure 12 PDF saved: {pdf_path}')
    print(f'  Figure 12 TIFF saved: {tif_path}')


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def run_shap_experiment(fig_dir=SHAP_FIG_DIR, save_numerical=True):
    """Run the full SHAP interpretability experiment."""
    os.makedirs(fig_dir, exist_ok=True)
    print('=' * 70)
    print('FuseVirome - SHAP Interpretability Experiment')
    print('=' * 70)

    print('\n[1/5] Loading data and reproducing final 29-feature split...')
    X_train, y_train, X_test, y_test, final_features = get_final_datasets()
    print(f'  Train: {X_train.shape}, Test: {X_test.shape}')

    print('\n[2/5] Training final Random Forest model...')
    rf = train_final_rf(X_train, y_train)
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f'  Test AUC: {test_auc:.4f}, TP={tp}, TN={tn}, FP={fp}, FN={fn}')

    print('\n[3/5] Computing SHAP values with TreeExplainer...')
    explainer, shap_values_positive, explanation, expected_value = compute_shap_values(
        rf, X_train, X_test, final_features
    )

    print('\n[4/5] Computing importance metrics...')
    imp_df = compute_importance_metrics(rf, X_test, y_test, shap_values_positive, final_features)
    print('\nTop 5 features by SHAP importance:')
    print(imp_df[['Feature', 'SHAP']].head(5).to_string(index=False))

    if save_numerical:
        imp_path = os.path.join(fig_dir, 'shap_importance.csv')
        imp_df.to_csv(imp_path, index=False)
        print(f'  Saved: {imp_path}')

        shap_path = os.path.join(fig_dir, 'shap_values.npy')
        np.save(shap_path, shap_values_positive)
        print(f'  Saved: {shap_path}')

    print('\n[5/5] Generating SHAP figures...')
    plot_global_summary(explanation, os.path.join(fig_dir, 'Figure8_SHAP_Global_Summary.png'))
    plot_beeswarm(explanation, os.path.join(fig_dir, 'Figure9_SHAP_Beeswarm.png'))
    plot_importance_comparison(imp_df, os.path.join(fig_dir, 'Figure10_SHAP_Importance_Comparison.png'))
    plot_dependence_top_features(
        explanation, X_test, final_features,
        os.path.join(fig_dir, 'Figure11_SHAP_Dependence_TopFeatures.png')
    )

    example_indices = select_example_indices(y_test, y_proba, y_pred)
    print(f'  Selected example indices: {example_indices}')
    plot_waterfall_examples(
        explanation, example_indices,
        os.path.join(fig_dir, 'Figure12_SHAP_Local_Waterfall.png')
    )

    print('\n' + '=' * 70)
    print('SHAP interpretability experiment complete.')
    print(f'Figures saved to: {fig_dir}')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        default=SHAP_FIG_DIR,
        help='Directory for generated SHAP artifacts (default: outputs/figures).',
    )
    run_shap_experiment(fig_dir=parser.parse_args().output_dir)
