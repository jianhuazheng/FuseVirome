#!/usr/bin/env python3
"""FuseVirome - Complete Reproducible Pipeline
Machine learning framework with dynamic reference construction for scoring
mNGS-detected viral signals in veterinary metatranscriptomic sequencing.

Run: python src/fusevirome.py
Output: Metrics in the terminal and generated figures in outputs/figures/.
"""

import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, recall_score,
    confusion_matrix, brier_score_loss, roc_curve)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.inspection import permutation_importance

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_Training_Data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_External_Validation_Data.csv')
RANDOM_SEED = 0  
np.random.seed(RANDOM_SEED)

FIG_DIR = os.path.join(BASE_DIR, 'outputs', 'figures', '')

PLOT_FIG_SIZE = (9.0, 6.75)
PLOT_DPI = 600
PLOT_FONT_FAMILY = 'DejaVu Sans'
PLOT_BASE_SIZE = 12
PLOT_TITLE_SIZE = 14
PLOT_PANEL_TITLE_SIZE = 12
PLOT_LABEL_SIZE = 12
PLOT_TICK_SIZE = 11
PLOT_LEGEND_SIZE = 10
PLOT_SMALL_SIZE = 10


def configure_plot_style():
    """Apply one font and sizing standard for all manuscript figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': [PLOT_FONT_FAMILY],
        'font.size': PLOT_BASE_SIZE,
        'axes.titlesize': PLOT_TITLE_SIZE,
        'axes.labelsize': PLOT_LABEL_SIZE,
        'xtick.labelsize': PLOT_TICK_SIZE,
        'ytick.labelsize': PLOT_TICK_SIZE,
        'legend.fontsize': PLOT_LEGEND_SIZE,
        'figure.titlesize': PLOT_TITLE_SIZE,
        'figure.dpi': PLOT_DPI,
        'savefig.dpi': PLOT_DPI,
    })


def save_plot(fig, out_path):
    """Save a fixed-size PNG so all figures share width, height, and DPI."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.set_size_inches(*PLOT_FIG_SIZE, forward=True)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor='white')


configure_plot_style()

# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data():
    """Load aggregated training and test data with leakage prevention."""
    train = pd.read_csv(TRAIN_PATH, encoding='utf-8')
    test = pd.read_csv(TEST_PATH, encoding='utf-8')
    
    leakage_cols = ['Y_label_max', 'qPCR']
    for col in leakage_cols:
        if col in train.columns:
            train = train.drop(columns=[col])
        if col in test.columns:
            test = test.drop(columns=[col])
    
    y_train = train['Y'].values
    if 'Final_Label' in test.columns:
        y_test = test['Final_Label'].values
    elif 'Y' in test.columns:
        y_test = test['Y'].values
    else:
        raise KeyError("Test set must have 'Y' or 'Final_Label' column")
    
    drop_cols_train = ['Y', 'Y_label_max', 'Virus_Name', 'Sample_ID', 'Anno_Rate']
    drop_cols_test = ['Final_Label', 'qPCR', 'Virus_Name', 'Sample_ID', 'Anno_Rate']
    X_train = train.drop(columns=[c for c in drop_cols_train if c in train.columns], errors='ignore')
    X_test = test.drop(columns=[c for c in drop_cols_test if c in test.columns], errors='ignore')
    
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    return X_train, y_train, X_test, y_test

# =============================================================================
# 2. FEATURE REFINEMENT
# =============================================================================

def stage1_null_removal(X_train):
    """Remove zero-variance and null features."""
    removed = []
    for col in X_train.columns:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            removed.append(col)
            continue
        if X_train[col].nunique() <= 1 or X_train[col].std() == 0:
            removed.append(col)
    return removed


def stage2_fused_exclusion(X_train):
    """Remove fused-track features redundant with read/contig tracks (r > 0.95)."""
    import scipy.stats as stats
    fused_features = [c for c in X_train.columns if '_fused' in c]
    removed = []
    for fc in fused_features:
        base_name = fc.replace('_fused', '')
        for counterpart in [base_name + '_reads', base_name + '_contigs']:
            if counterpart in X_train.columns and fc not in removed:
                r, _ = stats.pearsonr(X_train[fc].fillna(0), X_train[counterpart].fillna(0))
                if not np.isnan(r) and abs(r) > 0.95:
                    removed.append(fc)
                    break
    return removed


# =============================================================================
# 3. SIX CLASSIFIER COMPARISON
# =============================================================================

def build_classifiers():
    """Return one shared, reproducible classifier configuration.

    Logistic Regression is standardized to improve numerical conditioning and
    ensure convergence. Both metric calculation and ROC plotting must use this
    factory so that they cannot silently drift to different hyperparameters.
    """
    return {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3,
            random_state=RANDOM_SEED),
        'LR': make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight='balanced', max_iter=5000,
                random_state=RANDOM_SEED)
        ),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'DT': DecisionTreeClassifier(max_depth=3, class_weight='balanced',
            random_state=RANDOM_SEED),
    }

def compare_classifiers(X_train, y_train, X_test, y_test):
    """Train and evaluate 6 classifiers (RF, ExtraTrees, GB, LR, kNN, DT)."""
    classifiers = build_classifiers()
    results = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results.append({
            'Classifier': name, 'AUC': test_auc, 'F1': f1, 'Acc': acc,
            'Se': tp/(tp+fn), 'Sp': tn/(tn+fp),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Overfit': train_auc - test_auc
        })
    return pd.DataFrame(results)

# =============================================================================
# 4. ELEVEN-CONFIGURATION ABLATION STUDY
# =============================================================================

def ablation_study(X_train, y_train, X_test, y_test, final_features):
    """11-configuration track-level ablation study."""
    de_novo = ['Count_Total_Contigs', 'Count_Correct_Contigs', 'Count_Noise_Contigs']
    read_cov = ['Reads_Count_reads', 'Coverage_bp_reads', 'Coverage_Percent_reads',
        'Avg_Coverage_Depth_x_reads', 'Mean_BaseQ_reads', 'Mean_MapQ_reads',
        'Histo_Max_Bin_Percent_reads']
    contig_cov = ['Reads_Count_contigs', 'Coverage_bp_contigs', 'Coverage_Percent_contigs',
        'Avg_Coverage_Depth_x_contigs', 'Mean_MapQ_contigs',
        'Histo_Max_Bin_Percent_contigs', 'Mean_BaseQ_contigs']
    blast = ['identity_max', 'identity_mean', 'BitScore_max', 'BitScore_mean',
        'BitScore_sum', 'AlignLength_max', 'AlignLength_sum', 'qlen_mean',
        'Mismatches_mean', 'Evalue_min', 'Align_Qlen_Ratio_Max']
    fused = [c for c in final_features if '_fused' in c]
    configs = {
        'DeNovo': de_novo,
        'ReadCov': read_cov,
        'ContigCov': contig_cov,
        'BLAST': blast,
        'Read+Contig': read_cov + contig_cov,
        'Read+Contig+Fused': read_cov + contig_cov + fused,
        'DeNovo+Read+Contig': de_novo + read_cov + contig_cov,
        'DeNovo+Read+Contig+BLAST': de_novo + read_cov + contig_cov + blast,
        'Full_29f': final_features,
    }
    abl_results = []
    for name, feats in configs.items():
        feats = [f for f in feats if f in X_train.columns]
        if len(feats) == 0:
            continue
        X_tr = X_train[feats].fillna(X_train[feats].median())
        X_te = X_test[feats].fillna(X_test[feats].median())
        rf = RandomForestClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1)
        rf.fit(X_tr, y_train)
        y_proba = rf.predict_proba(X_te)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        abl_results.append({
            'Config': name, 'N': len(feats), 'AUC': auc, 'F1': f1, 'Acc': acc,
            'Se': tp/(tp+fn), 'Sp': tn/(tn+fp), 'FN': fn
        })
    return pd.DataFrame(abl_results)

# =============================================================================
# 5. FIGURE GENERATION
# =============================================================================

def generate_figures(rf_model, X_train, y_train, X_test, y_test, final_features,
                     clf_results, abl_results, fig_dir=FIG_DIR):
    """Generate manuscript figures (Original 5 + 2 New ones)."""
    fig_dir = os.path.join(os.path.abspath(fig_dir), '')
    os.makedirs(fig_dir, exist_ok=True)
    configure_plot_style()
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Common Color Map
    colors_map = {
        'RF': '#E74C3C', 'ExtraTrees': '#3498DB', 'GB': '#1ABC9C',
        'LR': '#2C3E50', 'kNN': '#E67E22', 'DT': '#16A085'
    }
    classifiers = build_classifiers()

    # ------------------- Original Figure 2 -------------------
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    roc_curves = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pp = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pp)
        auc_val = roc_auc_score(y_test, pp)
        roc_curves[name] = (fpr, tpr, auc_val)
        ax.plot(fpr, tpr, color=colors_map[name], linewidth=1.8,
                label=f'{name} (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Reference')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('ROC Curves of Six Classifiers', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=PLOT_LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure2_ROC.png')
    plt.close()
    print(f"  Figure 2: ROC curves saved")

    # ------------------- Original Figure 3 -------------------
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    fi = pd.Series(rf_model.feature_importances_, index=final_features)
    fi = fi.sort_values(ascending=False).head(10)
    fi_sorted = fi.sort_values(ascending=True)
    colors = ['#C0392B', '#E74C3C', '#2980B9', '#3498DB', '#1ABC9C',
              '#16A085', '#8E44AD', '#9B59B6', '#F39C12', '#E67E22']
    y_pos = range(len(fi_sorted))
    ax.barh(y_pos, fi_sorted.values, color=colors, height=0.7,
            edgecolor='white', linewidth=0.5)
    for i, (fname, gini_val) in enumerate(fi_sorted.items()):
        ax.text(gini_val + 0.002, i, f'{gini_val:.4f}',
                va='center', fontsize=PLOT_SMALL_SIZE, color='#333333')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(fi_sorted.index, fontsize=PLOT_SMALL_SIZE)
    ax.set_xlabel('Gini Importance', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Top 10 Feature Importance (29-Feature RF Model)',
                 fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.set_xlim(0, fi_sorted.max() * 1.18)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure3_Features.png')
    plt.close()
    print(f"  Figure 3: Top 10 feature importance saved")

    # ------------------- Feature importance summed by category -------------------
    def _cat_of(fname):
        if fname.startswith('Count_'):
            return 'Contig'
        if fname.endswith('_reads'):
            return 'Read'
        if fname.endswith('_contigs'):
            return 'Contig'
        if fname.endswith('_fused'):
            return 'Fused alignment'
        return 'Homology / BLAST'
    fi_all = pd.Series(rf_model.feature_importances_, index=final_features)
    fi_cat = fi_all.index.map(_cat_of)
    cat_order = ['Read', 'Contig', 'Fused alignment', 'Homology / BLAST']
    cat_sums = fi_all.groupby(fi_cat).sum().reindex(cat_order)
    cat_counts = fi_all.groupby(fi_cat).count().reindex(cat_order)
    cat_colors = {'Read': '#3498DB', 'Contig': '#1ABC9C',
                  'Fused alignment': '#E67E22', 'Homology / BLAST': '#8E44AD'}

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    cat_total = cat_sums.sum()
    cat_pct = cat_sums / cat_total * 100
    bars = ax.bar(cat_order, cat_pct.values,
                  color=[cat_colors[c] for c in cat_order],
                  width=0.6, edgecolor='white', linewidth=0.5)
    for bar, val, cnt in zip(bars, cat_pct.values, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + cat_pct.max() * 0.02,
                f'{val:.1f}%\n(n = {cnt})',
                ha='center', va='bottom', fontsize=PLOT_LABEL_SIZE, color='#333333')
    ax.set_ylabel('Share of total Gini importance (%)', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Feature Importance by Evidence Category',
                 fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.set_ylim(0, cat_pct.max() * 1.22)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_yticklabels([f'{t:.0f}' for t in np.arange(0, 61, 10)])
    ax.tick_params(axis='x', labelsize=PLOT_LABEL_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure3B_Feature_Category_Summary.png')
    plt.close()
    print(f"  Figure 3B: feature category summary saved")

    # ------------------- Original Figure 4 split into 4A and 4B -------------------
    np.random.seed(RANDOM_SEED)
    auc_boots = []
    for _ in range(10000):
        idx = np.random.choice(len(y_test), len(y_test), replace=True)
        if len(np.unique(y_test[idx])) > 1:
            auc_boots.append(roc_auc_score(y_test[idx], y_proba[idx]))
    auc_ci = np.percentile(auc_boots, [2.5, 97.5])
    auc_point = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    ax.hist(auc_boots, bins=80, color='#3498DB', edgecolor='white',
            alpha=0.7, density=True)
    ax.axvline(auc_point, color='#E74C3C', linewidth=2,
               label=f'AUC = {auc_point:.4f}')
    ax.axvline(auc_ci[0], color='#2C3E50', linestyle='--', linewidth=1,
               label=f'95% CI: [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]')
    ax.axvline(auc_ci[1], color='#2C3E50', linestyle='--', linewidth=1)
    ax.set_xlabel('AUC', fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel('Density', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Bootstrap AUC Distribution\n(10,000 iterations)',
                 fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.text(-0.12, 1.08, 'A', transform=ax.transAxes,
            fontsize=PLOT_PANEL_TITLE_SIZE + 4, fontweight='bold',
            va='top', ha='left', clip_on=False)
    ax.legend(fontsize=PLOT_LEGEND_SIZE, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure4A_Bootstrap_AUC.png')
    plt.close()
    print(f"  Figure 4A: Bootstrap AUC distribution saved")

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_proba, n_bins=10, strategy='uniform')
    brier = brier_score_loss(y_test, y_proba)

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Perfectly Calibrated')
    ax.plot(mean_predicted_value, fraction_of_positives, 'o-',
            color='#2E86AB', linewidth=2, markersize=5,
            label=f'RF (Brier={brier:.4f})')
    ax.set_xlabel('Mean Predicted Probability', fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel('Fraction of Positives', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Calibration Curve', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.text(-0.12, 1.08, 'B', transform=ax.transAxes,
            fontsize=PLOT_PANEL_TITLE_SIZE + 4, fontweight='bold',
            va='top', ha='left', clip_on=False)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.legend(fontsize=PLOT_LEGEND_SIZE, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure4B_Calibration_Curve.png')
    plt.close()
    print(f"  Figure 4B: Calibration curve saved")

    # ------------------- Original Figure 5 -------------------
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    thresholds = np.arange(0.01, 0.99, 0.01)
    prevalence = np.mean(y_test)
    net_benefit_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
    net_benefit_all = np.clip(net_benefit_all, -1, 1)
    net_benefit_none = np.zeros_like(thresholds)
    net_benefit_rf = []
    for thresh in thresholds:
        pred = (y_proba >= thresh).astype(int)
        tp = np.sum((pred == 1) & (y_test == 1))
        fp = np.sum((pred == 1) & (y_test == 0))
        n = len(y_test)
        nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefit_rf.append(nb)
    net_benefit_rf = np.array(net_benefit_rf)
    ax.plot(thresholds, net_benefit_all, 'k--', linewidth=1.5, alpha=0.6, label='Treat All')
    ax.plot(thresholds, net_benefit_none, 'k:', linewidth=1.5, alpha=0.6, label='Treat None')
    ax.plot(thresholds, net_benefit_rf, color='#2E86AB', linewidth=2.5, label='Random Forest')
    ax.fill_between(thresholds, net_benefit_none, net_benefit_rf,
                    where=(net_benefit_rf > net_benefit_none),
                    alpha=0.15, color='#2E86AB')
    ax.set_xlabel('Threshold Probability', fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel('Net Benefit', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('Decision Curve Analysis', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([-0.1, max(np.max(net_benefit_rf[net_benefit_rf < 10]), prevalence) * 1.1])
    ax.legend(loc='upper right', fontsize=PLOT_LEGEND_SIZE)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure5_DCA.png')
    plt.close()
    print(f"  Figure 5: DCA saved")

    # =========================================================================
    # NEW: [Figure 6] F1 Score Bar Chart for Six Classifiers
    # =========================================================================
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    
    # Map colors to bars based on classifier names
    bar_colors = [colors_map.get(c, '#95A5A6') for c in clf_results['Classifier']]
    
    bars = ax.bar(clf_results['Classifier'], clf_results['F1'], 
                  color=bar_colors, edgecolor='white', alpha=0.85, width=0.6)
    
    # Annotate bars with values
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.3f}', 
                ha='center', va='bottom', fontsize=PLOT_TICK_SIZE, fontweight='bold')
        
    ax.set_ylim(0, max(clf_results['F1']) * 1.15)
    ax.set_ylabel('F1 (Test Set)', fontsize=PLOT_LABEL_SIZE)
    ax.set_title('F1 Score Comparison Across Classifiers', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    save_plot(fig, f'{fig_dir}Figure6_F1_BarChart.png')
    plt.close()
    print(f"  Figure 6: F1 Bar chart saved")

    # =========================================================================
    # NEW: [Figure 3] Bootstrap Metrics as Three Separate Boxplots
    #      (one figure per metric: Recall / F1 / Accuracy)
    # =========================================================================
    # Data containers for plotting
    boot_recall, boot_f1, boot_accuracy = [], [], []
    clf_names = []
    n_bootstrap = 1000
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"  Running test-set bootstrap for Boxplots (this might take a moment)...")
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        recalls, f1s, accs = [], [], []
        for _ in range(n_bootstrap):
            idx = rng.choice(len(y_test), len(y_test), replace=True)
            y_test_b = y_test[idx]
            y_pred_b = y_pred[idx]
            recalls.append(recall_score(y_test_b, y_pred_b))
            f1s.append(f1_score(y_test_b, y_pred_b))
            accs.append(accuracy_score(y_test_b, y_pred_b))

        boot_recall.append(recalls)
        boot_f1.append(f1s)
        boot_accuracy.append(accs)
        clf_names.append(name)

    # Metric colors matching the requested style (red/green/blue) - unchanged
    metric_colors = {
        'Recall': '#B6D7A8',     # light green
        'F1': '#9FC5E8',         # light blue
        'Accuracy': '#F4A3A3'    # light red
    }
    edge_colors = {
        'Recall': '#38761D',
        'F1': '#3D85C6',
        'Accuracy': '#CC0000'
    }

    def draw_metric_boxplot(ax, metric, data, title_size, tick_size):
        """Draw one metric's classifier boxplot on an axes (shared by all exports)."""
        bplot = ax.boxplot(
            data,
            patch_artist=True,
            boxprops=dict(linewidth=0.9, color=edge_colors[metric]),
            capprops=dict(color=edge_colors[metric], linewidth=0.9),
            whiskerprops=dict(color=edge_colors[metric], linewidth=0.9),
            flierprops=dict(marker='o', color=edge_colors[metric], alpha=0.5, markersize=3),
            medianprops=dict(color=edge_colors[metric], linewidth=1.2)
        )
        color = metric_colors[metric]
        for patch in bplot['boxes']:
            patch.set_facecolor(color)
            patch.set_edgecolor(edge_colors[metric])
            patch.set_alpha(0.9)

        ax.set_title(metric, fontsize=title_size, fontweight='bold')
        # Short horizontal names fit the narrow panels without overlap;
        # match the abbreviations used in the ROC legend (RF, ET, ...)
        short_names = {'RF': 'RF', 'ExtraTrees': 'ET', 'GB': 'GB',
                       'LR': 'LR', 'kNN': 'kNN', 'DT': 'DT'}
        display_names = [short_names.get(n, n) for n in clf_names]
        ax.set_xticks(range(1, len(clf_names) + 1))
        ax.set_xticklabels(display_names, fontsize=tick_size)
        ax.set_ylabel('Score (Test-Set Bootstrap)', fontsize=tick_size)
        ax.tick_params(axis='y', labelsize=tick_size)
        ax.set_ylim(0.5, 1.02)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='-', alpha=0.35)

    # Bootstrap data used for the three standalone panel exports below.
    metric_data = {
        'Recall': boot_recall,
        'F1': boot_f1,
        'Accuracy': boot_accuracy,
    }

    # =========================================================================
    # PANEL EXPORTS: Recall / F1 / Accuracy boxplots
    # =========================================================================
    fig = plt.figure(figsize=(PLOT_FIG_SIZE[0], PLOT_FIG_SIZE[1] * 1.55), dpi=PLOT_DPI)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1], hspace=0.32, wspace=0.35)

    # (a) ROC curves of six classifiers across the whole top row
    ax_a = fig.add_subplot(gs[0, :])
    for name, (fpr, tpr, auc_val) in roc_curves.items():
        ax_a.plot(fpr, tpr, color=colors_map[name], linewidth=1.8,
                  label=f'{name} (AUC = {auc_val:.4f})')
    ax_a.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4, label='Reference')
    ax_a.set_xlabel('1 - Specificity (FPR)', fontsize=PLOT_LABEL_SIZE)
    ax_a.set_ylabel('Sensitivity (TPR)', fontsize=PLOT_LABEL_SIZE)
    ax_a.set_title('ROC Curves of Six Classifiers', fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax_a.legend(loc='lower right', fontsize=PLOT_LEGEND_SIZE)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3)
    ax_a.text(-0.04, 1.03, '(a)', transform=ax_a.transAxes,
              fontsize=PLOT_TITLE_SIZE, fontweight='bold')

    # (b-d) one boxplot panel per metric
    panel_letters = ['(b)', '(c)', '(d)']
    panel_exports = []   # (metric, data, axes, letter_artist) for standalone exports
    for i, (metric, data) in enumerate(metric_data.items()):
        ax = fig.add_subplot(gs[1, i])
        draw_metric_boxplot(ax, metric, data, PLOT_LABEL_SIZE, 7)
        letter_artist = ax.text(-0.02, 1.07, panel_letters[i], transform=ax.transAxes,
                                fontsize=PLOT_TITLE_SIZE, fontweight='bold', va='bottom')
        panel_exports.append((metric, data, ax, letter_artist))

    # Export panels (b)-(d) by cropping their exact pixels from the rendered
    # shared layout buffer, so their sizes and styles remain consistent.
    # The (b)-(d) letters are hidden for the export so the standalone panels
    # carry no panel letters, while the composite keeps them.
    panel_files = {
        'Recall':   'Figure2_3_PanelB_Recall.png',
        'F1':       'Figure2_3_PanelC_F1.png',
        'Accuracy': 'Figure2_3_PanelD_Accuracy.png',
    }
    for (_, _, _, letter_artist) in panel_exports:
        letter_artist.set_visible(False)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    buf_h = buf.shape[0]
    renderer = fig.canvas.get_renderer()
    for (metric, _, ax, _), filename in zip(panel_exports, panel_files.values()):
        # tightbbox includes title, tick labels and axis labels, not just the axes
        bbox_px = ax.get_tightbbox(renderer)
        x0, y0 = int(bbox_px.x0), int(bbox_px.y0)
        x1, y1 = int(np.ceil(bbox_px.x1)), int(np.ceil(bbox_px.y1))
        crop = buf[buf_h - y1: buf_h - y0, x0: x1]   # buffer origin is bottom-left
        Image.fromarray(crop).convert('RGB').save(f'{fig_dir}{filename}')
        print(f"  Panel export: {filename} saved ({crop.shape[1]/fig.dpi:.2f} x "
              f"{crop.shape[0]/fig.dpi:.2f} in)")

    plt.close(fig)

    n_cfg = len(abl_results)

    # =========================================================================
    # SUPPLEMENTARY: Table S4 redrawn - Features | AUC | Specificity | FN heatmap
    # =========================================================================
    metric_names = ['AUC', 'Sp', 'FN']
    metric_labels = ['AUC', 'Specificity', 'FN']
    heat_data = abl_results[metric_names].values.astype(float)

    # Normalize each column independently so AUC/Sp (0.5-1.0) and FN (count)
    # are each comparable down their own column; raw values are annotated.
    # FN is a failure count, so its scale is inverted (fewer FN = greener).
    col_min = heat_data.min(axis=0)
    col_max = heat_data.max(axis=0)
    norm_data = (heat_data - col_min) / np.where(col_max > col_min, col_max - col_min, 1)
    norm_data[:, metric_names.index('FN')] = 1 - norm_data[:, metric_names.index('FN')]

    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    im = ax.imshow(norm_data, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_labels, fontsize=PLOT_LABEL_SIZE)
    ax.set_yticks(range(n_cfg))
    ax.set_yticklabels(abl_results['Config'], fontsize=PLOT_SMALL_SIZE)
    ax.set_title('Ablation Study: AUC, Specificity and False Negatives\nacross Nine Feature Combinations',
                 fontsize=PLOT_TITLE_SIZE, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=PLOT_LABEL_SIZE)

    # Annotate each cell with its raw value (FN shown as integer count)
    for i in range(n_cfg):
        for j in range(len(metric_names)):
            val = heat_data[i, j]
            label = f'{val:.3f}' if metric_names[j] != 'FN' else f'{int(val)}'
            shade = norm_data[i, j]
            text_color = 'white' if (shade < 0.25 or shade > 0.8) else '#333333'
            ax.text(j, i, label, ha='center', va='center',
                    fontsize=PLOT_LABEL_SIZE, color=text_color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Low', 'High'], fontsize=PLOT_LEGEND_SIZE)
    cbar.set_label('Relative (normalized per column)', fontsize=PLOT_LABEL_SIZE)

    plt.tight_layout()
    save_plot(fig, f'{fig_dir}FigureS4_Ablation_Heatmap.png')
    plt.close()
    print(f"  Supplementary: Table S4 ablation heatmap saved")


# =============================================================================
# 6. ADDITIONAL VALIDATION
# =============================================================================

def compute_permutation_importance(rf_model, X_test, y_test, final_features):
    """Permutation importance (30 repeats)."""
    pi = permutation_importance(rf_model, X_test, y_test, n_repeats=30,
                                 random_state=RANDOM_SEED, scoring='roc_auc')
    pi_df = pd.DataFrame({
        'Feature': final_features,
        'PI_mean': pi.importances_mean,
        'PI_std': pi.importances_std
    }).sort_values('PI_mean', ascending=False)
    return pi_df


def compute_feature_stability(X_train, y_train, final_features):
    """Feature importance stability across 10 random seeds."""
    all_imp = {f: [] for f in final_features}
    for seed in range(10):
        rf = RandomForestClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=seed, n_jobs=1)
        rf.fit(X_train, y_train)
        for i, f in enumerate(final_features):
            all_imp[f].append(rf.feature_importances_[i])
    stability = pd.DataFrame({
        'Feature': final_features,
        'Mean': [np.mean(v) for v in all_imp.values()],
        'CV': [np.std(v)/np.mean(v) if np.mean(v) > 0 else 0 for v in all_imp.values()]
    }).sort_values('Mean', ascending=False)
    return stability


def compute_ppv_npv(se, sp):
    """PPV and NPV at different prevalence levels."""
    results = []
    for prev in [0.01, 0.05, 0.10, 0.20]:
        ppv = (se * prev) / (se * prev + (1 - sp) * (1 - prev))
        npv = (sp * (1 - prev)) / (sp * (1 - prev) + (1 - se) * prev)
        results.append({'Prevalence': f'{prev*100:.0f}%', 'PPV': ppv, 'NPV': npv})
    return pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================

def main(output_dir=FIG_DIR):
    print('=' * 70)
    print('FuseVirome - Reproducible Pipeline')
    print('=' * 70)
    print('\n[1/7] Loading data...')
    X_train_raw, y_train, X_test_raw, y_test = load_data()
    print(f'  Training: {X_train_raw.shape}, Test: {X_test_raw.shape}')
    print('\n[2/7] Feature refinement pipeline...')
    r1 = stage1_null_removal(X_train_raw)
    X1 = X_train_raw.drop(columns=r1)
    r2 = stage2_fused_exclusion(X1)
    X2 = X1.drop(columns=r2)
    final_features = X2.columns.tolist()
    print(f'  Path: {X_train_raw.shape[1]} -> {X1.shape[1]} -> {X2.shape[1]}')
    print(f'  Removed: S1={r1}, S2={r2}')
    train_medians = X2.median()
    X_train = X2.fillna(train_medians)
    X_test = X_test_raw[final_features].fillna(train_medians)
    print('\n[3/7] Six-classifier comparison...')
    clf_results = compare_classifiers(X_train, y_train, X_test, y_test)
    print(clf_results.to_string(index=False))
    print('\n[4/7] Training final RF model...')
    rf = RandomForestClassifier(n_estimators=100, max_depth=3,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    print('\n[5/7] Eleven-configuration ablation study...')
    abl_results = ablation_study(X_train, y_train, X_test, y_test, final_features)
    print(abl_results.to_string(index=False))
    print('\n[6/7] Additional validation...')
    pi_df = compute_permutation_importance(rf, X_test, y_test, final_features)
    print('\nPermutation Importance (Top 5):')
    print(pi_df.head(5).to_string(index=False))
    stab_df = compute_feature_stability(X_train, y_train, final_features)
    print('\nFeature Stability (Top 5, CV<0.40):')
    print(stab_df.head(5).to_string(index=False))
    y_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    ppv_npv = compute_ppv_npv(se, sp)
    print('\nPPV/NPV at different prevalence:')
    print(ppv_npv.to_string(index=False))
    print('\n[7/7] Generating figures...')
    generate_figures(rf, X_train, y_train, X_test, y_test,
                     final_features, clf_results, abl_results,
                     fig_dir=output_dir)
    print('\n' + '=' * 70)
    print('FuseVirome - Complete')
    print('=' * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir',
        default=FIG_DIR,
        help='Directory for generated figures (default: outputs/figures).',
    )
    main(output_dir=parser.parse_args().output_dir)
