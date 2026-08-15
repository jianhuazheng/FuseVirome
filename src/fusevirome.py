#!/usr/bin/env python3
"""FuseVirome v15 - Complete Reproducible Pipeline
Machine learning framework with dynamic reference construction for scoring
mNGS-detected viral signals in veterinary metatranscriptomic sequencing.

Run: python fusevirome_v15.py
Output: All metrics and figures matching manuscript Tables 1-8 and Figures 1-7.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, recall_score,
    confusion_matrix, brier_score_loss, roc_curve)
from sklearn.model_selection import cross_val_score, cross_validate 
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    from sklearn.metrics import calibration_curve
from sklearn.inspection import permutation_importance

from statsmodels.stats.outliers_influence import variance_inflation_factor

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_Training_Data.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_External_Validation_Data.csv')
RANDOM_SEED = 0  
np.random.seed(RANDOM_SEED)

MASTER_TOP10 = {
    'Coverage_bp_contigs', 'Coverage_Percent_reads', 'Reads_Count_contigs',
    'BitScore_mean', 'Coverage_bp_reads', 'qlen_mean',
    'Avg_Coverage_Depth_x_contigs', 'BitScore_max',
    'Count_Correct_Contigs', 'Mismatches_sum'
}

FIG_DIR = os.path.join(BASE_DIR, 'figures', '')
os.makedirs(FIG_DIR, exist_ok=True)

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
    fig.set_size_inches(*PLOT_FIG_SIZE, forward=True)
    fig.savefig(out_path, dpi=PLOT_DPI, facecolor='white')


configure_plot_style()

# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data():
    """Load aggregated training and test data with leakage prevention."""
    train = pd.read_csv(TRAIN_PATH, encoding='gbk')
    test = pd.read_csv(TEST_PATH, encoding='gbk')
    
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
# 2. THREE-STAGE FEATURE REFINEMENT
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


def stage3_collinearity(X_train, feature_cols):
    """Master-priority collinearity filtering."""
    corr_matrix = X_train[feature_cols].fillna(0).corr().abs()
    removed = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                f1, f2 = corr_matrix.columns[i], corr_matrix.columns[j]
                if f1 in removed or f2 in removed:
                    continue
                f1m = f1 in MASTER_TOP10
                f2m = f2 in MASTER_TOP10
                if f1m and not f2m:
                    removed.add(f2)
                elif f2m and not f1m:
                    removed.add(f1)
                elif f1m and f2m:
                    removed.add(f2 if X_train[f1].std() >= X_train[f2].std() else f1)
                else:
                    removed.add(f2 if X_train[f1].std() >= X_train[f2].std() else f1)
    return list(removed)

# =============================================================================
# 3. SIX CLASSIFIER COMPARISON
# =============================================================================

def compare_classifiers(X_train, y_train, X_test, y_test):
    """Train and evaluate 6 classifiers (RF, ExtraTrees, GB, LR, kNN, DT)."""
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED),
        'LR': LogisticRegression(class_weight='balanced', max_iter=5000, random_state=RANDOM_SEED),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'DT': DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=RANDOM_SEED),
    }
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
    configure_plot_style()
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Common Color Map
    colors_map = {
        'RF': '#E74C3C', 'ExtraTrees': '#3498DB', 'GB': '#1ABC9C',
        'LR': '#2C3E50', 'kNN': '#E67E22', 'DT': '#16A085'
    }
    classifiers = {
        'RF': RandomForestClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=100, max_depth=3,
            class_weight='balanced', random_state=RANDOM_SEED, n_jobs=1),
        'GB': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_SEED),
        'LR': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'DT': DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=RANDOM_SEED),
    }

    # ------------------- Original Figure 2 -------------------
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pp = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pp)
        auc_val = roc_auc_score(y_test, pp)
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
    # NEW: [Figure 7] Grouped Boxplot for Recall, F1, and Accuracy (Test Set Bootstrap)
    # =========================================================================
    fig, ax = plt.subplots(figsize=PLOT_FIG_SIZE, dpi=PLOT_DPI)

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

    # Metric colors matching the requested style (red/green/blue)
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

    n_clf = len(clf_names)
    group_width = 0.55
    box_width = group_width / 3
    group_gap = 1.15
    
    # x-axis positions: one group per classifier, three boxes per group
    positions = []
    for i in range(n_clf):
        base = i * group_gap
        positions.append([
            base - box_width,
            base,
            base + box_width
        ])

    # Reordered specifically to match "Recall, F1, Accuracy"
    all_data = [boot_recall, boot_f1, boot_accuracy]
    metric_names = ['Recall', 'F1', 'Accuracy']

    for j, (data, metric) in enumerate(zip(all_data, metric_names)):
        pos = [p[j] for p in positions]
        bplot = ax.boxplot(
            data,
            positions=pos,
            widths=box_width * 0.9,
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
            
        # Dummy patch for legend
        ax.plot([], [], color=color, marker='s', markersize=10,
                markeredgecolor=edge_colors[metric], linestyle='None', label=metric)

    ax.set_xticks([i * group_gap for i in range(n_clf)])
    display_names = ['Random Forest' if n == 'RF' else n for n in clf_names]
    ax.set_xticklabels(display_names, fontsize=PLOT_LABEL_SIZE)
    ax.set_ylabel('Score (Test-Set Bootstrap)', fontsize=PLOT_LABEL_SIZE)
    ax.set_ylim(0.5, 1.02)
    ax.set_xlim(-0.75, (n_clf - 1) * group_gap + 0.75)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=PLOT_LEGEND_SIZE, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='-', alpha=0.35)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0 - 0.045, pos.width, pos.height])
    save_plot(fig, f'{fig_dir}Figure7_Metrics_Boxplots.png')
    plt.close()
    print(f"  Figure 7: Recall, F1, and Accuracy Boxplots saved")


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

def main():
    print('=' * 70)
    print('FuseVirome v15 - Reproducible Pipeline')
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
    X_train = X2.fillna(X2.median())
    X_test = X_test_raw[final_features].fillna(X_test_raw[final_features].median())
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
    generate_figures(rf, X_train.values, y_train, X_test.values, y_test,
                     final_features, clf_results, abl_results)
    print('\n' + '=' * 70)
    print('FuseVirome v15 - Complete')
    print('=' * 70)

if __name__ == '__main__':
    main()
