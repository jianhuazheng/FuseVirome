#!/usr/bin/env python3
"""
FuseVirome Data Preparation Pipeline
====================================
Transforms raw sequencing alignment data into aggregated sample-level
feature matrices for downstream machine learning.

Input:
    - data/raw/data_base.csv (training raw alignments)
    - data/raw/data_test.csv (external test raw alignments)

Output:
    - data/processed/Aggregated_Training_Data.csv
    - data/processed/Aggregated_External_Validation_Data.csv

Usage:
    python src/data_preparation.py

    Or place raw files in data/ and run directly; the script will fall back
    to the project-root data/ directory if data/raw/ is not present.
"""

import os
import re
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TRAIN = os.path.join(BASE_DIR, 'data', 'raw', 'data_base.csv')
RAW_TEST = os.path.join(BASE_DIR, 'data', 'raw', 'data_test.csv')
OUT_TRAIN = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_Training_Data.csv')
OUT_TEST = os.path.join(BASE_DIR, 'data', 'processed', 'Aggregated_External_Validation_Data.csv')

ENCODING = 'gbk'  # Original data encoding

# =============================================================================
# PARSING UTILITIES
# =============================================================================

def parse_size(size_str):
    """Parse size strings like '1.2 Gbp', '3.5 Mbp', '500 Kbp' into base pairs."""
    if pd.isna(size_str) or not isinstance(size_str, str) or str(size_str).lower() == 'na':
        return np.nan
    size_str = str(size_str).strip().replace(',', '')
    match = re.search(r'([\d.]+)\s*(Gbp|Mbp|Kbp|bp)?', size_str, re.IGNORECASE)
    if not match:
        return np.nan
    value = float(match.group(1))
    unit = match.group(2)
    if unit:
        unit = unit.lower()
        if unit == 'gbp':
            value *= 1e9
        elif unit == 'mbp':
            value *= 1e6
        elif unit == 'kbp':
            value *= 1e3
    return value


def parse_percent(val):
    """
    Parse percentage values. Handles both string ('95.5%') and numeric inputs.
    
    BUG FIX: Original code only handled strings, causing float64 values to
    return NaN. This version correctly processes both types.
    """
    if pd.isna(val):
        return np.nan
    # Handle float/int directly (THE FIX)
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str) or str(val).lower() == 'na':
        return np.nan
    try:
        return float(str(val).replace('%', '').strip())
    except ValueError:
        return np.nan


# =============================================================================
# COLUMN RENAMING
# =============================================================================

def rename_columns(df):
    """
    Rename Chinese column names to English identifiers.
    
    Column naming convention:
        - Base columns: no suffix (read-track)
        - .1 suffix: contig-track
        - .2 suffix: fused-track
    """
    rename_map = {
        # Base identifiers
        '样本ID': 'Sample_ID',
        '病毒简称': 'Virus_Name',
        
        # Read-track columns (suffix _reads)
        'Reads 数': 'Reads_Count_reads',
        '覆盖碱基数': 'Coverage_bp_reads',
        '覆盖百分比 (%)': 'Coverage_Percent_reads',
        '平均覆盖深度 (x)': 'Avg_Coverage_Depth_x_reads',
        '均值 baseQ': 'Mean_BaseQ_reads',
        '均值 mapQ': 'Mean_MapQ_reads',
        'Histo max bin (%)': 'Histo_Max_Bin_Percent_reads',
        
        # Contig-track columns (.1 suffix -> _contigs)
        'Reads 数.1': 'Reads_Count_contigs',
        '覆盖碱基数.1': 'Coverage_bp_contigs',
        '覆盖百分比 (%).1': 'Coverage_Percent_contigs',
        '平均覆盖深度 (x).1': 'Avg_Coverage_Depth_x_contigs',
        '均值 baseQ.1': 'Mean_BaseQ_contigs',
        '均值 mapQ.1': 'Mean_MapQ_contigs',
        'Histo max bin (%).1': 'Histo_Max_Bin_Percent_contigs',
        
        # Fused-track columns (.2 suffix -> _fused)
        'Reads 数.2': 'Reads_Count_fused',
        '覆盖碱基数.2': 'Coverage_bp_fused',
        '覆盖百分比 (%).2': 'Coverage_Percent_fused',
        '平均覆盖深度 (x).2': 'Avg_Coverage_Depth_x_fused',
        '均值 baseQ.2': 'Mean_BaseQ_fused',
        '均值 mapQ.2': 'Mean_MapQ_fused',
        'Histo max bin (%).2': 'Histo_Max_Bin_Percent_fused',
        
        # Annotation columns
        'Anno_Annotation_Rate_%': 'Anno_Rate',
        'Anno_Matched_Contigs_Count': 'Anno_Matched_Count',
        'Anno_Annotated_Contigs_Count': 'Anno_Annotated_Count',
        
        # Contig alignment columns (already English, keep as-is)
        'identity': 'identity',
        'AlignLength': 'AlignLength',
        'Mismatches': 'Mismatches',
        'GapOpens': 'GapOpens',
        'Evalue': 'Evalue',
        'BitScore': 'BitScore',
        'qlen': 'qlen',
        'slen': 'slen',
        'Right_Annotated_Contigs': 'Right_Annotated_Contigs',
        'Annotation_Found': 'Annotation_Found',
        'qPCR': 'qPCR',
    }
    
    df = df.rename(columns=rename_map)
    return df


# =============================================================================
# DATA CLEANING
# =============================================================================

def clean_raw_data(df, is_train=True):
    """
    Clean raw alignment data.
    
    Steps:
        1. Filter out qPCR == 4 (indeterminate results)
        2. Parse Coverage_bp (size strings -> numeric)
        3. Parse Coverage_Percent (BUG FIXED: handles float64)
        4. Convert numeric columns
        5. Drop unused metadata columns
    """
    # 1. Filter qPCR
    if 'qPCR' in df.columns:
        df['qPCR'] = pd.to_numeric(df['qPCR'], errors='coerce').fillna(-1).astype(int)
        n_before = len(df)
        df = df[df['qPCR'] != 4].copy()
        n_after = len(df)
        if n_before != n_after:
            print(f"  Filtered qPCR==4: {n_before} -> {n_after} rows")
    
    # 2. Parse Coverage_bp (size strings)
    for col in ['Coverage_bp_reads', 'Coverage_bp_contigs', 'Coverage_bp_fused']:
        if col in df.columns:
            df[col] = df[col].apply(parse_size)
    
    # 3. Parse Coverage_Percent and Histo_Max_Bin_Percent (BUG FIX)
    percent_cols = [c for c in df.columns if 'Percent' in c or 'Histo' in c]
    for col in percent_cols:
        df[col] = df[col].apply(parse_percent)
    
    # 4. Convert numeric columns
    numeric_cols = [
        'Reads_Count_reads', 'Avg_Coverage_Depth_x_reads', 'Mean_BaseQ_reads', 'Mean_MapQ_reads',
        'Reads_Count_contigs', 'Avg_Coverage_Depth_x_contigs', 'Mean_BaseQ_contigs', 'Mean_MapQ_contigs',
        'Reads_Count_fused', 'Avg_Coverage_Depth_x_fused', 'Mean_BaseQ_fused', 'Mean_MapQ_fused',
        'identity', 'AlignLength', 'Mismatches', 'GapOpens', 'Evalue', 'BitScore', 'qlen', 'slen',
        'Anno_Rate', 'Anno_Matched_Count', 'Anno_Annotated_Count'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 5. Drop unused columns
    cols_to_drop = [
        'Contig_ID', 'accession.version', 'QueryStart', 'QueryEnd',
        'SubjectStart', 'SubjectEnd', 'taxid', 'Kingdom', 'Phylum',
        'class', 'Order', 'Family', 'Genus', 'Species', 'Strain',
        '中文名称', '参考长度', '中文名称.1', '参考长度.1', '中文名称.2', '参考长度.2',
        '中文名称_integrated', '样本ID.1', '病毒简称.1', '样本ID.2', '病毒简称.2',
    ]
    existing_to_drop = [c for c in cols_to_drop if c in df.columns]
    # Also drop Chinese-name columns that were renamed
    for col in list(df.columns):
        if any(kw in str(col) for kw in ['中文', '参考长度']):
            existing_to_drop.append(col)
    existing_to_drop = list(set(existing_to_drop))
    df = df.drop(columns=existing_to_drop, errors='ignore')
    
    # Drop Unnamed columns
    unnamed = [c for c in df.columns if 'Unnamed' in str(c)]
    if unnamed:
        df = df.drop(columns=unnamed, errors='ignore')
    
    return df


# =============================================================================
# DATA AGGREGATION
# =============================================================================

def aggregate_sample_level(df, label_fn='max'):
    """
    Aggregate contig-level data to sample-level features.
    
    Aggregation rules:
        - Sample columns: .first() (same within group)
        - Count features: .size(), .sum()
        - Quality features: .max(), .mean(), .sum() (on correctly annotated contigs only)
        - Label: .max() for train, .first() for test
    """
    group_keys = ['Sample_ID', 'Virus_Name']
    
    # A: Sample-level columns (coverage, baseQ, mapQ, etc.) - take first
    sample_cols = [c for c in df.columns if c.endswith(('_reads', '_contigs', '_fused', '_Percent')) and c != 'qPCR']
    df_sample = df.groupby(group_keys)[sample_cols].first()
    
    # B: Count features
    df_total = df.groupby(group_keys).size().rename('Count_Total_Contigs')
    df_correct = df.groupby(group_keys)['Right_Annotated_Contigs'].sum().rename('Count_Correct_Contigs')
    df_counts = pd.concat([df_total, df_correct], axis=1)
    df_counts['Count_Noise_Contigs'] = df_counts['Count_Total_Contigs'] - df_counts['Count_Correct_Contigs']
    
    # C: Quality features (only on correctly annotated contigs)
    df_correct_only = df[df['Right_Annotated_Contigs'] == 1].copy()
    agg_map = {
        'identity': ['max', 'mean'],
        'BitScore': ['max', 'mean', 'sum'],
        'AlignLength': ['max', 'sum'],
        'qlen': ['mean'],
        'Evalue': ['min'],
        'Mismatches': ['mean']
    }
    valid_agg = {k: v for k, v in agg_map.items() if k in df.columns}
    
    if valid_agg and len(df_correct_only) > 0:
        df_quality = df_correct_only.groupby(group_keys).agg(valid_agg)
        df_quality.columns = ['_'.join(col).strip() for col in df_quality.columns.values]
    else:
        df_quality = pd.DataFrame(index=df.groupby(group_keys).size().index)
    
    # Merge all
    df_agg = df_sample.join(df_counts, how='left').join(df_quality, how='left')
    
    # D: Derived features
    eps = 1e-6
    if 'AlignLength_max' in df_agg.columns and 'qlen_mean' in df_agg.columns:
        df_agg['Align_Qlen_Ratio_Max'] = df_agg['AlignLength_max'] / (df_agg['qlen_mean'] + eps)
    
    # E: Annotation rate
    if 'Anno_Rate' in df.columns:
        anno_rate = df.groupby(group_keys)['Anno_Rate'].mean().rename('Anno_Rate')
        df_agg = df_agg.join(anno_rate, how='left')
    elif 'Annotation_Found' in df.columns:
        anno_rate = df.groupby(group_keys)['Annotation_Found'].mean().rename('Anno_Rate')
        df_agg = df_agg.join(anno_rate, how='left')
    
    # F: Label
    if label_fn == 'max':
        y_label = df.groupby(group_keys)['qPCR'].max().rename('Y_label_max')
    else:
        y_label = df.groupby(group_keys)['qPCR'].first().rename('Y_label_max')
    
    df_agg = df_agg.join(y_label, how='left')
    df_agg['Y'] = (df_agg['Y_label_max'] > 0).astype(int)
    
    # Clean up
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan)
    df_agg = df_agg.reset_index()
    
    # Fill remaining NaN with 0
    numeric_cols = df_agg.select_dtypes(include=[np.number]).columns
    df_agg[numeric_cols] = df_agg[numeric_cols].fillna(0)
    
    return df_agg


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_dataset(raw_path, out_path, label_fn='max', dataset_name='Training'):
    """Process a single dataset: load -> rename -> clean -> aggregate -> save."""
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*60}")
    
    # 1. Load
    print(f"[1/5] Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path, encoding=ENCODING)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # 2. Rename
    print(f"[2/5] Renaming columns...")
    df = rename_columns(df)
    print(f"  Renamed to {len(df.columns)} English columns")
    
    # 3. Clean
    print(f"[3/5] Cleaning data (parse_percent BUG FIXED)...")
    df = clean_raw_data(df, is_train=(label_fn=='max'))
    print(f"  After cleaning: {len(df)} rows, {len(df.columns)} columns")
    
    # 4. Aggregate
    print(f"[4/5] Aggregating to sample-level...")
    df_agg = aggregate_sample_level(df, label_fn=label_fn)
    print(f"  Aggregated: {len(df_agg)} samples")
    print(f"  Positive: {df_agg['Y'].sum()}, Negative: {(df_agg['Y']==0).sum()}")
    print(f"  Features: {len([c for c in df_agg.columns if c not in ['Sample_ID', 'Virus_Name', 'Y', 'Y_label_max']])}")
    
    # 5. Save
    print(f"[5/5] Saving to {out_path}...")
    df_agg.to_csv(out_path, index=False)
    print(f"  Saved: {os.path.getsize(out_path):,} bytes")
    
    return df_agg


def main():
    print("="*60)
    print("FuseVirome Data Preparation Pipeline")
    print("="*60)
    print("\nThis pipeline:")
    print("  1. Loads raw alignment data (data_base.csv / data_test.csv)")
    print("  2. Renames Chinese columns to English identifiers")
    print("  3. Cleans data (parse_percent BUG FIXED)")
    print("  4. Aggregates contig-level -> sample-level features")
    print("  5. Saves Aggregated_Training_Data.csv and Aggregated_External_Validation_Data.csv")
    
    # Check input files and adjust paths if needed
    global RAW_TRAIN, RAW_TEST

    if not os.path.exists(RAW_TRAIN):
        alt_train = os.path.join(BASE_DIR, 'data', 'data_base.csv')
        if os.path.exists(alt_train):
            RAW_TRAIN = alt_train
        else:
            print(f"\nERROR: Training raw data not found at {RAW_TRAIN}")
            print("Please place data_base.csv in data/raw/ or data/ directory")
            return

    if not os.path.exists(RAW_TEST):
        alt_test = os.path.join(BASE_DIR, 'data', 'data_test.csv')
        if os.path.exists(alt_test):
            RAW_TEST = alt_test
        else:
            print(f"\nERROR: Test raw data not found at {RAW_TEST}")
            print("Please place data_test.csv in data/raw/ or data/ directory")
            return
    
    # Process training data
    train_agg = process_dataset(RAW_TRAIN, OUT_TRAIN, label_fn='max', dataset_name='Training')
    
    # Process external test data
    test_agg = process_dataset(RAW_TEST, OUT_TEST, label_fn='first', dataset_name='External Test')
    
    print(f"\n{'='*60}")
    print("Data Preparation Complete")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  Training: {OUT_TRAIN} ({len(train_agg)} samples)")
    print(f"  Test:     {OUT_TEST} ({len(test_agg)} samples)")
    print(f"\nNext step: Run fusevirome_v9.py for full ML analysis")


if __name__ == '__main__':
    main()
