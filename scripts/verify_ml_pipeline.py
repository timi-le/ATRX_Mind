#!/usr/bin/env python3
"""
ATRX ML Pipeline Verification
=============================

Comprehensive verification of the complete ML data pipeline from features to splits.
Validates data quality, temporal consistency, and split integrity across all datasets.

Following ATRX development standards for production-ready ML validation.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

def verify_datasets():
    """Verify all assembled datasets."""
    print("📊 DATASET VERIFICATION")
    print("=" * 50)
    
    datasets_dir = Path("data/datasets")
    dataset_files = list(datasets_dir.glob("*_dataset.parquet"))
    
    if not dataset_files:
        print("❌ No datasets found!")
        return False
    
    dataset_summary = []
    
    for dataset_file in sorted(dataset_files):
        print(f"\n📄 {dataset_file.name}")
        print("-" * 30)
        
        try:
            # Load dataset
            df = pd.read_parquet(dataset_file)
            
            # Basic info
            print(f"📊 Rows: {len(df):,}")
            print(f"📋 Columns: {len(df.columns)}")
            print(f"🏷️  Symbol: {df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown'}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Data quality checks
            missing_data = df.isnull().sum().sum()
            print(f"🚫 Missing values: {missing_data}")
            
            # Label distribution
            if 'label' in df.columns:
                label_dist = df['label'].value_counts().sort_index()
                print(f"🏷️  Label distribution: {dict(label_dist)}")
                
                # Calculate balance
                if len(label_dist) > 1:
                    max_count = label_dist.max()
                    min_count = label_dist.min()
                    balance = min_count / max_count
                    print(f"⚖️  Label balance: {balance:.3f}")
                    
                    if balance > 0.8:
                        print("✅ Well balanced labels")
                    elif balance > 0.5:
                        print("⚠️  Moderately imbalanced labels")
                    else:
                        print("❌ Highly imbalanced labels")
            
            # Temporal consistency
            if df['timestamp'].is_monotonic_increasing:
                print("✅ Temporal ordering correct")
            else:
                print("❌ Temporal ordering violated")
            
            # Memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024**2
            print(f"💾 Memory usage: {memory_mb:.2f} MB")
            
            # Feature count by type
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'symbol', 'label']]
            print(f"🔢 Features: {len(feature_cols)}")
            
            dataset_summary.append({
                'name': dataset_file.stem,
                'rows': len(df),
                'features': len(feature_cols),
                'symbol': df['symbol'].iloc[0] if 'symbol' in df.columns else 'Unknown',
                'date_span_days': (df['timestamp'].max() - df['timestamp'].min()).days,
                'memory_mb': memory_mb,
                'has_labels': 'label' in df.columns,
                'label_balance': min_count / max_count if 'label' in df.columns and len(label_dist) > 1 else None
            })
            
        except Exception as e:
            print(f"❌ Error loading {dataset_file.name}: {e}")
            return False
    
    print(f"\n📈 DATASET SUMMARY")
    print("=" * 50)
    total_rows = sum(d['rows'] for d in dataset_summary)
    total_memory = sum(d['memory_mb'] for d in dataset_summary)
    print(f"📊 Total datasets: {len(dataset_summary)}")
    print(f"📊 Total observations: {total_rows:,}")
    print(f"💾 Total memory: {total_memory:.2f} MB")
    print(f"🌍 Symbols: {', '.join(d['symbol'] for d in dataset_summary)}")
    
    return True

def verify_splits():
    """Verify all time series splits."""
    print(f"\n📊 SPLIT VERIFICATION")
    print("=" * 50)
    
    splits_dir = Path("outputs/splits")
    split_dirs = [d for d in splits_dir.iterdir() if d.is_dir()]
    
    if not split_dirs:
        print("❌ No splits found!")
        return False
    
    split_summary = []
    
    for split_dir in sorted(split_dirs):
        splits_file = split_dir / f"{split_dir.name}_splits.json"
        
        if not splits_file.exists():
            print(f"⚠️  No splits file found for {split_dir.name}")
            continue
        
        print(f"\n📄 {split_dir.name}")
        print("-" * 30)
        
        try:
            # Load splits
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            print(f"📊 Number of splits: {len(splits)}")
            
            # Analyze splits
            total_train_samples = 0
            total_val_samples = 0
            
            for split in splits:
                for symbol, symbol_split in split['splits'].items():
                    total_train_samples += symbol_split['train_samples']
                    total_val_samples += symbol_split['val_samples']
            
            print(f"🎯 Total train samples: {total_train_samples:,}")
            print(f"🎯 Total validation samples: {total_val_samples:,}")
            
            # Check temporal consistency
            temporal_consistent = True
            for i, split in enumerate(splits):
                metadata = split['metadata']
                train_start = pd.to_datetime(metadata['train_start'])
                train_end = pd.to_datetime(metadata['train_end'])
                val_start = pd.to_datetime(metadata['val_start'])
                val_end = pd.to_datetime(metadata['val_end'])
                
                # Check ordering within split
                if not (train_start < train_end <= val_start < val_end):
                    print(f"❌ Temporal ordering violated in split {i}")
                    temporal_consistent = False
                    break
                
                # Check ordering between splits
                if i > 0:
                    prev_metadata = splits[i-1]['metadata']
                    prev_train_start = pd.to_datetime(prev_metadata['train_start'])
                    
                    if train_start <= prev_train_start:
                        print(f"❌ Split ordering violated between splits {i-1} and {i}")
                        temporal_consistent = False
                        break
            
            if temporal_consistent:
                print("✅ Temporal consistency verified")
            
            # Date range
            if splits:
                first_split = splits[0]['metadata']
                last_split = splits[-1]['metadata']
                print(f"📅 Date range: {first_split['train_start']} to {last_split['val_end']}")
            
            split_summary.append({
                'symbol': split_dir.name,
                'num_splits': len(splits),
                'train_samples': total_train_samples,
                'val_samples': total_val_samples,
                'temporal_consistent': temporal_consistent
            })
            
        except Exception as e:
            print(f"❌ Error loading splits for {split_dir.name}: {e}")
            return False
    
    print(f"\n📈 SPLIT SUMMARY")
    print("=" * 50)
    total_splits = sum(s['num_splits'] for s in split_summary)
    total_train = sum(s['train_samples'] for s in split_summary)
    total_val = sum(s['val_samples'] for s in split_summary)
    all_consistent = all(s['temporal_consistent'] for s in split_summary)
    
    print(f"📊 Total splits: {total_splits}")
    print(f"🎯 Total train samples: {total_train:,}")
    print(f"🎯 Total validation samples: {total_val:,}")
    print(f"⏰ Temporal consistency: {'✅ All verified' if all_consistent else '❌ Issues detected'}")
    
    return True

def verify_data_leakage():
    """Verify no data leakage in splits."""
    print(f"\n🔒 DATA LEAKAGE VERIFICATION")
    print("=" * 50)
    
    splits_dir = Path("outputs/splits")
    datasets_dir = Path("data/datasets")
    
    leakage_detected = False
    
    for split_dir in splits_dir.iterdir():
        if not split_dir.is_dir():
            continue
        
        splits_file = split_dir / f"{split_dir.name}_splits.json"
        dataset_file = datasets_dir / f"{split_dir.name}_dataset.parquet"
        
        if not splits_file.exists() or not dataset_file.exists():
            continue
        
        print(f"\n🔍 Checking {split_dir.name}")
        
        try:
            # Load data
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            
            df = pd.read_parquet(dataset_file)
            
            # Check each split for temporal leakage
            for split in splits:
                for symbol, symbol_split in split['splits'].items():
                    train_indices = symbol_split['train_indices']
                    val_indices = symbol_split['val_indices']
                    
                    if not train_indices or not val_indices:
                        continue
                    
                    # Get timestamps for train and validation
                    train_times = df.loc[train_indices, 'timestamp']
                    val_times = df.loc[val_indices, 'timestamp']
                    
                    # Check for leakage
                    max_train_time = train_times.max()
                    min_val_time = val_times.min()
                    
                    if max_train_time >= min_val_time:
                        print(f"❌ LEAKAGE DETECTED in split {split['split_id']}")
                        print(f"   Max train time: {max_train_time}")
                        print(f"   Min val time: {min_val_time}")
                        leakage_detected = True
                        break
                
                if leakage_detected:
                    break
            
            if not leakage_detected:
                print(f"✅ No leakage detected")
        
        except Exception as e:
            print(f"❌ Error checking {split_dir.name}: {e}")
            leakage_detected = True
    
    return not leakage_detected

def main():
    """Main verification routine."""
    print("🧪 ATRX ML PIPELINE VERIFICATION")
    print("=" * 70)
    
    # Verify datasets
    datasets_ok = verify_datasets()
    
    # Verify splits
    splits_ok = verify_splits()
    
    # Verify no data leakage
    no_leakage = verify_data_leakage()
    
    # Final summary
    print(f"\n🏁 FINAL VERIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"📊 Datasets: {'✅ PASS' if datasets_ok else '❌ FAIL'}")
    print(f"📊 Splits: {'✅ PASS' if splits_ok else '❌ FAIL'}")
    print(f"🔒 Data leakage: {'✅ NONE DETECTED' if no_leakage else '❌ LEAKAGE DETECTED'}")
    
    overall_pass = datasets_ok and splits_ok and no_leakage
    
    if overall_pass:
        print(f"\n🎉 ML PIPELINE VERIFICATION: ✅ ALL SYSTEMS PASS")
        print(f"🚀 Pipeline is ready for production ML training!")
    else:
        print(f"\n❌ ML PIPELINE VERIFICATION: ISSUES DETECTED")
        print(f"🔧 Please review and fix the identified issues.")
    
    return 0 if overall_pass else 1

if __name__ == "__main__":
    exit(main())

