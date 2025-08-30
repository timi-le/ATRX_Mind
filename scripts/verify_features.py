#!/usr/bin/env python3
"""
Verify computed features in feature Parquet files
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_features():
    """Verify the computed features in all feature files"""
    
    features_dir = Path("data/features")
    
    print("🔍 FEATURE FILES VERIFICATION")
    print("=" * 70)
    
    for feature_file in sorted(features_dir.glob("*_features.parquet")):
        print(f"\n📄 {feature_file.name}")
        print("-" * 50)
        
        try:
            # Load and examine the data
            df = pd.read_parquet(feature_file)
            
            print(f"📊 Rows: {len(df):,}")
            print(f"📋 Total columns: {len(df.columns)}")
            print(f"💾 Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"🏷️  Symbol: {df['symbol'].iloc[0]}")
            
            # Identify original vs computed columns
            original_cols = ['timestamp', 'symbol', 'open', 'high', 'low', 'close',
                           'tick_volume', 'real_volume', 'spread',
                           'bid_open', 'bid_high', 'bid_low', 'bid_close',
                           'ask_open', 'ask_high', 'ask_low', 'ask_close',
                           'mid_open', 'mid_high', 'mid_low', 'mid_close',
                           'spread_close', 'bid_change', 'ask_change']
            
            feature_cols = [col for col in df.columns if col not in original_cols]
            
            print(f"🔧 Computed features ({len(feature_cols)}):")
            for i, col in enumerate(feature_cols, 1):
                non_null_pct = (1 - df[col].isnull().sum() / len(df)) * 100
                print(f"   {i:2d}. {col:<20} ({non_null_pct:5.1f}% non-null)")
            
            # Check for any features with high missing rates
            high_missing = [col for col in feature_cols 
                          if df[col].isnull().sum() / len(df) > 0.1]
            if high_missing:
                print(f"⚠️  High missing features: {high_missing}")
            
            # Show sample feature values
            print(f"\n💼 Sample feature values:")
            if feature_cols:
                sample_features = feature_cols[:5]  # Show first 5 features
                sample_data = df[sample_features].head(3)
                print(sample_data.to_string(index=False, float_format='%.6f'))
            
        except Exception as e:
            print(f"❌ Error reading {feature_file.name}: {e}")
    
    print(f"\n✅ Feature verification completed!")

if __name__ == "__main__":
    verify_features()
