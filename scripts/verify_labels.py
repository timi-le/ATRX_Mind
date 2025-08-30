#!/usr/bin/env python3
"""
Verify generated Triple-Barrier labels
"""

import pandas as pd
import numpy as np
from pathlib import Path

def verify_labels():
    """Verify the generated label files"""
    
    labels_dir = Path("data/labels")
    
    print("🏷️  TRIPLE-BARRIER LABELS VERIFICATION")
    print("=" * 70)
    
    for label_file in sorted(labels_dir.glob("*_labels.parquet")):
        print(f"\n📄 {label_file.name}")
        print("-" * 50)
        
        try:
            # Load and examine the labels
            df = pd.read_parquet(label_file)
            
            print(f"📊 Total observations: {len(df):,}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"🏷️  Symbols: {df['symbol'].unique()}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            # Analyze label distribution
            label_counts = df['label'].value_counts().sort_index()
            total_valid = df['label'].notna().sum()
            total_missing = df['label'].isna().sum()
            
            print(f"\n📈 Label Distribution:")
            for label, count in label_counts.items():
                percentage = count / total_valid * 100 if total_valid > 0 else 0
                if label == 1:
                    print(f"   🔼 Upper hits (+1): {count:,} ({percentage:.1f}%)")
                elif label == -1:
                    print(f"   🔽 Lower hits (-1): {count:,} ({percentage:.1f}%)")
                elif label == 0:
                    print(f"   ⏱️  Timeouts (0):   {count:,} ({percentage:.1f}%)")
                else:
                    print(f"   ❓ Other ({label}):   {count:,} ({percentage:.1f}%)")
            
            if total_missing > 0:
                missing_pct = total_missing / len(df) * 100
                print(f"   🚫 Missing (NaN):   {total_missing:,} ({missing_pct:.1f}%)")
            
            # Quality checks
            print(f"\n✅ Quality Metrics:")
            
            # Check for reasonable label balance
            if total_valid > 0:
                upper_pct = label_counts.get(1, 0) / total_valid * 100
                lower_pct = label_counts.get(-1, 0) / total_valid * 100
                timeout_pct = label_counts.get(0, 0) / total_valid * 100
                
                imbalance = abs(upper_pct - lower_pct)
                if imbalance < 10:
                    print(f"   📊 Label balance: ✅ Well balanced ({imbalance:.1f}% imbalance)")
                elif imbalance < 20:
                    print(f"   📊 Label balance: ⚠️  Moderate imbalance ({imbalance:.1f}%)")
                else:
                    print(f"   📊 Label balance: ❌ High imbalance ({imbalance:.1f}%)")
                
                # Check timeout ratio
                if timeout_pct > 80:
                    print(f"   ⏱️  Timeout ratio: ⚠️  High ({timeout_pct:.1f}%) - consider smaller barriers")
                elif timeout_pct < 20:
                    print(f"   ⏱️  Timeout ratio: ⚠️  Low ({timeout_pct:.1f}%) - consider larger barriers")
                else:
                    print(f"   ⏱️  Timeout ratio: ✅ Reasonable ({timeout_pct:.1f}%)")
            
            # Check data integrity
            if df['timestamp'].is_monotonic_increasing:
                print(f"   📅 Time order: ✅ Properly sorted")
            else:
                print(f"   📅 Time order: ❌ Not sorted")
            
            if df['symbol'].notna().all():
                print(f"   🏷️  Symbols: ✅ All present")
            else:
                print(f"   🏷️  Symbols: ❌ Missing values")
            
            # Sample data
            print(f"\n💼 Sample labels:")
            sample_data = df[['timestamp', 'symbol', 'label']].head(5)
            print(sample_data.to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error reading {label_file.name}: {e}")
    
    print(f"\n✅ Label verification completed!")

if __name__ == "__main__":
    verify_labels()
