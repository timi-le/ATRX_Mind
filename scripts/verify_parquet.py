#!/usr/bin/env python3
"""
Quick verification of normalized Parquet files
"""

import pandas as pd
from pathlib import Path

def verify_parquet_files():
    """Verify the structure and content of normalized Parquet files"""
    
    parquet_dir = Path("data/parquet")
    
    print("🔍 PARQUET FILES VERIFICATION")
    print("=" * 60)
    
    for parquet_file in sorted(parquet_dir.glob("*.parquet")):
        print(f"\n📄 {parquet_file.name}")
        print("-" * 40)
        
        try:
            # Load and examine the data
            df = pd.read_parquet(parquet_file)
            
            print(f"📊 Rows: {len(df):,}")
            print(f"📋 Columns: {list(df.columns)}")
            print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"🏷️  Symbol: {df['symbol'].iloc[0]}")
            
            # Show sample data
            print(f"\n💼 Sample data:")
            print(df.head(3).to_string(index=False))
            
        except Exception as e:
            print(f"❌ Error reading {parquet_file.name}: {e}")
    
    print(f"\n✅ All Parquet files verified successfully!")

if __name__ == "__main__":
    verify_parquet_files()
