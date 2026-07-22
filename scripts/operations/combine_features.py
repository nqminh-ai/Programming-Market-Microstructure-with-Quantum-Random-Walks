"""Stream-combine per-day feature parquet files into one multi-day store.

Usage
-----
    python scripts/operations/combine_features.py [--symbol BTCUSDT] [--days 30]
"""

import argparse
import re
from pathlib import Path
import sys
import pyarrow.parquet as pq
import pyarrow as pa

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol whose feature files to combine")
    parser.add_argument("--input-dir", type=Path, default=Path("data/assets/btcusdt/features"))
    parser.add_argument("--output", type=Path, default=Path("data/assets/btcusdt/features/features_BTCUSDT_multiday.parquet"))
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    # No year restriction: a hardcoded "2026-*" pattern would silently skip
    # files once the symbol/year changed without any warning.
    pattern = f"features_{args.symbol}_*.parquet"
    matched = sorted(args.input_dir.glob(pattern))
    # Only per-day files may be combined. The glob also matches previously
    # written aggregates living in the same directory -- features_X_multiday
    # and features_X_recent_subset -- and folding those back in silently
    # duplicates every row they already contain.
    dated = re.compile(rf"^features_{re.escape(args.symbol)}_\d{{4}}-\d{{2}}-\d{{2}}\.parquet$")
    files = [path for path in matched if dated.match(path.name)]
    skipped = [path.name for path in matched if not dated.match(path.name)]
    print(f"Matched {len(matched)} file(s) in {args.input_dir} for pattern {pattern!r}.")
    if skipped:
        print(f"Skipping {len(skipped)} non-daily file(s): {', '.join(skipped)}")
    # Only take the last `days` files to keep memory usage reasonable
    if len(files) > args.days:
        files = files[-args.days:]

    if not files:
        print("No files found!")
        sys.exit(1)

    print(f"Found {len(files)} files to combine (last {args.days} days).")
    
    # Use ParquetWriter to append without loading everything into memory
    first_file = pq.ParquetFile(files[0])
    schema = first_file.schema_arrow
    
    writer = pq.ParquetWriter(args.output, schema, compression="snappy")
    
    total_rows = 0
    try:
        for f in files:
            if f.name == args.output.name:
                continue
            print(f"Streaming {f.name}...")
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=1_000_000):
                writer.write_batch(batch)
                total_rows += batch.num_rows
    finally:
        writer.close()
        
    print(f"Total rows written: {total_rows:,}")
    print(f"Written to {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()
