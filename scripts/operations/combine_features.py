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

    expected_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    print(f"Expecting {expected_rows:,} rows.")

    # Write to a temporary name and only rename on success. A killed run
    # otherwise leaves a partial file at the final path whose footer was never
    # written: it is the right size and the right name, and every reader fails
    # on it later with a message about magic bytes rather than about the run
    # that produced it. Being killed here is not hypothetical -- combining
    # while the feature pipeline was running exhausted memory and did exactly
    # this at file 32 of 69.
    staging = args.output.with_suffix(args.output.suffix + ".partial")
    if staging.exists():
        staging.unlink()

    # Use ParquetWriter to append without loading everything into memory
    schema = pq.ParquetFile(files[0]).schema_arrow

    total_rows = 0
    try:
        with pq.ParquetWriter(staging, schema, compression="snappy") as writer:
            for f in files:
                print(f"Streaming {f.name}...")
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=1_000_000):
                    writer.write_batch(batch)
                    total_rows += batch.num_rows
    except BaseException:
        staging.unlink(missing_ok=True)
        raise

    if total_rows != expected_rows:
        staging.unlink(missing_ok=True)
        print(f"Row mismatch: wrote {total_rows:,}, expected {expected_rows:,}.")
        sys.exit(1)

    # Re-open the finished file so a truncated footer is caught here rather
    # than by whatever reads the store next.
    written = pq.ParquetFile(staging).metadata.num_rows
    if written != expected_rows:
        staging.unlink(missing_ok=True)
        print(f"Verification failed: file holds {written:,} rows, expected {expected_rows:,}.")
        sys.exit(1)

    staging.replace(args.output)
    print(f"Total rows written: {total_rows:,}")
    print(f"Written to {args.output}")
    print("Done!")

if __name__ == "__main__":
    main()
