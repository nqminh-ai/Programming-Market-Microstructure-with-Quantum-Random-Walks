"""Read selected columns of a feature store without exhausting memory.

The research scripts each grew their own loader, and each arrived at the same
three mistakes: reading every column at its stored width, sorting the result
unconditionally, and downcasting only afterwards. At 32M rows none of that
mattered. At the 69-day stores -- 227M rows for BTCUSDT -- the plain
``read_parquet`` of seven columns is 7.7GB before the sort doubles it, and the
process is killed on a 16GB machine.

One implementation, so the next study to need a feature store does not
rediscover the problem. The approach is:

* read one column at a time and release each Arrow buffer as soon as it has
  been converted -- reading the whole table and calling ``to_pandas`` holds the
  Arrow copy and the pandas copy at once;
* apply the narrowing cast *during* that conversion, not after, so the wide
  version never coexists with the rest of the frame;
* cap rows per column rather than after assembling everything;
* sort only when the data is not already ordered, since sorting copies the
  whole frame and the stores are written in date order from per-day files.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _read_column(
    handle: pq.ParquetFile, path: Path, name: str, max_rows: int
) -> pa.ChunkedArray:
    """Read ``name``, stopping once ``max_rows`` rows are in hand.

    Reading the column whole and slicing afterwards does not bound anything:
    the slice is a view onto the full-length array, so a 100M-row cap on a
    227M-row store still costs all 227M rows. Row groups are the smallest unit
    parquet lets us stop at, so the overshoot is at most one of them.
    """
    if not max_rows:
        return pq.read_table(path, columns=[name]).column(0)

    blocks = []
    rows = 0
    for index in range(handle.metadata.num_row_groups):
        block = handle.read_row_group(index, columns=[name])
        blocks.append(block)
        rows += block.num_rows
        if rows >= max_rows:
            break
    return pa.concat_tables(blocks).column(0)


def load_feature_columns(
    path: str | Path,
    columns: Sequence[str],
    *,
    downcast: Mapping[str, str] | None = None,
    max_rows: int = 0,
    sort_by: str | None = "timestamp",
) -> pd.DataFrame:
    """Return ``columns`` from the parquet at ``path`` as a DataFrame.

    Columns absent from the file are skipped rather than raising, matching what
    the callers already did: a store written before a feature existed is a
    reason to fall back, not to fail.

    ``downcast`` maps a column name to the dtype it should be read as. ``price``
    is deliberately never in those maps: log returns over long horizons need
    float64, and float32 on a ~60,000 price loses the tick-scale differences
    the labels depend on.
    """
    path = Path(path)
    handle = pq.ParquetFile(path)
    present = set(handle.schema.names)
    wanted = [name for name in columns if name in present]
    if not wanted:
        raise ValueError(f"{path.name} has none of the requested columns: {list(columns)}")
    casts = dict(downcast or {})

    data: dict[str, np.ndarray] = {}
    for name in wanted:
        column = _read_column(handle, path, name, max_rows)
        dtype = casts.get(name)
        if dtype is not None:
            column = column.cast(pa.type_for_alias(dtype))
        values = column.to_numpy(zero_copy_only=False)
        del column
        if max_rows and len(values) > max_rows:
            # Only the tail of one row group, so the view's base is at most a
            # row group larger than the slice. Slicing a *fully* read column
            # would not free anything: a numpy slice keeps its base alive, so
            # the cap would shrink the visible length and nothing else.
            values = values[:max_rows]
        data[name] = values
        gc.collect()

    frame = pd.DataFrame(data, copy=False)
    del data
    gc.collect()

    if sort_by and sort_by in frame.columns and not frame[sort_by].is_monotonic_increasing:
        frame = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
        gc.collect()
    return frame
