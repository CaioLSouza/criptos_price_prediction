"""Download and parse the weekly Niño region SST / SST-anomaly data from NOAA CPC.

Source: NOAA Climate Prediction Center weekly SST index file. The file is a
fixed-width text table with one row per week (week-centered date) and eight
value columns: SST and SST anomaly (SSTA) for each of the Niño 1+2, Niño 3,
Niño 3.4 and Niño 4 regions.

    https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for  (base 1991-2020)
    https://www.cpc.ncep.noaa.gov/data/indices/wksst8110.for  (base 1981-2010)

Header layout (first lines of the file):

    Weekly SST data starts week centered on 3Jan1990
             Nino1+2      Nino3        Nino34        Nino4
    Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA
    03JAN1990     23.4-0.4     25.1-0.3     26.6 0.0     28.6 0.3

The SST and SSTA fields can touch when the anomaly is negative (e.g.
"23.4-0.4"), so the eight values are extracted with a signed-decimal regex
(``-?\\d+\\.\\d+``) which splits "23.4-0.4" into 23.4 and -0.4 correctly and
does not depend on exact column spacing.

Running the module downloads the latest file and writes a tidy CSV to
``nino34_weekly.csv`` next to this script. It uses only the Python standard
library so it can run on a bare runner without installing anything.
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

# Primary is the current 1991-2020 climatology file; the older 1981-2010 file is
# kept as a fallback in case CPC renames or temporarily removes the primary.
SOURCE_URLS = (
    "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for",
    "https://www.cpc.ncep.noaa.gov/data/indices/wksst8110.for",
)

# The eight numeric fields in file order (region SST/SSTA pairs).
_VALUE_COLUMNS = (
    "nino1_2_sst",
    "nino1_2_ssta",
    "nino3_sst",
    "nino3_ssta",
    "nino34_sst",
    "nino34_ssta",
    "nino4_sst",
    "nino4_ssta",
)

# Week-centered date at the start of every data row, e.g. "03JAN1990".
_DATE_RE = re.compile(r"^\s*(\d{2}[A-Za-z]{3}\d{4})")
# Signed decimal values; "23.4-0.4" -> ["23.4", "-0.4"].
_VALUE_RE = re.compile(r"-?\d+\.\d+")

OUTPUT_COLUMNS = [
    "week_centered_date",
    "nino1_2_sst",
    "nino1_2_ssta",
    "nino3_sst",
    "nino3_ssta",
    "nino34_sst",
    "nino34_ssta",
    "nino4_sst",
    "nino4_ssta",
]


def fetch_raw(urls=SOURCE_URLS) -> str:
    """Return the raw text of the first source URL that responds successfully."""
    last_error: Exception | None = None
    for url in urls:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:  # noqa: S310 (trusted host)
                return response.read().decode("utf-8", errors="replace")
        except Exception as error:  # pragma: no cover - network dependent
            last_error = error
            print(f"WARN: failed to download {url}: {error}", file=sys.stderr)
    raise RuntimeError(f"Could not download weekly Nino data from any source: {last_error}")


def parse(raw: str) -> list[dict]:
    """Parse the CPC weekly file into a list of row dicts."""
    rows: list[dict] = []
    for line in raw.splitlines():
        date_match = _DATE_RE.match(line)
        if not date_match:
            continue
        try:
            week = datetime.strptime(date_match.group(1).upper(), "%d%b%Y").date()
        except ValueError:
            continue
        values = _VALUE_RE.findall(line[date_match.end():])
        if len(values) != len(_VALUE_COLUMNS):
            continue
        record = {"week_centered_date": week.isoformat()}
        record.update({col: float(v) for col, v in zip(_VALUE_COLUMNS, values)})
        rows.append(record)
    if not rows:
        raise ValueError("No data rows parsed; the source format may have changed.")
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_path = Path(__file__).resolve().parent / "nino34_weekly.csv"
    rows = parse(fetch_raw())
    write_csv(rows, out_path)
    first, last = rows[0]["week_centered_date"], rows[-1]["week_centered_date"]
    print(f"Wrote {len(rows)} weekly rows ({first} -> {last}) to {out_path}")


if __name__ == "__main__":
    main()
