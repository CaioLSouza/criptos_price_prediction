# Weekly Niño 3.4 anomaly data

`nino34_weekly.csv` holds the weekly sea surface temperature (SST) and SST
anomaly (SSTA) for the four Niño regions, from NOAA's Climate Prediction Center
weekly index file. The Niño 3.4 anomaly (`nino34_ssta`) is the standard ENSO
(El Niño / La Niña) indicator and is intended here as an exogenous variable for
the crypto price models.

## Columns

| column | description |
| --- | --- |
| `week_centered_date` | ISO date of the week center (weeks are Wednesday-centered) |
| `nino1_2_sst`, `nino1_2_ssta` | Niño 1+2 region SST (°C) and anomaly |
| `nino3_sst`, `nino3_ssta` | Niño 3 region SST (°C) and anomaly |
| `nino34_sst`, `nino34_ssta` | Niño 3.4 region SST (°C) and anomaly ← ENSO index |
| `nino4_sst`, `nino4_ssta` | Niño 4 region SST (°C) and anomaly |

Anomalies are relative to the 1991–2020 climatology (CPC `wksst9120.for`).

## Source and refresh

- Source: <https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for>
- Series starts the week centered on 1990-01-03 and is updated by CPC every
  Monday.
- The file is produced by `fetch_nino34_weekly.py` (standard library only).

NOAA is not reachable from the Claude Code web sandbox (its egress is limited to
GitHub), so the CSV is generated and committed by the
`.github/workflows/nino34-weekly.yml` GitHub Action, which runs the script on a
GitHub-hosted runner. To refresh locally in an environment with internet:

```bash
python input_data/fetch_nino34_weekly.py
```
