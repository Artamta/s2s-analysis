# S2S Master Model Data Dictionary

This document contains the exact variables, units, lead times, and grid dimensions for all 5 datasets used in the analysis.

### ERA5 (Ground Truth)

**Dimensions & Grid:**
- `time`: 24
- `latitude`: 201
- `longitude`: 201

**Variables & Units:**
- `z500`: Geopotential (m**2 s**-2)
- `t2m`: 2 metre temperature (K)
- `tp`: Total precipitation (m)

**Key Coordinates Details:**

---
### FuXi S2S

**Dimensions & Grid:**
- `time`: 2
- `channel`: 76
- `lat`: 121
- `lon`: 240

**Variables & Units:**
- `data`: N/A (N/A)

**Key Coordinates Details:**

---
### Spire (mean_stddev group)

**Dimensions & Grid:**
- `reference_time`: 13
- `step`: 46
- `isobar`: 4
- `latitude`: 361
- `longitude`: 720

**Variables & Units:**
- `geopotential_height_at_isobaric_levels`: Geopotential height at pressure levels (m)
- `air_temperature`: 2m air temperature (K)
- `precipitation_amount_stddev`: Ensemble std dev of Accumulated precipitation (kg m-2)
- `precipitation_amount`: Accumulated precipitation (kg m-2)
- `air_temperature_stddev`: Ensemble std dev of 2m air temperature (K)
- `geopotential_height_at_isobaric_levels_stddev`: Ensemble std dev of Geopotential height at pressure levels (m)

**Key Coordinates Details:**
- **Lead Times (step)**: 46 steps, Max: 3974400 seconds

---
### ECMWF (S2S Archive)

#### GRIB Message Group 1
 ECMWF Group 1

**Dimensions & Grid:**
- `number`: 100
- `step`: 46
- `latitude`: 34
- `longitude`: 34

**Variables & Units:**
- `mx2t6`: Maximum temperature at 2 metres in the last 6 hours (K)
- `mn2t6`: Minimum temperature at 2 metres in the last 6 hours (K)

**Key Coordinates Details:**
- **Lead Times (step)**: 46 steps, Max: 3974400000000000 nanoseconds
- **Ensemble Members (number)**: 100 members.

---
#### GRIB Message Group 2
 ECMWF Group 2

**Dimensions & Grid:**
- `number`: 100
- `latitude`: 34
- `longitude`: 34

**Variables & Units:**
- `t2m`: 2 metre temperature (K)

**Key Coordinates Details:**
- **Lead Times (step)**: scalar value: 86400000000000 nanoseconds
- **Ensemble Members (number)**: 100 members.

---
#### GRIB Message Group 3
 ECMWF Group 3

**Dimensions & Grid:**
- `number`: 100
- `step`: 46
- `latitude`: 34
- `longitude`: 34

**Variables & Units:**
- `tp`: Total Precipitation (kg m**-2)

**Key Coordinates Details:**
- **Lead Times (step)**: 46 steps, Max: 3974400000000000 nanoseconds
- **Ensemble Members (number)**: 100 members.

---
### NCEP (S2S Archive)

#### GRIB Message Group 1
 NCEP Group 1

**Dimensions & Grid:**
- `number`: 15
- `step`: 44
- `latitude`: 34
- `longitude`: 34

**Variables & Units:**
- `mx2t6`: Maximum temperature at 2 metres in the last 6 hours (K)
- `mn2t6`: Minimum temperature at 2 metres in the last 6 hours (K)

**Key Coordinates Details:**
- **Lead Times (step)**: 44 steps, Max: 3801600000000000 nanoseconds
- **Ensemble Members (number)**: 15 members.

---
#### GRIB Message Group 2
 NCEP Group 2

**Dimensions & Grid:**
- `number`: 15
- `step`: 44
- `latitude`: 34
- `longitude`: 34

**Variables & Units:**
- `tp`: Total Precipitation (kg m**-2)

**Key Coordinates Details:**
- **Lead Times (step)**: 44 steps, Max: 3801600000000000 nanoseconds
- **Ensemble Members (number)**: 15 members.

---
