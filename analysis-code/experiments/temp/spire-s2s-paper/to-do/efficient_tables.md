# Efficient Tables for S2S Paper: Main Text vs. Supplementary Material

To present a massive S2S dataset efficiently in a journal article, follow the two-part strategy below.

---

## Part 1: Main Text (Condensed Table)
This table groups the 50+ variables into **8 physical parameter categories**, making it compact, clean, and easily fitting into a single page in your main manuscript draft. (Note: The dimensions column is dropped as they are identical across fields, and variables are grouped using clean brace notation).

### Markdown Version (Main Text)

| Parameter Category | Primary Zarr Fields | Units | Ensemble Derivatives |
| :--- | :--- | :---: | :--- |
| **2m Air Temperature** | `air_temperature`<br>`air_temperature_max`<br>`air_temperature_min` | K<br>K<br>K | Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs |
| **Wind Fields (10m & 100m)** | `eastward_wind`<br>`northward_wind`<br>`wind_speed`<br>`eastward_wind_100m`<br>`northward_wind_100m`<br>`wind_speed_100m` | m s⁻¹<br>m s⁻¹<br>m s⁻¹<br>m s⁻¹<br>m s⁻¹<br>m s⁻¹ | Mean, Std Dev, 9 Percentiles, 7 Probs (for all 6 fields)<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs |
| **Precipitation** | `precipitation_amount` | kg m⁻² | Mean, Std Dev, 9 Percentiles, 7 Probs |
| **Upper-Air Variables (Pressure Levels)** | `air_temperature_at_isobaric_levels`<br>`eastward_wind_at_isobaric_levels`<br>`northward_wind_at_isobaric_levels`<br>`specific_humidity_at_isobaric_levels`<br>`geopotential_height_at_isobaric_levels`<br>`wind_speed_at_isobaric_levels` | K<br>m s⁻¹<br>m s⁻¹<br>kg kg⁻¹<br>m<br>m s⁻¹ | Mean, Std Dev (at 1000, 850, 700, 500 hPa)<br>Mean, Std Dev (at 1000, 850, 700, 500 hPa)<br>Mean, Std Dev (at 1000, 850, 700, 500 hPa)<br>Mean, Std Dev (at 1000, 850, 700, 500 hPa)<br>Mean, Std Dev (at 1000, 850, 700, 500 hPa)<br>Mean, Std Dev (at 1000, 850, 700, 500 hPa) |
| **Pressure Fields** | `air_pressure_at_sea_level`<br>`sp`<br>`msl` | Pa<br>Pa<br>Pa | Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs |
| **Geopotential Thickness** | `geopotential_thickness` | m | Mean, Std Dev, 9 Percentiles |
| **Radiation & Surface Fluxes** | `surface_downwelling_shortwave_flux`<br>`ttr`<br>`slhf`<br>`ssr`<br>`str`<br>`sshf`<br>`ssrd`<br>`strd` | W m⁻²<br>W m⁻² s<br>W m⁻² s<br>W m⁻² s<br>W m⁻² s<br>W m⁻² s<br>W m⁻² s<br>W m⁻² s | Mean, Std Dev, 9 Percentiles, 7 Probs (for all 8 fields)<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs<br>Mean, Std Dev, 9 Percentiles, 7 Probs |
| **Weather Regimes** | `conus`<br>`euatl` | 1 (Prob)<br>1 (Prob) | Regime Probabilities (5 weather regimes)<br>Regime Probabilities (8 weather regimes) |

### LaTeX Code (Main Text Table)
Copy this standard table directly into your main LaTeX manuscript file:

```latex
\begin{table*}[t]
\centering
\caption{Summary of physical parameter categories, units, and ensemble derivatives in the Spire AI-S2S hindcast dataset.}
\label{tab:s2s_summary}
\begin{tabular}{lllc}
\toprule
\textbf{Parameter Category} & \textbf{Primary Variable Fields} & \textbf{Units} & \textbf{Ensemble Derivatives} \\
\midrule
2m Air Temperature & 
\begin{tabular}[t]{@{}l@{}} air\_temperature \\ air\_temperature\_max \\ air\_temperature\_min \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} K \\ K \\ K \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \end{tabular} \\
\addlinespace
Wind Fields (10m \& 100m) & 
\begin{tabular}[t]{@{}l@{}} eastward\_wind \\ northward\_wind \\ wind\_speed \\ eastward\_wind\_100m \\ northward\_wind\_100m \\ wind\_speed\_100m \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} m s$^{-1}$ \\ m s$^{-1}$ \\ m s$^{-1}$ \\ m s$^{-1}$ \\ m s$^{-1}$ \\ m s$^{-1}$ \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \end{tabular} \\
\addlinespace
Precipitation & 
precipitation\_amount & 
kg m$^{-2}$ & 
Mean, Std Dev, 9 Percentiles, 7 Probs \\
\addlinespace
Upper-Air Variables & 
\begin{tabular}[t]{@{}l@{}} air\_temperature\_at\_isobaric\_levels \\ eastward\_wind\_at\_isobaric\_levels \\ northward\_wind\_at\_isobaric\_levels \\ specific\_humidity\_at\_isobaric\_levels \\ geopotential\_height\_at\_isobaric\_levels \\ wind\_speed\_at\_isobaric\_levels \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} K \\ m s$^{-1}$ \\ m s$^{-1}$ \\ kg kg$^{-1}$ \\ m \\ m s$^{-1}$ \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Mean, Std Dev (at 1000, 850, 700, 500~hPa) \\ Mean, Std Dev (at 1000, 850, 700, 500~hPa) \\ Mean, Std Dev (at 1000, 850, 700, 500~hPa) \\ Mean, Std Dev (at 1000, 850, 700, 500~hPa) \\ Mean, Std Dev (at 1000, 850, 700, 500~hPa) \\ Mean, Std Dev (at 1000, 850, 700, 500~hPa) \end{tabular} \\
\addlinespace
Pressure Fields & 
\begin{tabular}[t]{@{}l@{}} air\_pressure\_at\_sea\_level \\ sp \\ msl \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Pa \\ Pa \\ Pa \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \end{tabular} \\
\addlinespace
Geopotential Thickness & 
geopotential\_thickness & 
m & 
Mean, Std Dev, 9 Percentiles \\
\addlinespace
Radiation \& Fluxes & 
\begin{tabular}[t]{@{}l@{}} surface\_downwelling\_shortwave\_flux \\ ttr \\ slhf \\ ssr \\ str \\ sshf \\ ssrd \\ strd \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} W m$^{-2}$ \\ W m$^{-2}$ s \\ W m$^{-2}$ s \\ W m$^{-2}$ s \\ W m$^{-2}$ s \\ W m$^{-2}$ s \\ W m$^{-2}$ s \\ W m$^{-2}$ s \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \\ Mean, Std Dev, 9 Percentiles, 7 Probs \end{tabular} \\
\addlinespace
Weather Regimes & 
\begin{tabular}[t]{@{}l@{}} conus \\ euatl \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} 1 (Prob) \\ 1 (Prob) \end{tabular} & 
\begin{tabular}[t]{@{}l@{}} Regime Probabilities (5 weather regimes) \\ Regime Probabilities (8 weather regimes) \end{tabular} \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## Part 2: Supplementary Material (SI / Appendix)
Place the **full 50+ row table** in the **Supplementary Information (SI)** or **Appendix**.

> [!NOTE]
> The full table code using the LaTeX `longtable` environment is already prepared and saved in the workspace at **[arraylake_metadata.md](file:///home/raj.ayush/s2s/s2s_anlysis/arraylake_metadata.md)**.
