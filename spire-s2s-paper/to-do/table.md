# Table 6: List of Single-Level Parameters

Here is the exact transcription of the parameters table from `/home/raj.ayush/s2s/s2s_anlysis/spire-s2s-paper/to-do/image.png` in both Markdown and LaTeX formats.

---

## 1. Markdown Table

| Name | Abbreviation | Unit | Frequency |
| :--- | :---: | :---: | :--- |
| 10-m u | 10u | m s⁻¹ | Instantaneous once per day (0000 UTC) |
| 10-m v | 10v | m s⁻¹ | Instantaneous once per day (0000 UTC) |
| CAPE | cape | J kg⁻¹ | Daily average |
| Skin temperature | skt | K | Daily average |
| Snow depth water equivalent | sd | kg m⁻² | Daily average |
| Snow density | rsn | kg m⁻³ | Daily average |
| Snowfall water equivalent | sf | kg m⁻² | Accumulated once per day |
| Snow albedo | asn | % | Daily average |
| Soil moisture top 20 cm | sm20 | kg m⁻³ | Daily average |
| Soil moisture top 100 cm | sm100 | kg m⁻³ | Daily average |
| Soil temperature to 20 cm | st20 | K | Daily average |
| Soil temperature top 100 cm | st100 | K | Daily average |
| Surface air max temperature | mx2t6 | K | Instantaneous four times per day |
| Surface air min temperature | mn2t6 | K | Instantaneous four times per day |
| Surface air temperature | 2t | K | Daily average |
| Surface air dewpoint temperature | 2d | K | Daily average |
| Sea surface temperature | wtmp | K | Daily average |
| Sea ice cover | ci | Proportion | Daily average |
| Surface pressure | sp | Pa | Instantaneous once per day (0000 UTC) |
| Mean sea level pressure | msl | Pa | Instantaneous once per day (0000 UTC) |
| Total cloud cover | tcc | % | Daily average |
| Total column water | tcw | kg m⁻² | Daily average |
| Total precipitation | tp | kg m⁻² | Accumulated four times per day |
| Convective precipitation | cp | kg m⁻² | Accumulated once per day |
| Northward turbulent surface stress | nsss | N m⁻² s | Accumulated once per day |
| Eastward turbulent surface stress | ewss | N m⁻² s | Accumulated once per day |
| Water runoff and drainage | ro | kg m⁻² | Accumulated once per day |
| Surface water runoff | sro | kg m⁻² | Accumulated once per day |
| Land-sea mask | lsm | Proportion of land | Instantaneous once per day (0000 UTC) |
| Orography | orog | gpm | Instantaneous once per day (0000 UTC) |
| Soil type | slt | Categorical | Instantaneous once per day (0000 UTC) |
| Top net thermal radiation | ttr | W m⁻² s | Accumulated once per day |
| Surface latent heat flux | slhf | W m⁻² s | Accumulated once per day |
| Surface net solar radiation | ssr | W m⁻² s | Accumulated once per day |
| Surface net thermal radiation | str | W m⁻² s | Accumulated once per day |
| Surface sensible heat flux | sshf | W m⁻² s | Accumulated once per day |
| Solar radiation downward | ssrd | W m⁻² s | Accumulated once per day |
| Surface thermal radiation downward | strd | W m⁻² s | Accumulated once per day |

---

## 2. LaTeX Table Code (`booktabs` style)

You can copy and paste this code directly into your LaTeX manuscript:

```latex
\begin{table*}[t]
\centering
\caption{List of single-level parameters.}
\label{tab:single_level_params}
\begin{tabular}{llcl}
\toprule
\textbf{Name} & \textbf{Abbreviation} & \textbf{Unit} & \textbf{Frequency} \\
\midrule
10-m u & 10u & m s$^{-1}$ & Instantaneous once per day (0000 UTC) \\
10-m v & 10v & m s$^{-1}$ & Instantaneous once per day (0000 UTC) \\
CAPE & cape & J kg$^{-1}$ & Daily average \\
Skin temperature & skt & K & Daily average \\
Snow depth water equivalent & sd & kg m$^{-2}$ & Daily average \\
Snow density & rsn & kg m$^{-3}$ & Daily average \\
Snowfall water equivalent & sf & kg m$^{-2}$ & Accumulated once per day \\
Snow albedo & asn & \% & Daily average \\
Soil moisture top 20 cm & sm20 & kg m$^{-3}$ & Daily average \\
Soil moisture top 100 cm & sm100 & kg m$^{-3}$ & Daily average \\
Soil temperature to 20 cm & st20 & K & Daily average \\
Soil temperature top 100 cm & st100 & K & Daily average \\
Surface air max temperature & mx2t6 & K & Instantaneous four times per day \\
Surface air min temperature & mn2t6 & K & Instantaneous four times per day \\
Surface air temperature & 2t & K & Daily average \\
Surface air dewpoint temperature & 2d & K & Daily average \\
Sea surface temperature & wtmp & K & Daily average \\
Sea ice cover & ci & Proportion & Daily average \\
Surface pressure & sp & Pa & Instantaneous once per day (0000 UTC) \\
Mean sea level pressure & msl & Pa & Instantaneous once per day (0000 UTC) \\
Total cloud cover & tcc & \% & Daily average \\
Total column water & tcw & kg m$^{-2}$ & Daily average \\
Total precipitation & tp & kg m$^{-2}$ & Accumulated four times per day \\
Convective precipitation & cp & kg m$^{-2}$ & Accumulated once per day \\
Northward turbulent surface stress & nsss & N m$^{-2}$ s & Accumulated once per day \\
Eastward turbulent surface stress & ewss & N m$^{-2}$ s & Accumulated once per day \\
Water runoff and drainage & ro & kg m$^{-2}$ & Accumulated once per day \\
Surface water runoff & sro & kg m$^{-2}$ & Accumulated once per day \\
Land-sea mask & lsm & Proportion of land & Instantaneous once per day (0000 UTC) \\
Orography & orog & gpm & Instantaneous once per day (0000 UTC) \\
Soil type & slt & Categorical & Instantaneous once per day (0000 UTC) \\
Top net thermal radiation & ttr & W m$^{-2}$ s & Accumulated once per day \\
Surface latent heat flux & slhf & W m$^{-2}$ s & Accumulated once per day \\
Surface net solar radiation & ssr & W m$^{-2}$ s & Accumulated once per day \\
Surface net thermal radiation & str & W m$^{-2}$ s & Accumulated once per day \\
Surface sensible heat flux & sshf & W m$^{-2}$ s & Accumulated once per day \\
Solar radiation downward & ssrd & W m$^{-2}$ s & Accumulated once per day \\
Surface thermal radiation downward & strd & W m$^{-2}$ s & Accumulated once per day \\
\bottomrule
\end{tabular}
\end{table*}
```
