# 00 - Data audit

- Source file: `C:\Users\ASUS\Desktop\mxene-geometry\terra\h2so4_corpus_final.csv`
- Rows: **192**
- Columns: **83**
- Unique source papers (`paper_id`): **49**
- Fully duplicated rows: **0**
- Duplicated rows ignoring `__conf`/`__src` metadata: **0**

## Rows per paper

- min 1, median 3, mean 3.92, max 15

| paper_id | n_rows |
| --- | ---: |
| 10.1039/d6ta00010j | 15 |
| 10.1016/j.ensm.2023.103148 | 9 |
| 10.1016/j.jallcom.2022.166647 | 9 |
| 10.1021/acsnano.9b07325 | 9 |
| 10.1002/advs.202305991 | 8 |
| 10.1039/c8dt04374d | 8 |
| 10.1002/slct.202200690 | 8 |
| 10.1038/s41586-018-0109-z | 8 |
| 10.1038/nenergy.2017.105 | 8 |
| 10.1002/eem2.12876 | 5 |
| 10.20517/energymater.2024.45 | 5 |
| 10.1016/j.jallcom.2023.173355 | 5 |
| 10.1002/adfm.202102874 | 5 |
| 10.1002/adfm.201705506 | 5 |
| 10.1002/aelm.201800179 | 4 |
| 10.1002/adma.201701264 | 4 |
| 10.1002/adma.202304757 | 4 |
| 10.1016/j.jallcom.2021.161304 | 4 |
| 10.1016/j.electacta.2022.139871 | 4 |
| 10.1002/smll.201804732 | 4 |
| 10.1002/celc.202100558 | 4 |
| 10.3390/molecules30020241 | 4 |
| 10.1016/j.apmt.2019.06.013 | 3 |
| 10.1016/j.carbpol.2022.120519 | 3 |
| 10.1016/j.carbon.2025.120021 | 3 |
| ssrn-5211970.pdf | 3 |
| 10.1002/smll.202205947 | 3 |
| 10.1038/nature13970 | 3 |
| 10.1002/smll.201802225 | 3 |
| 10.1002/adma.201504705 | 3 |
| 10.1039/d5ta06789h | 3 |
| 10.1002/aelm.201700339 | 3 |
| 10.1016/j.ceramint.2020.05.247 | 2 |
| 10.25259/ajc_264_2024 | 2 |
| 10.1039/c8nr06014b | 2 |
| 10.1007/s40820-024-01567-2 | 2 |
| 10.1021/acsnano.9b10066 | 2 |
| 10.1002/smtd.202500499 | 2 |
| 10.1016/j.jobab.2021.10.001 | 1 |
| 10.1016/j.apsusc.2019.144250 | 1 |
| 10.1002/batt.201800014 | 1 |
| 10.1021/acsnano.2c03351 | 1 |
| 10.1002/smll.202202203 | 1 |
| 10.1002/aenm.201601372 | 1 |
| 10.1021/acsanm.4c01912 | 1 |
| 10.1039/c8nr01550c | 1 |
| 10.1021/acsnano.3c11551 | 1 |
| 10.1002/smsc.202500367 | 1 |
| 10.1002/sus2.61 | 1 |

## Target availability

Raw `gravimetric_capacitance` non-null: 122 / 192

| literal kind | n |
| --- | ---: |
| exact | 116 |
| missing | 70 |
| inequality | 4 |
| approximate | 2 |

**Usable single-valued capacitance rows: 118**

## Column summary

| column | dtype | non-null | missing % | unique | example |
| --- | --- | ---: | ---: | ---: | --- |
| source_file | object | 192 | 0.0 | 49 | 1-s2.0-S0008622325000375-main.pdf |
| source_path | object | 192 | 0.0 | 49 | 1-s2.0-S0008622325000375-main.pdf |
| doi | object | 189 | 1.56 | 48 | 10.1016/j.carbon.2025.120021 |
| year | float64 | 189 | 1.56 | 12 | 2025.0 |
| authors | object | 192 | 0.0 | 46 | Hailong Shen et al. |
| electrode_label | object | 192 | 0.0 | 182 | NS-Ti3C2Tx-P |
| role | object | 192 | 0.0 | 4 | primary |
| is_primary | bool | 192 | 0.0 | 2 | True |
| mxene_formula | object | 192 | 0.0 | 2 | Ti3C2Tx |
| mxene_formula__unit | float64 | 0 | 100.0 | 0 |  |
| mxene_formula__conf | object | 192 | 0.0 | 2 | stated |
| composition | object | 192 | 0.0 | 80 | N/S co-doped Ti3C2Tx (Ti, C, O, F, N, S) |
| composition__unit | object | 3 | 98.44 | 1 | wt. % |
| composition__conf | object | 192 | 0.0 | 2 | stated |
| synthesis_method | object | 192 | 0.0 | 122 | LiF/HCl etching; photothermal-assisted thiourea N/S co-doping under high-pressur |
| synthesis_method__unit | float64 | 0 | 100.0 | 0 |  |
| synthesis_method__conf | object | 192 | 0.0 | 2 | stated |
| layer_no | object | 160 | 16.67 | 32 | few-layer |
| layer_no__unit | object | 2 | 98.96 | 1 | - |
| layer_no__conf | object | 192 | 0.0 | 3 | derived |
| layer_no__src | object | 3 | 98.44 | 3 | [UNCONFIRMED] Section 2.2: the MXene used in all films is the delaminated supern |
| interlayer_spacing | object | 126 | 34.38 | 80 | 1.42 |
| interlayer_spacing__unit | object | 125 | 34.9 | 3 | nm |
| interlayer_spacing__conf | object | 192 | 0.0 | 4 | stated |
| interlayer_spacing__src | object | 59 | 69.27 | 57 | [UNVERIFIED] Bragg from the text-stated (002) position: 'the peak of (002) plane |
| flake_size | object | 99 | 48.44 | 30 | 200–1000 |
| flake_size__unit | object | 85 | 55.73 | 5 | nm |
| flake_size__conf | object | 192 | 0.0 | 4 | stated |
| flake_size__src | object | 8 | 95.83 | 6 | [UNVERIFIED] 'For our system (e.g., 1 μm flake size, 1 nm flake thickness, and 3 |
| porosity | object | 11 | 94.27 | 8 | ~99.8 |
| porosity__unit | object | 11 | 94.27 | 1 | % |
| porosity__conf | object | 192 | 0.0 | 3 | stated |
| porosity__src | object | 3 | 98.44 | 3 | [UNVERIFIED] The 90 mg mL−1 device is the same standard backbone as the 180 mg m |
| pore_diameter | object | 38 | 80.21 | 16 | 1–3 |
| pore_diameter__unit | object | 38 | 80.21 | 2 | nm |
| pore_diameter__conf | object | 192 | 0.0 | 3 | stated |
| pore_diameter__src | object | 9 | 95.31 | 5 | Text: 'pore sizes of films based on porous Ti3C2Tx nanosheets (Fig. 5b) mainly d |
| tortuosity | float64 | 0 | 100.0 | 0 |  |
| tortuosity__unit | float64 | 0 | 100.0 | 0 |  |
| tortuosity__conf | object | 192 | 0.0 | 2 | stated |
| electrode_thickness | object | 85 | 55.73 | 65 | 7.00 |
| electrode_thickness__unit | object | 85 | 55.73 | 5 | μm |
| electrode_thickness__conf | object | 192 | 0.0 | 4 | derived |
| electrode_thickness__src | object | 11 | 94.27 | 11 | [UNVERIFIED] Back-calculated from the paper's own numbers: C_v = C_g x density a |
| mass_loading | object | 80 | 58.33 | 53 | 1.69 |
| mass_loading__unit | object | 81 | 57.81 | 6 | mg cm−2 |
| mass_loading__conf | object | 192 | 0.0 | 4 | stated |
| mass_loading__src | object | 43 | 77.6 | 34 | [UNVERIFIED] Stated: 'the mass of the Ti3C2Tx film was approximately 21 mg. The  |
| electrolyte | object | 192 | 0.0 | 21 | 2 M H2SO4 |
| electrolyte__unit | float64 | 0 | 100.0 | 0 |  |
| electrolyte__conf | object | 192 | 0.0 | 3 | derived |
| ssa | object | 37 | 80.73 | 37 | 2.95 |
| ssa__unit | object | 37 | 80.73 | 3 | m2 g−1 |
| ssa__conf | object | 192 | 0.0 | 3 | stated |
| ssa__src | object | 2 | 98.96 | 2 | Text states the series range 'specific surface area (SSA) (8.4-65.4 m2 g-1)' and |
| scan_rate | object | 121 | 36.98 | 11 | 2 |
| scan_rate__unit | object | 121 | 36.98 | 4 | mV s−1 |
| scan_rate__conf | object | 192 | 0.0 | 4 | stated |
| scan_rate__src | object | 46 | 76.04 | 36 | Methods (Section 2.5): "Cyclic voltammetry (CV) tests were recorded from −0.3–0. |
| current_density | float64 | 60 | 68.75 | 7 | 1.0 |
| current_density__unit | object | 60 | 68.75 | 8 | A g−1 |
| current_density__conf | object | 192 | 0.0 | 4 | stated |
| current_density__src | object | 19 | 90.1 | 14 | [UNVERIFIED] Fig. 4 caption: '(c) GCD of MP0, MP2, MP5 and MP8 at current densit |
| gravimetric_capacitance | object | 122 | 36.46 | 111 | 424 |
| gravimetric_capacitance__unit | object | 122 | 36.46 | 3 | F g−1 |
| gravimetric_capacitance__conf | object | 192 | 0.0 | 4 | stated |
| gravimetric_capacitance__src | object | 41 | 78.65 | 38 | [UNVERIFIED] Fig. 4(c), GCD of all four films at 1.0 A g-1 (three-electrode). At |
| volumetric_capacitance | object | 95 | 50.52 | 91 | 1023 |
| volumetric_capacitance__unit | object | 95 | 50.52 | 3 | F cm−3 |
| volumetric_capacitance__conf | object | 192 | 0.0 | 4 | stated |
| volumetric_capacitance__src | object | 48 | 75.0 | 48 | [UNVERIFIED] C_v = C_g x density. density = mass loading / thickness = 1.35e-3 g |
| areal_capacitance | object | 75 | 60.94 | 69 | 0.717 |
| areal_capacitance__unit | object | 75 | 60.94 | 5 | F cm−2 |
| areal_capacitance__conf | object | 192 | 0.0 | 4 | derived |
| areal_capacitance__src | object | 41 | 78.65 | 41 | [UNVERIFIED] C_areal = C_g x mass loading = 424 F g-1 x 1.69e-3 g cm-2 = 0.7166  |
| formula_match | bool | 192 | 0.0 | 1 | True |
| formula_reasoning | object | 192 | 0.0 | 2 | Primary formula 'Ti3C2Tx' contains Ti3C2 (Ti3C2Tx or Ti3C2). |
| synthesis_match | bool | 192 | 0.0 | 1 | True |
| synthesis_reasoning | object | 192 | 0.0 | 1 | Synthesis explicitly mentions both LiF and HCl. |
| electrolyte_match | bool | 192 | 0.0 | 1 | True |
| electrolyte_reasoning | object | 192 | 0.0 | 1 | Electrolyte explicitly mentions H2SO4 (sulfuric acid). |
| overall | object | 192 | 0.0 | 1 | PASS |
| overall_reasoning | object | 192 | 0.0 | 1 | Passes all criteria. |

## Unit inconsistencies (normalised unit -> count)

Different spellings of the same unit (`F g-1`, `F/g`) are collapsed; what
remains are *genuinely different physical bases* that require care.

- `composition__unit`: {'wt%': 3}
- `layer_no__unit`: {'-': 2}
- `interlayer_spacing__unit`: {'nm': 38, 'a': 87} **<- MIXED BASES**
- `flake_size__unit`: {'nm': 18, 'nm-um': 1, 'um': 66} **<- MIXED BASES**
- `porosity__unit`: {'%': 11}
- `pore_diameter__unit`: {'nm': 19, 'um': 19} **<- MIXED BASES**
- `electrode_thickness__unit`: {'um': 71, 'mm': 6, 'nm': 8} **<- MIXED BASES**
- `mass_loading__unit`: {'mg/cm2': 72, 'ug/cm2': 3, 'mg': 3, 'wt%': 3} **<- MIXED BASES**
- `ssa__unit`: {'m2/g': 37}
- `scan_rate__unit`: {'mv/s': 119, 'v/s': 2} **<- MIXED BASES**
- `current_density__unit`: {'a/g': 41, 'ma/cm2': 10, 'a/cm3': 5, 'ua': 2, 'ua/cm2': 2} **<- MIXED BASES**
- `gravimetric_capacitance__unit`: {'f/g': 122}
- `volumetric_capacitance__unit`: {'f/cm3': 95}
- `areal_capacitance__unit`: {'f/cm2': 36, 'mf/cm2': 39} **<- MIXED BASES**

## Categorical / low-cardinality columns

### `role` (4 shown)

- `variant`: 84
- `control`: 42
- `primary`: 41
- `composite`: 25

### `mxene_formula` (2 shown)

- `Ti3C2Tx`: 188
- `N-Ti3C2`: 4

### `mxene_formula__conf` (2 shown)

- `stated`: 185
- `derived`: 7

### `composition__unit` (1 shown)

- `wt. %`: 3

### `composition__conf` (2 shown)

- `stated`: 183
- `derived`: 9

### `synthesis_method__conf` (2 shown)

- `stated`: 139
- `derived`: 53

### `layer_no__unit` (1 shown)

- `-`: 2

### `layer_no__conf` (3 shown)

- `derived`: 103
- `stated`: 80
- `uncertain`: 9

### `layer_no__src` (3 shown)

- `[UNCONFIRMED] Section 2.2: the MXene used in all films is the delaminated supernatant from LiF/HCl etching ("the supernatant and precipitate were ultrasonicated for 15 min, then centrifuged the mixture at 3000 rpm for 1 h and collected the required upper dispersion... The resulting super liquid was the solution of MXene"), and Section 2.3 states this same MXene dispersion (40 mg) is used for the composite films. Raman (Fig. 2b) is assigned to "Ti-C bond vibrations within the exfoliated Ti3C2Tx... confirms the successful etching and exfoliation of Al" for all films. Same descriptor as the already-filled MXene control row; no layer count is ever quoted.`: 1
- `[UNCONFIRMED] Same delaminated (exfoliated) MXene dispersion is used for this film - Section 2.3: "A total of 40 mg of the prepared MXene was then added to the dispersion"; Section 2.2 describes collection of the delaminated upper dispersion. Raman discussion refers to "exfoliated Ti3C2Tx" in all films. No numeric layer count is reported anywhere.`: 1
- `[UNVERIFIED] 'After etching and sonicating, the obtained Ti3C2Tx presents a few-layer or single-layer nanosheet, which stacks to form a wrinkled morphology'; corroborated by 'there is no characteristic peak at 2θ = 20° related to the (004) crystal plane, owing to the prepared MXene is single or few layers'. Describes the Ti3C2Tx component used to build the CPCM/MXene composite.`: 1

### `interlayer_spacing__unit` (3 shown)

- `Å`: 68
- `nm`: 38
- `A`: 19

### `interlayer_spacing__conf` (4 shown)

- `stated`: 88
- `derived`: 50
- `figure`: 31
- `uncertain`: 23

### `flake_size` (30 shown)

- `less than 1`: 15
- `micrometer-sized`: 9
- `∼0.3`: 9
- `7.6–9.5`: 8
- `219 ± 47`: 8
- `several`: 8
- `1.2 ± 0.2`: 5
- `~1.3`: 4
- `≈1`: 4
- `0.5-1.5`: 3
- `2-3`: 3
- `few-layer`: 3
- `approximately 1`: 2
- `less than 400`: 2
- `approximately 400`: 1
- `approximately 600`: 1
- `approximately 1.3`: 1
- `300 to 500`: 1
- `up to several micrometers`: 1
- `200–1000`: 1
- `0.116±0.047`: 1
- `MXene: 0.116±0.047; RGO: 1.74±0.68`: 1
- `500 nm to 1.5 µm`: 1
- `570`: 1
- `about 250`: 1
- `single-layer flakes`: 1
- `240`: 1
- `280`: 1
- `1–3`: 1
- `<1`: 1

### `flake_size__unit` (5 shown)

- `μm`: 43
- `µm`: 19
- `nm`: 18
- `microns`: 4
- `nm–µm`: 1

### `flake_size__conf` (4 shown)

- `stated`: 86
- `derived`: 76
- `uncertain`: 29
- `figure`: 1

### `flake_size__src` (6 shown)

- `[UNVERIFIED] p.2: "of 321 flakes analysed, over 70% had dimensions of 0.5-1.5 μm (Extended Data Fig. 4a, b)". Material-level TEM lateral-size distribution for the Ti3C2Tx used in all three rolled clay electrodes.`: 2
- `[UNVERIFIED] 'For our system (e.g., 1 μm flake size, 1 nm flake thickness, and 3.7 g cm-3 Ti3C2Tx MXene density)...'; corroborated by the Figure 1d DLS distribution peaking at ≈900 nm. Material-level value (same MXene dispersion for all MX-PS samples).`: 2
- `[UNVERIFIED] 'For our system (e.g., 1 μm flake size, 1 nm flake thickness, and 3.7 g cm-3 Ti3C2Tx MXene density)...'; corroborated by the Figure 1d DLS distribution peaking at ≈900 nm. Material-level value (same MXene dispersion, only its concentration differs between samples).`: 1
- `[UNVERIFIED] 'For our system (e.g., 1 μm flake size, 1 nm flake thickness, and 3.7 g cm-3 Ti3C2Tx MXene density), ref. [51] gives the critical MXene concentration...'. Corroborated by Figure 1d (DLS hydrodynamic-diameter distribution of the as-synthesized Ti3C2Tx), whose main peak sits at ≈900 nm. Material-level: one MXene dispersion is used for every MX-PS sample in the paper.`: 1
- `[UNCONFIRMED] Figure 2c (SEM of TCS-2, page 4) carries the printed in-panel annotation 'd ~ 400 nm' (verified at 900 dpi zoom, digits unambiguous). Consistent with the text statement that size decreases from ~600 nm (TCS-1, 1 h) to ~250 nm (TCS-3, 3 h), with TCS-2 (2 h) in between.`: 1
- `[UNVERIFIED] p.2: "Transmission electron microscopy (TEM) analysis showed that, of 321 flakes analysed, over 70% had dimensions of 0.5-1.5 μm (Extended Data Fig. 4a, b)." Material-level lateral-size distribution of the LiF+HCl-etched Ti3C2Tx used for all three rolled clay electrodes (not a per-electrode measurement).`: 1

### `porosity` (8 shown)

- `60`: 3
- `26.85`: 2
- `40`: 1
- `~99.8`: 1
- `59`: 1
- `66`: 1
- `72`: 1
- `80`: 1

### `porosity__unit` (1 shown)

- `%`: 11

### `porosity__conf` (3 shown)

- `stated`: 137
- `uncertain`: 52
- `derived`: 3

### `porosity__src` (3 shown)

- `[UNVERIFIED] The 90 mg mL−1 device is the same standard backbone as the 180 mg mL−1 primary device, differing only in dispersion concentration: 'the effect of MXene concentration on the electrochemical performance is briefly evaluated by comparing these results with the ones of a supercapacitor prepared with a reduced MXene concentration of 90 mg mL-1'. The standard backbone is described as 'The resulting porous silica backbones exhibit ≈60% porosity with ≈50 μm average pore size'. [+-5%]`: 1
- `[UNVERIFIED] Authors' own convention: 'The lowest achievable density was 1.15 g/cm3, which corresponds to ~80% porosity (considering the theoretical density of the -OH terminated Ti3C2; 5.2 g/cm3)'. Applying 1 - rho/5.2 to this film: 1 - 2.11/5.2 = 0.594 = 59.4%. The same formula reproduces the paper's stated 72% for the 1.44 g/cm3 film (1 - 1.44/5.2 = 72.3%). [+-4%]`: 1
- `[UNVERIFIED] 1 - 1.78/5.2 = 0.658 = 65.8%, using the authors' theoretical density of 5.2 g/cm3 for -OH terminated Ti3C2 (same formula that reproduces their stated 72% for 1.44 g/cm3 and ~80% for 1.15 g/cm3). [+-4%]`: 1

### `pore_diameter` (16 shown)

- `larger than 2`: 9
- `3-5`: 4
- `10–70`: 4
- `3–5`: 4
- `1–2`: 4
- `≈50`: 3
- `1–3`: 1
- `5–20`: 1
- `≈20`: 1
- `approximately 1–2`: 1
- `2–20`: 1
- `3`: 1
- `4.81`: 1
- `∼3.9`: 1
- `2–10`: 1
- `5–10`: 1

### `pore_diameter__unit` (2 shown)

- `nm`: 19
- `μm`: 19

### `pore_diameter__conf` (3 shown)

- `stated`: 126
- `uncertain`: 52
- `derived`: 14

### `pore_diameter__src` (5 shown)

- `Text: series-level range 3-5 nm (Fig. 5b)`: 3
- `[UNVERIFIED] '... a mesoporous structure with pore sizes mainly ranging from 10-70 nm (Figure 3i-inset).' SERIES-LEVEL: quoted once for the whole BET set of four films, not per-sample.`: 3
- `Text: 'pore sizes of films based on porous Ti3C2Tx nanosheets (Fig. 5b) mainly distribute in range of 3-5 nm' (series-level, not per-sample)`: 1
- `[UNVERIFIED] Same reasoning as the porosity entry: the 90 mg mL−1 device uses the standard freeze-cast backbone, 'The resulting porous silica backbones exhibit ≈60% porosity with ≈50 μm average pore size'. Not a per-sample measurement. [+-5%]`: 1
- `[UNVERIFIED] 'The hysteresis loops in type IV physisorption isotherms (Figure 3i) suggest a mesoporous structure with pore sizes mainly ranging from 10-70 nm (Figure 3i-inset).' SERIES-LEVEL: one range quoted for the BET set covering all four films (Figure 3i legend lists 95.6/62.1/24.3/10.5 m2/g), not a per-sample value.`: 1

### `tortuosity__conf` (2 shown)

- `stated`: 138
- `uncertain`: 54

### `electrode_thickness__unit` (5 shown)

- `μm`: 39
- `µm`: 29
- `nm`: 8
- `mm`: 6
- `um`: 3

### `electrode_thickness__conf` (4 shown)

- `stated`: 136
- `uncertain`: 34
- `derived`: 15
- `figure`: 7

### `electrode_thickness__src` (11 shown)

- `[UNVERIFIED] Back-calculated from the paper's own numbers: C_v = C_g x density and density = mass_loading / thickness. Stated: 'The mass load of NS-Ti3C2Tx-P is 1.69 mg cm-2'; 'the specific capacitance of NS-Ti3C2Tx-P can reach 424 F g-1 (1023 F cm-3) at 2 mV s-1'. density = 1023/424 = 2.413 g cm-3; t = 1.69e-3 g cm-2 / 2.413 g cm-3 = 7.00e-4 cm = 7.00 um. The same arithmetic on the two sister samples returns 8.00 and 5.02 um, i.e. the authors used round micrometre thicknesses, which confirms the chain. [+-3%]`: 1
- `[UNVERIFIED] Same chain as NS-Ti3C2Tx-P. Stated: mass load 2.12 mg cm-2; 'the mass capacitance of NS-Ti3C2Tx-A and Ti3C2Tx are only 358 F g-1 and 330 F g-1, and the volume capacitance are 949 F cm-3 and 789 F cm-3' (2 mV s-1). density = 949/358 = 2.651 g cm-3; t = 2.12e-3/2.651 = 8.00e-4 cm = 8.00 um. [+-3%]`: 1
- `[UNVERIFIED] Same chain. Stated: mass load 1.20 mg cm-2; 330 F g-1 / 789 F cm-3 at 2 mV s-1. density = 789/330 = 2.391 g cm-3; t = 1.20e-3/2.391 = 5.02e-4 cm = 5.02 um. [+-3%]`: 1
- `[UNVERIFIED] Fig. 2(g) (cross-sectional SEM of MP0) carries the same style of printed full-film arrow, but its annotation reads '19.__ μm' – the two decimal digits are washed out by a bright SEM feature and are not recoverable at the raster resolution of the embedded image. Only the integer part '19' is legible, so the value is bracketed 19.0–20.0 μm. Measuring the arrow against the panel's own 10 μm bar gives ~18.6–20.0 μm, consistent. [+-3%]`: 1
- `[UNVERIFIED] Fig. 2(h) (cross-sectional SEM of MP2) carries a printed yellow double-headed arrow spanning the whole film with the annotation '19.88 μm'. Read directly from the annotation, not measured. [+-2%]`: 1
- `[UNVERIFIED] Fig. 2(i) (cross-sectional SEM of MP5): printed yellow full-film arrow annotated '30.76 μm' (last digit could conceivably be 0 rather than 6; immaterial at 0.2%). Independent check by measuring the arrow against the panel's own 30 μm scale bar gives 30.4 μm, consistent. [+-3%]`: 1
- `[UNVERIFIED] Fig. 2(j) (cross-sectional SEM of MP8): printed yellow full-film arrow annotated '31.73 μm'. Arrow-vs-30 μm-scale-bar measurement gives 31.9 μm, consistent. [+-3%]`: 1
- `[UNVERIFIED] Experimental Section: "Note that the film thickness was controlled by varying the volume amount of the mixed solution used. Resultant electrode thickness values were in the range from 3 to 6 μm." SERIES-LEVEL range, not a per-sample value. CAVEAT: the sentence sits at the end of the 'Preparation of the P-MXene/CPolymer-A Electrodes' paragraph, so it is stated most directly for the composite films; all films (including the pristine MXene film) are made by the same vacuum-filtration route with thickness set by filtered volume, so the range is taken to cover the pristine film too. NOT read from a figure: the pristine cross-section SEM (Fig. 2d) is a 500 nm-scale-bar zoom on layer stacking and does not span the film. [+-35%]`: 1
- `[UNCONFIRMED] Printed label burned into Figure 2c (cross-sectional SEM of the Ti3C2Tx film): '3 um, 1.19 mg cm-2'. The white arrow spans the entire film from top surface to substrate. Not a scale-bar estimate - it is the authors' own printed number. Cross-check: 1.19 mg cm-2 / 3.0e-4 cm = 3.97 g cm-3, exactly the 0%-microgel density point read from Figure 2g.`: 1
- `[UNCONFIRMED] Printed label burned into Figure 2b (cross-sectional SEM of RAMX-50% film): '3.5 um, 1.16 mg cm-2'. The white arrow spans the whole film. Cross-check: 1.16 mg cm-2 / 3.5e-4 cm = 3.31 g cm-3, matching the ~3.3 g cm-3 density at 50% microgel in Figure 2g, and matching 736/221 = 3.33 from Figure 3d.`: 1
- `[UNVERIFIED] Figure 3D, cross-section SEM of the spray-coated MXene electrode (scale bar 1 um). The whole electrode is framed: sputtered Au / substrate below, free surface above. At 1000 dpi the scale bar is 379 px = 1 um; the MXene layer spans from the Au top edge to the film surface, i.e. about 500-700 px depending on lateral position because the film top is wavy (measured range ~1.2-1.9 um). Reported as the mid-range value. [+-25%]`: 1

### `mass_loading__unit` (6 shown)

- `mg cm−2`: 46
- `mg cm-2`: 16
- `mg/cm2`: 10
- `µg cm−2`: 3
- `mg`: 3
- `wt. %`: 3

### `mass_loading__conf` (4 shown)

- `stated`: 87
- `derived`: 47
- `uncertain`: 45
- `figure`: 13

### `electrolyte` (21 shown)

- `1 M H2SO4`: 74
- `3 M H2SO4`: 28
- `PVA/H2SO4 gel electrolyte`: 20
- `3 M H2SO4 aqueous electrolyte`: 10
- `0.5 M H2SO4`: 9
- `1 m H2SO4`: 9
- `deaerated 3 M sulfuric acid (H2SO4)`: 8
- `3 M H2SO4 aqueous solution`: 5
- `1 M H2SO4 aqueous solution`: 4
- `H2SO4 aqueous solution`: 4
- `3 m poly(vinyl alcohol) (PVA)/H2SO4`: 4
- `H2SO4-PVA gel electrolyte`: 4
- `2 M H2SO4`: 3
- `1 M PVA/H2SO4 gel electrolyte`: 3
- `3 M H2SO4 + 0.3 M KI`: 1
- `1 mol/L H2SO4`: 1
- `1 m H2SO4 aqueous`: 1
- `2 M PVA/H2SO4 gel electrolyte`: 1
- `3 M H2SO4/PVA gel electrolyte`: 1
- `aqueous H2SO4-PVA gel electrolyte`: 1
- `1M H2SO4`: 1

### `electrolyte__conf` (3 shown)

- `derived`: 170
- `stated`: 20
- `uncertain`: 2

### `ssa__unit` (3 shown)

- `m2 g−1`: 15
- `m2/g`: 15
- `m2 g-1`: 7

### `ssa__conf` (3 shown)

- `stated`: 139
- `uncertain`: 51
- `derived`: 2

### `ssa__src` (2 shown)

- `Text states the series range 'specific surface area (SSA) (8.4-65.4 m2 g-1)' and 'from 8.4 to 65.4 m2 g-1 (Figure 2g)'. Figure 2g shows SSA rising monotonically with microgel content, so 8.4 is the 0%-microgel (pure Ti3C2Tx film) end. My independent pixel read of the 0% red marker gives 8.4 m2 g-1, matching the stated endpoint exactly. [+-2%]`: 1
- `Text states the series range 'SSA ... (from 8.4 to 65.4 m2 g-1) (Figure 2g)'; Figure 2g shows SSA increasing monotonically with microgel content, so 65.4 is the 100%-microgel (microgel film) end. My independent pixel read of the 100% red marker gives 64.7-65.1 m2 g-1, matching the stated endpoint within 1%. [+-2%]`: 1

### `scan_rate` (11 shown)

- `2`: 47
- `5`: 32
- `10`: 22
- `1`: 5
- `2000`: 4
- `5-100`: 3
- `2,000`: 3
- `5-50`: 2
- `100`: 1
- `200`: 1
- `20`: 1

### `scan_rate__unit` (4 shown)

- `mV s−1`: 75
- `mV/s`: 22
- `mV s-1`: 22
- `V s−1`: 2

### `scan_rate__conf` (4 shown)

- `stated`: 129
- `uncertain`: 31
- `figure`: 26
- `derived`: 6

### `current_density__unit` (8 shown)

- `A g−1`: 20
- `A/g`: 12
- `A g-1`: 9
- `mA cm−2`: 8
- `A cm−3`: 5
- `μA`: 2
- `mA cm-2`: 2
- `µA cm−2`: 2

### `current_density__conf` (4 shown)

- `stated`: 124
- `uncertain`: 52
- `figure`: 8
- `derived`: 8

### `current_density__src` (14 shown)

- `Fig. 6d x-axis; the capacitance above is the first data point, on the 1 A/g tick.`: 3
- `Fig. 6b x-axis; the capacitance above is the first data point, on the 1 A/g tick.`: 3
- `[UNVERIFIED] Fig. 4 caption: '(c) GCD of MP0, MP2, MP5 and MP8 at current density of 1.0 A g-1'. Three-electrode cell, 1 M H2SO4.`: 2
- `[UNVERIFIED] Fig. 4 caption: '(c) GCD of MP0, MP2, MP5 and MP8 at current density of 1.0 A g-1'. Three-electrode cell, 1 M H2SO4 (Section 2.5). Condition for the gravimetric capacitance recorded in this row.`: 1
- `Fig. 6b/6d x-axis: the capacitance value above was read at the first data point, which sits on the 1 A/g tick (same current-density series 1, 2, 4, 6, 8, 10 A/g that the text lists for T-40).`: 1
- `Fig. 6d x-axis; the capacitance above is the first data point, on the 1 A/g tick (series 1, 2, 4, 6, 8, 10 A/g).`: 1
- `[UNVERIFIED] 'the corresponding volumetric capacitance values at current density of 1 A g-1 are 1068 F cm-3 (274 F g-1), 1355 F cm-3 (372 F g-1), and 1293 F cm-3 (404 F g-1), respectively' – this is the condition of the 1068 F cm-3 already recorded in this row. Three-electrode, 3 M H2SO4.`: 1
- `[UNVERIFIED] Same sentence: 1355 F cm-3 (372 F g-1) at 1 A g-1 for 3d-Ti3C2Tx-film – the condition of the 1355 F cm-3 already in this row. Three-electrode, 3 M H2SO4.`: 1
- `[UNVERIFIED] Same sentence: 1293 F cm-3 (404 F g-1) at 1 A g-1 for the aerogel – the condition of the 1293 F cm-3 already in this row. Three-electrode, 3 M H2SO4.`: 1
- `Text: "The areal capacitances of up to 73, 134, and 340 mF cm−2 can be achieved by using GCD curves when the thickness of the MXene films are 4, 11, and 21 µm, respectively (Figure 4f)." The already-recorded 73 mF cm−2 / 183 F cm−3 for this 4 µm film are the 'up to' (maximum) values, which the paper explicitly ties to 0.25 mA cm−2 for the other thicknesses ("340 mF cm−2 at 0.25 mA cm−2"; "134, 123, 118, 103, and 90 mF cm−2 at 0.25, 0.5, 1.0, 2.5, and 5.0 mA cm−2"). Verified on the zoomed Fig 4f panel (p.6, 900 dpi crop): the leftmost point of every series sits at 0.25 mA cm−2 on the x-axis, and the green 4 µm point there reads ~72 mF cm−2, matching the stated 73 (within 2%). Same 0.25 mA cm−2 already filled for the 11 and 21 µm rows. Note this is the device (two-electrode MSC) test condition, as are the capacitances already in this row.`: 1
- `[UNCONFIRMED] Fig. 4b legend, lowest GCD current density listed for the Ti3C2Tx/P-100-H electrode ('2 mA/cm2'; series runs 2, 5, 7, 10, 15, 20, 30 mA/cm2). This is the condition paired with the areal_capacitance reported above; it is NOT the condition for the already-filled 286 F g-1 / 1065 F cm-3, which the text gives at 2 mV s-1.`: 1
- `Fig. 2f caption: "Capacitance retention test of a 13 µm hydrogel film performed by galvanostatic cycling at 10 A g−1. The inset depicts galvanostatic cycling profiles collected at 1, 2, 5 and 10 A g−1." This is the galvanostatic cycling-stability current for this electrode (>90% retention after 10,000 cycles); it is NOT the condition for the 370 F g−1 / 1,500 F cm−3 values, which are CV values at 2 mV s−1. GCD profiles for this same film were also taken at 1, 2 and 5 A g−1.`: 1
- `[UNVERIFIED] Condition of the areal capacitance added for this row. Figure 2 caption (p.8): "(b) GCD curves obtained at 2 mA cm-2 (c) specific areal capacitance of the BMX yarn electrodes from the GCD curves"; the point read in Fig. 2c sits on the 2 mA cm-2 abscissa.`: 1
- `[UNVERIFIED] Condition of the areal capacitance added for this row. Figure 2 caption (p.8): "(b) GCD curves obtained at 2 mA cm-2 (c) specific areal capacitance of the BMX yarn electrodes from the GCD curves ... All data were obtained from the BMX yarn electrodes with different MXene loadings"; the point read in Fig. 2c sits on the 2 mA cm-2 abscissa.`: 1

### `gravimetric_capacitance__unit` (3 shown)

- `F g−1`: 70
- `F/g`: 30
- `F g-1`: 22

### `gravimetric_capacitance__conf` (4 shown)

- `stated`: 120
- `figure`: 35
- `uncertain`: 34
- `derived`: 3

### `volumetric_capacitance__unit` (3 shown)

- `F cm−3`: 59
- `F cm-3`: 23
- `F/cm3`: 13

### `volumetric_capacitance__conf` (4 shown)

- `stated`: 115
- `uncertain`: 31
- `derived`: 27
- `figure`: 19

### `areal_capacitance__unit` (5 shown)

- `F cm−2`: 34
- `mF cm−2`: 25
- `mF/cm2`: 7
- `mF cm-2`: 7
- `F/cm2`: 2

### `areal_capacitance__conf` (4 shown)

- `stated`: 110
- `uncertain`: 41
- `derived`: 24
- `figure`: 17

### `overall` (1 shown)

- `PASS`: 192
