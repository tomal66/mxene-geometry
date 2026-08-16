# 01 - Preprocessing log

## Target cleaning (`gravimetric_capacitance` -> `target_cap_F_g`)

A value is kept only if it is a single defensible number.  Ranges (`300-350`), inequalities (`over 200`, `<300`) and multi-phase strings are **never** collapsed to a midpoint - they are dropped and counted here.

| disposition | n rows |
| --- | ---: |
| `kept:exact` | 116 |
| `dropped:no_value_reported` | 70 |
| `dropped:inequality` | 4 |
| `kept:approximate` | 2 |

### Rows dropped for a non-empty but non-defensible target value

| paper_id | raw value | parsed kind |
| --- | --- | --- |
| 10.1038/s41586-018-0109-z | `over 200` | inequality |
| 10.1038/s41586-018-0109-z | `over 200` | inequality |
| 10.1038/s41586-018-0109-z | `over 200` | inequality |
| 10.1002/smsc.202500367 | `exceeding 100` | inequality |

**Rows retained for modelling: 118 / 192** (from 40 papers).
Approximate targets (`~`, `about`, mean+/-sd) retained and flagged: 2.

Physically extreme capacitance values flagged (<20 or >1500 F/g): 2. These are retained (nothing is hidden) and listed here:

| paper_id | target_cap_F_g |
| --- | ---: |
| 10.1016/j.jallcom.2023.173355 | 1609.5 |
| 10.1002/adma.202304757 | 7.26 |

## Numeric descriptor standardisation

Conversions applied only where the source unit is unambiguous (`nm -> A` x10, `mm -> um` x1000, `ug/cm2 -> mg/cm2` /1000, `V/s -> mV/s` x1000). Incompatible bases (e.g. `mA/cm2` or `A/cm3` for a gravimetric current density; `mg` or `wt %` for an areal mass loading) are **left missing**, never converted.

| standardised_column    | source_column       | canonical_unit   |   n_raw_nonnull |   n_standardised |   n_approximate |   n_outside_plausible_band | plausible_band   | rejection_reasons                                                                                                             |
|:-----------------------|:--------------------|:-----------------|----------------:|-----------------:|----------------:|---------------------------:|:-----------------|:------------------------------------------------------------------------------------------------------------------------------|
| interlayer_A           | interlayer_spacing  | A (angstrom)     |             126 |              119 |               5 |                          0 | [5.0, 60.0]      | missing_unit=1; value:missing=66; value:multi=2; value:range=4                                                                |
| scan_rate_mV_s         | scan_rate           | mV/s             |             121 |              116 |               0 |                          0 | [0.1, 100000.0]  | value:missing=71; value:range=5                                                                                               |
| current_density_A_g    | current_density     | A/g              |              60 |               41 |               0 |                          0 | [0.01, 1000.0]   | incompatible_unit:a/cm3=5; incompatible_unit:ma/cm2=10; incompatible_unit:ua=2; incompatible_unit:ua/cm2=2; value:missing=132 |
| mass_loading_mg_cm2    | mass_loading        | mg/cm2           |              80 |               74 |               4 |                          0 | [0.01, 100.0]    | incompatible_unit:mg=2; incompatible_unit:wt%=3; value:missing=112; value:range=1                                             |
| electrode_thickness_um | electrode_thickness | um               |              85 |               83 |              31 |                          1 | [0.01, 5000.0]   | value:missing=107; value:range=2                                                                                              |
| ssa_m2_g               | ssa                 | m2/g             |              37 |               36 |               0 |                          0 | [0.1, 3000.0]    | value:inequality=1; value:missing=155                                                                                         |
| flake_size_um          | flake_size          | um               |              99 |               40 |              37 |                          0 | [0.001, 1000.0]  | value:inequality=19; value:missing=93; value:multi=2; value:qualitative=21; value:range=17                                    |
| pore_diameter_nm       | pore_diameter       | nm               |              38 |                7 |               5 |                          0 | [0.1, 100000.0]  | value:inequality=9; value:missing=154; value:range=22                                                                         |

## Electrolyte parsing (`h2so4_M`)

Parsed from the `electrolyte` field only.  `synthesis_method` also contains acid concentrations (e.g. wet-spinning in 98 wt% H2SO4) but those are *synthesis* solutions, not the test electrolyte, so that field is never consulted.

| electrolyte                         | status                         |   h2so4_M |   n_rows |
|:------------------------------------|:-------------------------------|----------:|---------:|
| 0.5 M H2SO4                         | parsed_single                  |       0.5 |        9 |
| 1 M H2SO4                           | parsed_single                  |       1   |       74 |
| 1 M H2SO4 aqueous solution          | parsed_single                  |       1   |        4 |
| 1 M PVA/H2SO4 gel electrolyte       | parsed_single                  |       1   |        3 |
| 1 m H2SO4                           | parsed_single                  |       1   |        9 |
| 1 m H2SO4 aqueous                   | parsed_single                  |       1   |        1 |
| 1 mol/L H2SO4                       | parsed_single                  |       1   |        1 |
| 1M H2SO4                            | parsed_single                  |       1   |        1 |
| 2 M H2SO4                           | parsed_single                  |       2   |        3 |
| 2 M PVA/H2SO4 gel electrolyte       | parsed_single                  |       2   |        1 |
| 3 M H2SO4                           | parsed_single                  |       3   |       28 |
| 3 M H2SO4 + 0.3 M KI                | parsed_multi_anchored_on_H2SO4 |       3   |        1 |
| 3 M H2SO4 aqueous electrolyte       | parsed_single                  |       3   |       10 |
| 3 M H2SO4 aqueous solution          | parsed_single                  |       3   |        5 |
| 3 M H2SO4/PVA gel electrolyte       | parsed_single                  |       3   |        1 |
| 3 m poly(vinyl alcohol) (PVA)/H2SO4 | parsed_single                  |       3   |        4 |
| H2SO4 aqueous solution              | no_concentration_reported      |     nan   |        4 |
| H2SO4-PVA gel electrolyte           | no_concentration_reported      |     nan   |        4 |
| PVA/H2SO4 gel electrolyte           | no_concentration_reported      |     nan   |       20 |
| aqueous H2SO4-PVA gel electrolyte   | no_concentration_reported      |     nan   |        1 |
| deaerated 3 M sulfuric acid (H2SO4) | parsed_single                  |       3   |        8 |

## Categorical family mappings

- `layer_class`: {'single_or_few_layer': 55, 'delaminated': 30, nan: 23, 'multilayer': 10}
- `composition_family`: {'pristine_mxene': 69, 'mxene_carbon_composite': 19, 'mxene_polymer_composite': 15, 'structured_pristine_mxene': 11, 'doped_mxene': 3, 'mxene_inorganic_composite': 1}
- `synthesis_family`: {'LiF_HCl_MILD': 118}
- `electrolyte_is_gel`: {False: 112, True: 6}

## Paper grouping

`paper_id` = DOI when present, else `source_file`.  DOI and source_file are 1:1 in this corpus, so the two definitions agree; 3 rows lack a DOI and fall back to their filename.
Unique papers among modelled rows: **40**.