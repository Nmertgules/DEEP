# Execution and validation record

All five notebooks were executed with two distinct web photographs: NASA's astronaut portrait and Rachel Michetti's coffee photograph. The submitted notebooks retain the actual plots, tables, execution counts, and check outputs.

- Execution time (UTC): 2026-09-28T10:38:00.364161+00:00
- Environment: Python 3.12.14, Windows AMD64
- Every notebook started in a fresh IPython kernel and ran from top to bottom.
- Image checksums matched the originals; there were no cell errors or stderr outputs.

| Notebook | Executed code cells | Embedded figures | Result |
| --- | ---: | ---: | --- |
| [01_resize_image.ipynb](01_resize_image.ipynb) | 4 | 2 | PASS |
| [02_intensity_quantization.ipynb](02_intensity_quantization.ipynb) | 4 | 2 | PASS |
| [03_four_to_sixteen_level_mapping.ipynb](03_four_to_sixteen_level_mapping.ipynb) | 4 | 2 | PASS |
| [04_whole_image_vs_quadrants.ipynb](04_whole_image_vs_quadrants.ipynb) | 4 | 2 | PASS |
| [05_brightness_contrast.ipynb](05_brightness_contrast.ipynb) | 4 | 2 | PASS |

## Checks

- **01:** PASS: all eight photo resizes, output dtypes, and tiny/odd-dimension checks.
- **02:** PASS: both photographs, quantization error bounds, and all 256 input intensities.
- **03:** PASS: remapping keeps four levels; direct quantization produces 16 on the full ramp.
- **04:** PASS: pixel equality for both photos and odd dimensions; quantized range verified.
- **05:** PASS: both photos, all 256 possible intensities, and endpoints. Ramp max difference: 0.

## Numerical observations

| Photograph | Whole/quadrant outputs (Task 04) | NumPy/OpenCV maximum difference (Task 05) |
| --- | --- | ---: |
| Astronaut | Identical | 0 |
| Coffee | Identical | 0 |

The four-level remapping uses four output values; the direct 16-level quantizer uses the original photo. Runtime measurements include warm-up and repeated batches and are specific to this environment.

[Machine-readable measurements and package versions](validation.json) · [Photo sources and licenses](assets/README.md)

All 20 executed code cells passed the automatic checks. All 10 saved figures were visually reviewed: labels, intensity scales, legends, and layouts are readable and complete.

A separate earlier run of Task 01 in an empty working directory also passed: it downloaded both original web photographs, verified their hashes, and completed every cell. This checks the unchanged shared standalone input-loading path.

## Repository audit

The final submission contains five independent notebooks numbered 01 through 05. Notebook headings, README links, execution records, and SHA-256 hashes match the current files. All local documentation links resolve, and no obsolete task filenames or results remain.
