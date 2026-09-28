# Execution and validation record

All six notebooks were executed with two distinct web photographs: NASA's astronaut portrait and Rachel Michetti's coffee photograph. The submitted notebooks retain the actual plots, tables, execution counts, and check outputs.

- Execution time (UTC): 2026-09-28T10:11:38.134023+00:00
- Environment: Python 3.12.14, Windows AMD64
- Every notebook started in a fresh IPython kernel and ran from top to bottom.
- Image checksums matched the originals; there were no cell errors or stderr outputs.

| Notebook | Executed code cells | Embedded figures | Result |
| --- | ---: | ---: | --- |
| [01_resize_image.ipynb](01_resize_image.ipynb) | 4 | 2 | PASS |
| [02_intensity_quantization.ipynb](02_intensity_quantization.ipynb) | 4 | 2 | PASS |
| [03_four_to_sixteen_level_mapping.ipynb](03_four_to_sixteen_level_mapping.ipynb) | 4 | 2 | PASS |
| [04_quadrant_quantization.ipynb](04_quadrant_quantization.ipynb) | 4 | 2 | PASS |
| [05_whole_image_vs_quadrants.ipynb](05_whole_image_vs_quadrants.ipynb) | 4 | 2 | PASS |
| [06_brightness_contrast.ipynb](06_brightness_contrast.ipynb) | 4 | 2 | PASS |

## Checks

- **01:** PASS: all eight photo resizes, output dtypes, and tiny/odd-dimension checks.
- **02:** PASS: both photographs, quantization error bounds, and all 256 input intensities.
- **03:** PASS: remapping keeps four levels; direct quantization produces 16 on the full ramp.
- **04:** PASS: both photographs, residual bounds, exact reassembly, and odd-dimension cases.
- **05:** PASS: pixel equality for both photos and odd dimensions; quantized range verified.
- **06:** PASS: both photos, all 256 possible intensities, and endpoints. Ramp max difference: 0.

## Numerical observations

| Photograph | Quadrant residual range | Whole/quadrant outputs | NumPy/OpenCV maximum difference |
| --- | --- | --- | ---: |
| Astronaut | 0 to 63 | Identical | 0 |
| Coffee | 0 to 63 | Identical | 0 |

The four-level remapping uses four output values; the direct 16-level quantizer uses the original photo. Runtime measurements include warm-up and repeated batches and are specific to this environment.

[Machine-readable measurements and package versions](validation.json) · [Photo sources and licenses](assets/README.md)

The executed code cells and embedded figures passed the automatic checks. All 12 saved figures were also visually reviewed: labels, intensity scales, legends, and layouts are readable and complete.

A separate run of Task 01 in an empty working directory also passed: it downloaded both original web photographs, verified their hashes, and completed every cell. This checks the shared standalone input-loading path.
