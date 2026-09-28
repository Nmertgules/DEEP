# Digital Image Processing — Week 2

This repository contains a clean, reproducible set of Week 2 image-processing exercises.

**Group members:** Nmertgules and Iliya

## Exercises

Each task is a separate Jupyter notebook in week2/.

| Notebook | Purpose |
| --- | --- |
| [01_resize_image.ipynb](week2/01_resize_image.ipynb) | Resize an image at four scale factors |
| [02_intensity_quantization.ipynb](week2/02_intensity_quantization.ipynb) | Compare grayscale quantization at 2, 4, 8, and 16 levels |
| [03_four_to_sixteen_level_mapping.ipynb](week2/03_four_to_sixteen_level_mapping.ipynb) | Map four source levels onto a 16-step palette and explain the information limit |
| [04_quadrant_quantization.ipynb](week2/04_quadrant_quantization.ipynb) | Quantize four image regions and inspect the residual |
| [05_whole_image_vs_quadrants.ipynb](week2/05_whole_image_vs_quadrants.ipynb) | Compare whole-image and quadrant-wise processing time |
| [06_brightness_contrast.ipynb](week2/06_brightness_contrast.ipynb) | Compare a linear transform in NumPy and OpenCV |

## Setup and use

Install the dependencies and Jupyter with:

    python -m pip install -r requirements.txt

Open any notebook with Jupyter or Google Colab. To start Jupyter locally:

    jupyter notebook week2/01_resize_image.ipynb

Each notebook uses the scikit-image camera sample by default. Set IMAGE_PATH in its code cell to a local image to use custom input. The notebooks are committed without saved cell outputs to keep the repository small; run the cell to reproduce the visualizations and measurements.

## Method notes

- The resize exercise uses area interpolation for downscaling and cubic interpolation for upscaling.
- Mapping four quantized levels onto a 16-step palette does not restore discarded image detail; at most four distinct output levels remain.
- The timing comparison checks that both methods produce identical pixels and includes splitting and reassembly in the quadrant measurement.
- Timing depends on the machine, image size, and current system load.
