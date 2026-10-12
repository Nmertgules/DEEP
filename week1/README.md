# Digital Image Processing — Week 1: OpenCV with CUDA on Google Colab

**Group members: Nmertgules and Iliya**

> *Google Colab üzerinde CUDA destekli OpenCV kurulumu (donanımınız destekliyorsa yerel ortamda VS Code ile de yapılabilir).*

Notebook: [01_cuda_opencv_colab_setup.ipynb](01_cuda_opencv_colab_setup.ipynb) · [Open in Colab](https://colab.research.google.com/github/iliyadindar/DEEP/blob/main/week1/01_cuda_opencv_colab_setup.ipynb)

## Why a build is needed

`pip install opencv-python` (and the OpenCV that Colab pre-installs) is compiled **without CUDA**, so `cv2.cuda.getCudaEnabledDeviceCount()` returns `0` even on a GPU runtime. The GPU modules (`cudaarithm`, `cudafilters`, `cudaimgproc`, …) are in the separate **opencv_contrib** repository and must be compiled with NVIDIA's `nvcc`.

## What the notebook does

1. Checks the GPU and its compute capability (`nvidia-smi`; a T4 is 7.5).
2. Optionally mounts Google Drive. If a cached build exists there, it is restored in seconds.
3. Otherwise it clones `opencv` + `opencv_contrib` (tag `5.0.0`) and builds **only** `core, imgproc, imgcodecs, python3, cudev, cudaarithm, cudafilters, cudaimgproc`, for **only** this GPU's architecture (`CUDA_ARCH_BIN=<cc>`). This is what keeps the build in the tens-of-minutes range.
4. Installs into `/opt/opencv-cuda`, removes Colab's CPU-only OpenCV wheels, and points Python to the new build with a `.pth` file.
5. Verifies with `cv2.cuda.getCudaEnabledDeviceCount()`, the CUDA section of `cv2.getBuildInformation()`, and a GPU vs. CPU Gaussian blur comparison.

Run it with **Runtime → Change runtime type → T4 GPU**, then **Run all**. Set `USE_DRIVE = True` to keep the build for later weeks: each Colab notebook gets its own runtime, so the [Week 4](../week4/README.md) notebooks restore the build from that Drive cache.

## Tested on Colab (12 Oct 2026)

| | |
| --- | --- |
| GPU / driver | Tesla T4 (compute capability 7.5), driver 580.82.07 |
| CUDA toolkit on Colab | 13.0 (`nvcc` V13.0.88) |
| Python | 3.13 |
| OpenCV built | 5.0.0 + opencv_contrib 5.0.0 |
| Build time | 43.0 min (2 CPU cores) |
| `cv2.cuda.getCudaEnabledDeviceCount()` | 1 |
| Build info | `NVIDIA CUDA: YES (ver 13.0, CUFFT CUBLAS FAST_MATH)`, `NVIDIA GPU arch: 75` |
| GPU vs. CPU Gaussian 5×5 on a 1920×1080 float image | max difference 4.6 × 10⁻⁵ |

When `cv2` is imported, Colab may print `Error disabling cv2.imshow()`. This message is harmless: the build has no GUI module, so there is no `imshow` for Colab to patch. Use matplotlib to show images.

The notebook is saved without outputs; the values above come from that Colab run. The run used `USE_DRIVE = False`, so the Drive-cache save/restore path has not been tested yet.

## Locally (VS Code)

On your own machine with an NVIDIA GPU, install the driver and the CUDA Toolkit, then run the same `cmake` command with your GPU's compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`) and an install folder of your choice. On Windows, use Visual Studio's generator from the *x64 Native Tools* prompt instead of Ninja. Then select that Python interpreter in VS Code. Without an NVIDIA GPU there is nothing to accelerate, and the course notebooks fall back to the CPU.
