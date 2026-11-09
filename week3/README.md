# Digital Image Processing — Week 3

**Group members: Nmertgules and Iliya**

The same five tasks are organized into two versions. Each version has its own README, execution instructions, and test results. A notebook runs both versions in one place.

| Version | Contents |
| --- | --- |
| [Notebook](uzamsal_filtreler.ipynb) · [Colab](https://colab.research.google.com/github/iliyadindar/DEEP/blob/main/week3/uzamsal_filtreler.ipynb) | Runs the unchanged original scripts task by task: shows the source of the functions each task asks for, the scripts' own tests and measurements, and the result images inline; then the 1-thread/4-thread comparison. Turkish explanations, saved with outputs. |
| [Original](orijinal/) · [README and results](orijinal/README.md) | Original group scripts, helpers, and saved result figures. Original code is unchanged. |
| [4 Threads](4_thread/) · [README and results](4_thread/README.md) | Five task entry points sharing native CPU kernels, tested with four worker threads. Includes one-thread/four-thread comparisons and measured results. |

Tasks: linear contrast stretching, histogram equalization, CLAHE, 3×3 mean convolution, and 5×5 median filtering.

The 4-thread version corrects CLAHE padding and constant-image histogram equalization. Both original known limitations and the new test results are documented in the respective READMEs. The 4-thread results matched OpenCV on all five main test inputs. Performance varies by task and machine.

Generate inputs with `python week3/orijinal/00_test_goruntusu_olustur.py` before running either version. The 4-thread version reads those same inputs and saves its outputs separately. The notebook does this itself.

## Notebook run (9 Nov 2026)

The outputs saved in [uzamsal_filtreler.ipynb](uzamsal_filtreler.ipynb) come from one top-to-bottom run (Python 3.12, NumPy 2.5.4, OpenCV 5.0.0, Numba 0.68.1, WSL2 with a 20-thread CPU; about 30 s). All results match the numbers in the [original README](orijinal/README.md): manual stretching, histogram equalization, CLAHE (including the 500 × 377 padding case), the 3×3 mean and the 5×5 median filter all have a maximum difference of **0** from OpenCV, and PSNR rises 22.35 → 29.26 dB (mean) and 14.94 → 35.05 dB (median).

| Task | 1 thread (ms) | 4 threads (ms) | Speedup |
| --- | ---: | ---: | ---: |
| 1 · Linear contrast stretching | 0.326 | 0.152 | 2.14× |
| 2 · Histogram equalization | 0.278 | 0.093 | 3.00× |
| 3 · CLAHE | 2.643 | 0.811 | 3.26× |
| 4 · 3×3 mean convolution | 8.970 | 2.328 | 3.85× |
| 5 · 5×5 median filter | 43.531 | 12.313 | 3.54× |

On this machine four threads are faster in every task, including Task 1, which was slower with four threads in the [recorded 4-thread run](4_thread/README.md). Thread start-up cost is fixed, so whether it pays off for a 0.3 ms job depends on the CPU; the heavier tasks speed up on every machine tested.

CUDA is not included in this update.
