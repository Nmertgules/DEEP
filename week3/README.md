# Digital Image Processing — Week 3

**Group members: Nmertgules and Iliya**

The same five tasks are organized into two versions. Each version has its own README, execution instructions, and test results.

| Version | Contents |
| --- | --- |
| [Original](orijinal/) · [README and results](orijinal/README.md) | Original group scripts, helpers, and saved result figures. Original code is unchanged. |
| [4 Threads](4_thread/) · [README and results](4_thread/README.md) | Five task entry points sharing native CPU kernels, tested with four worker threads. Includes one-thread/four-thread comparisons and measured results. |

Tasks: linear contrast stretching, histogram equalization, CLAHE, 3×3 mean convolution, and 5×5 median filtering.

The 4-thread version corrects CLAHE padding and constant-image histogram equalization. Both original known limitations and the new test results are documented in the respective READMEs. The 4-thread results matched OpenCV on all five main test inputs. Performance varies by task and machine.

Generate inputs with `python week3/orijinal/00_test_goruntusu_olustur.py` before running either version. The 4-thread version reads those same inputs and saves its outputs separately.

CUDA is not included in this update.
