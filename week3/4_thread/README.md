# Digital Image Processing — Week 3 · 4 Threads

**Group members: Nmertgules and Iliya**

The five task scripts use the shared `cpu_parallel.py` implementation. Numba compiles the pixel loops to native CPU code, and `set_num_threads(4)` selects four workers. Histogram chunks are counted independently before merging; CLAHE tiles and output pixels are processed in parallel. The small CDF and final reductions remain sequential.

## Run from the repository root

```bash
python -m pip install -r week3/4_thread/requirements.txt
python week3/orijinal/00_test_goruntusu_olustur.py
python week3/4_thread/soru1_dogrusal_olcekleme.py
python week3/4_thread/soru2_histogram_esitleme.py
python week3/4_thread/soru3_clahe.py
python week3/4_thread/soru4_ortalama_filtre.py
python week3/4_thread/soru5_medyan_filtre.py
python week3/4_thread/test_cpu.py
```

Each task accepts an image path and optional `--threads 1` or `--threads 4` (default: 4), plus `--output`. Inputs default to the original image generator's outputs in `../orijinal/cikti/`. New outputs go into this folder's `results/`; original results are not overwritten.

## Tested results

All five 384×512 task outputs matched OpenCV exactly (maximum pixel difference 0). One-thread and four-thread outputs were identical. Four distinct native worker IDs (0, 1, 2, 3) were observed using the OpenMP threading layer.

### Latest timing comparison

These are the recorded median times from the [latest test run](son_test_sonuclari.json).

| Task | 1 thread (ms) | 4 threads (ms) | Speedup (1-thread time / 4-thread time) | Maximum pixel difference from OpenCV |
| --- | ---: | ---: | ---: | ---: |
| 1 · Linear contrast stretching | 0.241 | 0.709 | 0.34× | 0 |
| 2 · Histogram equalization | 0.357 | 0.144 | 2.48× | 0 |
| 3 · CLAHE | 2.444 | 1.050 | 2.33× | 0 |
| 4 · 3×3 mean convolution | 13.782 | 4.523 | 3.05× | 0 |
| 5 · 5×5 median filter | 68.791 | 27.158 | 2.53× | 0 |

A ratio above 1 means four threads were faster; below 1 means they were slower.

- **Task 1:** Four threads took about 2.95 times as long in this run. The small workload did not benefit from parallel execution in this measurement.
- **Task 2:** Histogram counting and mapping produced the same pixels with approximately 2.48× speedup.
- **Task 3:** Parallel CLAHE tile processing and interpolation produced the same pixels with approximately 2.33× speedup.
- **Task 4:** The 3×3 mean filter produced the same pixels with approximately 3.05× speedup.
- **Task 5:** The 5×5 median filter produced the same pixels with approximately 2.53× speedup.

### Measurement method and additional checks

Each timing is the median of nine batches, with five calls per batch. A warm-up excludes JIT compilation. Timings include allocations and CPU preprocessing. The comparison uses the **same compiled implementation with one versus four workers**, not the original Python-loop scripts.

Measurements depend on machine load, image size, and thread overhead. The [initial run](test_sonuclari.json) gave different timings, including a small speedup for Task 1; four workers do not guarantee faster execution. Rerunning `test_cpu.py` writes a fresh report to `results/cpu_results.json`.

The 28 additional shape comparisons passed the test tolerance of at most one intensity level. One 3×5 histogram-equalization case differed from OpenCV by one level due to rounding; all other recorded shape comparisons matched exactly. The constant-image equalization check also passed.

The new implementation corrects CLAHE border padding and preserves constant images during histogram equalization. The [original version](../orijinal/) is retained without code changes. This folder uses CPU threads, not CUDA.
