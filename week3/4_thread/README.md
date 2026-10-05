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

All five 384×512 task outputs matched OpenCV exactly (maximum pixel difference 0). One-thread and four-thread outputs were identical. Four distinct native worker IDs (0, 1, 2, 3) were observed using the OpenMP threading layer. Additional dimension and constant-image checks passed.

The initial nine-batch measurements showed speedups of approximately 1.09×, 1.25×, 1.64×, 1.81×, and 1.79× for Tasks 1–5. The benchmark compares the same compiled implementation with one versus four workers, excludes JIT compilation by warm-up, and includes allocations/preprocessing. Results depend on the machine and workload; these are not speedups against the original Python-loop scripts. See [test_sonuclari.json](test_sonuclari.json) for the initial measurements. Rerunning `test_cpu.py` saves a fresh report in `results/cpu_results.json`.

The new implementation corrects CLAHE border padding and preserves constant images during histogram equalization. The [original version](../orijinal/) is retained without code changes. This folder uses CPU threads, not CUDA.

A rerun after reorganizing the folders again passed all correctness checks. Its timings varied: Task 1 was slower with four workers, while Tasks 2–5 were faster. See [latest test measurements](son_test_sonuclari.json). Four workers do not guarantee a speedup for small workloads.
