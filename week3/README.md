# Digital Image Processing — Week 3

**Group members: Nmertgules and Iliya**

Five tasks cover contrast stretching, histogram equalization, CLAHE, mean filtering, and median filtering. The submitted Python scripts are preserved as provided by the group.

| Task | File |
| --- | --- |
| 01 · Linear contrast stretching | [soru1_dogrusal_olcekleme.py](soru1_dogrusal_olcekleme.py) |
| 02 · Histogram equalization | [soru2_histogram_esitleme.py](soru2_histogram_esitleme.py) |
| 03 · OpenCV and manual CLAHE | [soru3_clahe.py](soru3_clahe.py) |
| 04 · 3×3 mean convolution | [soru4_ortalama_filtre.py](soru4_ortalama_filtre.py) |
| 05 · OpenCV and manual 5×5 median filter | [soru5_medyan_filtre.py](soru5_medyan_filtre.py) |

## Run

Install NumPy, OpenCV, and Matplotlib in your Python environment. Run the image generator first; it creates the default inputs in `week3/cikti/`. Then run the five tasks and the optional comparison figure script:

```bash
python week3/00_test_goruntusu_olustur.py
python week3/soru1_dogrusal_olcekleme.py
python week3/soru2_histogram_esitleme.py
python week3/soru3_clahe.py
python week3/soru4_ortalama_filtre.py
python week3/soru5_medyan_filtre.py
python week3/06_karsilastirma_gorseli.py
```

Each task also accepts an input image path. Results are saved under `week3/cikti/`.

## Validation status

The uploaded scripts passed Python syntax parsing. Their computations and embedded comparisons have not been executed as part of this upload. These files do not include explicit multithread or CUDA implementations.
