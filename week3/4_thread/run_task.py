"""Shared CLI for the five native CPU tasks."""
import argparse
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
from numba import get_num_threads
from cpu_parallel import run

def main(task, default_image):
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?', default=str(here.parent/'orijinal'/'cikti'/default_image))
    parser.add_argument('--threads', type=int, choices=[1, 4], default=4)
    parser.add_argument('--output', default=str(here/'results'/f'task{task}_4threads.png'))
    args = parser.parse_args()
    image = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit('Cannot read input. Run week3/orijinal/00_test_goruntusu_olustur.py first or supply an image path.')
    run(task, image, args.threads)  # compile/warm up before timing
    start = perf_counter()
    result = run(task, image, args.threads)
    elapsed = (perf_counter()-start)*1000
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result):
        raise SystemExit('Could not save output image')
    print(f'Task {task}: workers={get_num_threads()}, time={elapsed:.3f} ms')
    print(f'Output: {output}; range={int(result.min())}..{int(result.max())}')
