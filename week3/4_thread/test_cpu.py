import json,time,sys
from pathlib import Path
import numpy as np,cv2
import cpu_parallel as cpu
from numba import set_num_threads,threading_layer
root=Path(__file__).resolve().parent
inputs=Path(__file__).resolve().parents[1]/'orijinal/cikti'
out=root/'results';out.mkdir(exist_ok=True)
rows=[]
for task,name in [(1,'dusuk_kontrast'),(2,'dusuk_kontrast'),(3,'dusuk_kontrast'),(4,'gurultulu_gauss'),(5,'tuz_biber')]:
 im=cv2.imread(str(inputs/(name+'.png')),0)
 reference=[None,lambda:cv2.normalize(im,None,0,255,cv2.NORM_MINMAX),lambda:cv2.equalizeHist(im),lambda:cv2.createCLAHE(clipLimit=8,tileGridSize=(8,8)).apply(im),lambda:cv2.blur(im,(3,3)),lambda:cv2.medianBlur(im,5)][task]()
 results={};times={}
 for workers in [1,4]:
  # warmup excludes compilation and initializes thread pool
  results[workers]=cpu.run(task,im,workers)
  samples=[]
  for batch in range(9):
   t=time.perf_counter()
   for _ in range(5):cpu.run(task,im,workers)
   samples.append((time.perf_counter()-t)*1000/5)
  times[workers]=float(np.median(samples))
 d=int(np.abs(results[4].astype(int)-reference.astype(int)).max())
 assert np.array_equal(results[1],results[4]),'1/4 thread mismatch'
 assert d<=1,(task,d)
 cv2.imwrite(str(out/f'task{task}_4threads.png'),results[4])
 row={'task':task,'max_difference_opencv':d,'one_thread_ms':times[1],'four_threads_ms':times[4],'speedup':times[1]/times[4]};rows.append(row);print(row,flush=True)
set_num_threads(4);ids=cpu.worker_ids();assert len(set(ids))==4,ids
print('Native worker IDs:',ids,'layer:',threading_layer(),flush=True)
rng=np.random.default_rng(14);edge=[]
for shape in [(32,33),(33,32),(3,5),(1,40),(37,53),(1,1),(8,8)]:
 im=rng.integers(0,256,shape,dtype=np.uint8)
 for task in [2,3,4,5]:
  ref={2:lambda:cv2.equalizeHist(im),3:lambda:cv2.createCLAHE(clipLimit=8,tileGridSize=(8,8)).apply(im),4:lambda:cv2.blur(im,(3,3)),5:lambda:cv2.medianBlur(im,5)}[task]()
  a=cpu.run(task,im,4);d=int(np.abs(a.astype(int)-ref.astype(int)).max());assert d<=1,(task,shape,d)
  edge.append({'task':task,'shape':shape,'max_difference':d})
constant=np.full((8,8),128,np.uint8);assert np.array_equal(cpu.run(2,constant),constant)
report={'cpu_workers':ids.tolist(),'threading_layer':threading_layer(),'main_results':rows,'edge_tests':edge,'constant_equalization_pass':True,'gpu_tested':False}
(out/'cpu_results.json').write_text(json.dumps(report,indent=2));print('CPU TESTS PASS',flush=True)
