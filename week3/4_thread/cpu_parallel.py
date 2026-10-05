"""Local Week 3 trial: real native CPU loops, selectable 1 or 4 threads."""
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads, get_thread_id

@njit(parallel=True)
def worker_ids():
    ids=np.empty(4,np.int32)
    for i in prange(4): ids[i]=get_thread_id()
    return ids

@njit(parallel=True)
def extrema(a):
    lo=np.empty(4,np.int32); hi=np.empty(4,np.int32)
    for c in prange(4):
        mn=255; mx=0
        for i in range(c*a.size//4,(c+1)*a.size//4):
            v=int(a[i]); mn=min(mn,v); mx=max(mx,v)
        lo[c]=mn;hi[c]=mx
    return lo.min(),hi.max()

@njit(parallel=True)
def stretch(a,mn,mx):
    out=np.empty_like(a)
    for i in prange(a.size):
        out[i]=0 if mn==mx else int((int(a[i])-mn)*(255.0/(mx-mn))+0.5)
    return out

@njit(parallel=True)
def histogram(a):
    partial=np.zeros((4,256),np.int64)
    for c in prange(4):
        for i in range(c*a.size//4,(c+1)*a.size//4): partial[c,int(a[i])]+=1
    hist=np.zeros(256,np.int64)
    for v in prange(256):
        for c in range(4):hist[v]+=partial[c,v]
    return hist

@njit
def equalization_lut(hist,n):
    first=0
    while hist[first]==0:first+=1
    lut=np.zeros(256,np.uint8)
    if hist[first]==n:
        for v in range(256):lut[v]=v
        return lut
    total=0
    for v in range(256):
        total+=hist[v]
        if total>0:lut[v]=max(0,min(255,int((total-hist[first])/(n-hist[first])*255+0.5)))
    return lut

@njit(parallel=True)
def apply_lut(a,lut):
    out=np.empty_like(a)
    for i in prange(a.size):out[i]=lut[a[i]]
    return out

@njit
def reflect(i,n):
    if n==1:return 0
    period=2*n-2;i=i%period
    return i if i<n else period-i

@njit(parallel=True)
def clahe_padding(im,gx,gy):
    h,w=im.shape
    # Match OpenCV: if either dimension needs padding, pad both axes.
    ph=0;pw=0
    if h%gy!=0 or w%gx!=0:ph=gy-h%gy;pw=gx-w%gx
    out=np.empty((h+ph,w+pw),np.uint8)
    for y in prange(h+ph):
        for x in range(w+pw):out[y,x]=im[reflect(y,h),reflect(x,w)]
    return out

@njit(parallel=True)
def tile_luts(padded,gx,gy,clip):
    th=padded.shape[0]//gy;tw=padded.shape[1]//gx;area=th*tw
    luts=np.empty((gy,gx,256),np.uint8)
    limit=max(1,int(clip*area/256)) if clip>0 else 0
    scale=np.float32(255)/np.float32(area)
    for t in prange(gx*gy):
        ty=t//gx;tx=t%gx;hist=np.zeros(256,np.int64)
        for y in range(ty*th,(ty+1)*th):
            for x in range(tx*tw,(tx+1)*tw):hist[padded[y,x]]+=1
        if limit>0:
            excess=0
            for v in range(256):
                if hist[v]>limit:excess+=hist[v]-limit;hist[v]=limit
            batch=excess//256;remaining=excess%256
            for v in range(256):hist[v]+=batch
            if remaining:
                step=max(256//remaining,1);v=0
                while v<256 and remaining>0:hist[v]+=1;v+=step;remaining-=1
        total=0
        for v in range(256):
            total+=hist[v];luts[ty,tx,v]=min(255,max(0,int(np.rint(np.float32(total)*scale))))
    return luts

@njit(parallel=True)
def interpolate(im,luts,tw,th):
    h,w=im.shape;gy,gx,_=luts.shape;out=np.empty_like(im)
    iw=np.float32(1)/np.float32(tw);ih=np.float32(1)/np.float32(th)
    for y in prange(h):
        yf=np.float32(y)*ih-np.float32(0.5);y1=int(np.floor(yf));y2=y1+1
        ya=np.float32(yf-np.float32(y1));yb=np.float32(1)-ya
        y1=max(0,min(gy-1,y1));y2=max(0,min(gy-1,y2))
        for x in range(w):
            xf=np.float32(x)*iw-np.float32(0.5);x1=int(np.floor(xf));x2=x1+1
            xa=np.float32(xf-np.float32(x1));xb=np.float32(1)-xa
            x1=max(0,min(gx-1,x1));x2=max(0,min(gx-1,x2));v=im[y,x]
            top=np.float32(luts[y1,x1,v])*xb+np.float32(luts[y1,x2,v])*xa
            bot=np.float32(luts[y2,x1,v])*xb+np.float32(luts[y2,x2,v])*xa
            out[y,x]=min(255,max(0,int(np.rint(top*yb+bot*ya))))
    return out

@njit(parallel=True)
def mean3(im):
    h,w=im.shape;out=np.empty_like(im)
    for y in prange(h):
        for x in range(w):
            total=0
            for dy in range(-1,2):
                for dx in range(-1,2):total+=int(im[reflect(y+dy,h),reflect(x+dx,w)])
            out[y,x]=(total+4)//9
    return out

@njit(parallel=True)
def median5(im):
    h,w=im.shape;out=np.empty_like(im)
    for y in prange(h):
        for x in range(w):
            vals=np.empty(25,np.int32);k=0
            for dy in range(-2,3):
                for dx in range(-2,3):
                    vals[k]=im[max(0,min(h-1,y+dy)),max(0,min(w-1,x+dx))];k+=1
            for i in range(1,25):
                key=vals[i];j=i-1
                while j>=0 and vals[j]>key:vals[j+1]=vals[j];j-=1
                vals[j+1]=key
            out[y,x]=vals[12]
    return out

def run(task,im,threads=4,clip=8.0,grid=(8,8)):
    set_num_threads(threads);im=np.ascontiguousarray(im,dtype=np.uint8)
    if task==1:
        a=im.ravel();mn,mx=extrema(a);return stretch(a,mn,mx).reshape(im.shape)
    if task==2:
        a=im.ravel();return apply_lut(a,equalization_lut(histogram(a),a.size)).reshape(im.shape)
    if task==3:
        gx,gy=grid;p=clahe_padding(im,gx,gy);l=tile_luts(p,gx,gy,clip)
        return interpolate(im,l,p.shape[1]//gx,p.shape[0]//gy)
    if task==4:return mean3(im)
    if task==5:return median5(im)
    raise ValueError('Task must be 1..5')
