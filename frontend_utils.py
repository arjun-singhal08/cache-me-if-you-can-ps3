from __future__ import annotations
import re,sys,subprocess,tempfile,io
from pathlib import Path
from typing import Iterable
import joblib,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent

def safe_name(name): return re.sub(r"[^A-Za-z0-9._-]","_",Path(name).name)
def save_uploads(files,directory):
    out=[]
    for f in files:
        p=directory/safe_name(f.name); f.seek(0); p.write_bytes(f.getbuffer()); out.append(p)
    return out

def run_cli(command,cwd,timeout=360):
    r=subprocess.run(command,cwd=str(cwd),capture_output=True,text=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError((r.stderr.strip() or r.stdout.strip() or "Inference failed")[-5000:])
    return r.stdout

def run_door(uploaded_file):
    with tempfile.TemporaryDirectory() as x:
        td=Path(x); inp=save_uploads([uploaded_file],td)[0]; out=td/'door_predictions.csv'
        run_cli([sys.executable,'predict.py','--input',str(inp),'--output',str(out),'--model-dir',str(ROOT/'Door'/'model')],ROOT/'Door')
        return pd.read_csv(out),pd.read_csv(inp)

def run_shm(uploaded_files):
    with tempfile.TemporaryDirectory() as x:
        td=Path(x); inp=td/'input';inp.mkdir();save_uploads(uploaded_files,inp);out=td/'shm_predictions.csv'
        run_cli([sys.executable,'predict.py','--input',str(inp),'--output',str(out),'--model-dir',str(ROOT/'SHM'/'models')],ROOT/'SHM',420)
        return pd.read_csv(out)

def run_acv(uploaded_file):
    uploaded_file.seek(0); raw=pd.read_excel(uploaded_file)
    # Use the teammate-provided Python API and genuine production artifacts.
    sys.path.insert(0,str(ROOT))
    from ACV.predict import predict_acv_with_diagnostics
    uploaded_file.seek(0)
    pred,diag=predict_acv_with_diagnostics(uploaded_file,model_dir=ROOT/'ACV'/'models',filename=uploaded_file.name)
    pred['ranked_cars']=pred['ranked_cars'].astype(str).str.replace(' ','',regex=False)
    return pred,raw,diag

def run_rail(uploaded_files):
    rail=ROOT/'Rail_Corrugation'; sys.path.insert(0,str(rail))
    from src.features import load_and_extract
    model=joblib.load(rail/'models'/'rail_best_model_final.joblib')
    rows=[];diag={}
    with tempfile.TemporaryDirectory() as x:
        td=Path(x)
        for up in uploaded_files:
            p=save_uploads([up],td)[0]; feats,names=load_and_extract(str(p)); pred=str(model.predict(feats.reshape(1,-1))[0])
            if pred in {'0','1','2'}: pred={'0':'Normal','1':'Side I','2':'Side II'}[pred]
            rows.append({'file_id':up.name,'prediction':pred});diag[up.name]=dict(zip(names,feats))
    return pd.DataFrame(rows),diag

def csv_bytes(df): return df.to_csv(index=False).encode('utf-8')
def parse_door_times(series):
    def one(v):
        p=str(v).split('-')
        if len(p)==7 and all(q.isdigit() for q in p):
            y,mo,d,h,mi,s,ms=map(int,p);return pd.Timestamp(y,mo,d,h,mi,s,ms*1000)
        return pd.to_datetime(v,errors='coerce')
    return series.map(one)
def acv_temperature_series(df):
    out=pd.DataFrame();
    if 'Time' in df.columns: out['Time']=df['Time']
    cols=[c for c in df.columns if str(c).startswith('Car ') and 'temp' in str(c).lower()]
    for c in cols[:32]: out[c]=pd.to_numeric(df[c],errors='coerce')
    return out

def stress_from_bytes(data):
    df=pd.read_csv(io.BytesIO(data)); nums=df.select_dtypes(include=np.number)
    if nums.empty:return np.array([])
    cols=[c for c in nums if 'stress' in str(c).lower()]; col=cols[0] if cols else nums.columns[-1]
    a=nums[col].to_numpy(float);return a[np.isfinite(a)]
def rainflow_diagnostics(data):
    stress=stress_from_bytes(data)
    if len(stress)<4:return pd.DataFrame()
    sys.path.insert(0,str(ROOT/'SHM'))
    from src.features.physics_features import rainflow_astm
    ranges,means,counts=rainflow_astm(stress)
    return pd.DataFrame({'Stress range':ranges,'Stress amplitude':ranges/2.0,'Mean stress':means,'Cycle count':counts})
