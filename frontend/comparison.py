import pandas as pd
import plotly.graph_objects as go

def rail_compare(pred, diag, left, right):
    names=[]; a=[]; b=[]
    da,db=diag.get(left,{}),diag.get(right,{})
    common=[k for k in da if k in db and ("rms" in k.lower() or "energy" in k.lower())]
    for k in common[:10]:
        try: names.append(k.replace("_"," ").title()); a.append(float(da[k])); b.append(float(db[k]))
        except: pass
    fig=go.Figure(); fig.add_bar(name=left,x=names,y=a); fig.add_bar(name=right,x=names,y=b); fig.update_layout(barmode="group",title="Bilateral vibration feature comparison")
    return fig

def shm_compare(pred, left, right):
    sub=pred[pred.file_id.isin([left,right])]
    fig=go.Figure(go.Bar(x=sub.file_id,y=sub.prediction)); fig.update_layout(title="Cumulative fatigue damage comparison",yaxis_title="Predicted damage D")
    return fig

def door_summary_compare(items, left, right):
    rows=[]
    for name in [left,right]:
        p=items[name]; ab=int((p.prediction=="Abnormal resistance").sum()); total=len(p)
        rows.append({"Recording":name,"Cycles":total,"Abnormal":ab,"Flagged rate (%)":round(100*ab/total,1) if total else 0})
    return pd.DataFrame(rows)
