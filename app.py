from __future__ import annotations
import io,time
from datetime import datetime
import numpy as np,pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from frontend.themes import THEMES,css
from frontend.components import assessment_card,intro,error_box,chart_config,technical_details
from frontend.translators import door_assessment,rail_assessment,acv_assessment,shm_assessment
from frontend.comparison import rail_compare,shm_compare,door_summary_compare
from frontend_utils import run_door,run_rail,run_shm,run_acv,csv_bytes,parse_door_times,acv_temperature_series,rainflow_diagnostics

st.set_page_config(page_title='Senlytics | Intelligent Train Condition Monitoring',page_icon='◉',layout='wide',initial_sidebar_state='collapsed')
if 'theme' not in st.session_state: st.session_state.theme='Operations Dark'
if 'history' not in st.session_state: st.session_state.history=[]

# Top controls before CSS/hero
c0,c1=st.columns([5,1.25])
with c1:
    theme=st.selectbox('Theme',list(THEMES),index=list(THEMES).index(st.session_state.theme),label_visibility='collapsed')
st.session_state.theme=theme; T=THEMES[theme]; st.markdown(css(T),unsafe_allow_html=True)

st.markdown("""<div class='hero'><div style='display:flex;justify-content:space-between;gap:1rem;align-items:flex-start'><div><div class='brand'>Senlytics<span class='brand-dot'>.</span></div><div class='tag'>Intelligent Train Condition Monitoring · Every signal. One intelligent view.</div></div><div><span class='status'>● SYSTEM ONLINE</span><div class='soft' style='text-align:right;margin-top:.45rem'>4 subsystems · inference ready</div></div></div></div>""",unsafe_allow_html=True)

wheel_zoom=st.toggle('Enable mouse-wheel zoom on charts',value=False,help='Turn this on when closely inspecting charts. Leave it off for easier page scrolling.')
PLOT_CFG=chart_config(wheel_zoom)

def style_fig(fig):
    fig.update_layout(template=T['plot'],paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color=T['text'],legend_title_text='',margin=dict(l=20,r=20,t=55,b=25))
    return fig

def add_history(subsystem,filename,result):
    st.session_state.history.insert(0,{'Time':datetime.now().strftime('%H:%M'),'Subsystem':subsystem,'File':filename,'Result':result})
    st.session_state.history=st.session_state.history[:12]

def success(msg,elapsed): st.success(f'✓ {msg} · completed in {elapsed:.2f} s')

def raw_door_chart(raw,title):
    if raw is None or raw.empty:return None
    xcol='Datetime' if 'Datetime' in raw.columns else raw.columns[0]
    x=parse_door_times(raw[xcol])
    wanted=[]
    for needle in ['current','position','voltage','electromotive','back']:
        hit=next((c for c in raw.columns if needle in str(c).lower()),None)
        if hit and hit not in wanted:wanted.append(hit)
    fig=go.Figure()
    for c in wanted[:4]: fig.add_trace(go.Scatter(x=x,y=pd.to_numeric(raw[c],errors='coerce'),name=str(c),mode='lines'))
    fig.update_layout(title=title,xaxis_title='Time',yaxis_title='Sensor value',hovermode='x unified')
    return style_fig(fig)

def door_timeline(pred):
    p=pred.copy();p['start']=parse_door_times(p.start_time);p['end']=parse_door_times(p.end_time)
    p['Cycle']=[f'Cycle {i+1}' for i in range(len(p))]
    colors={'Normal':T['green'],'Abnormal resistance':T['red']}
    fig=px.timeline(p,x_start='start',x_end='end',y='Cycle',color='prediction',color_discrete_map=colors,hover_data=['start_time','end_time','prediction'])
    fig.update_yaxes(visible=False);fig.update_layout(title='Cycle segmentation timeline',height=330)
    return style_fig(fig)

shm_tab,rail_tab,door_tab,acv_tab,history_tab=st.tabs(['🏗 Structural Health','〰 Rail Corrugation','🚪 Passenger Doors','❄ ACV Refrigerant','↺ Recent Analyses'])

with shm_tab:
    intro('Structural Health Monitoring','Estimate cumulative fatigue damage from dynamic stress telemetry. Compare inspections and inspect genuine Rainflow cycle evidence.')
    uploads=st.file_uploader('Upload one or more dynamic-stress CSV files',type=['csv'],accept_multiple_files=True,key='shm')
    if uploads:
        st.caption(f'{len(uploads)} file(s) selected · CSV telemetry')
        if st.button('Analyze structural health',type='primary',key='shmgo'):
            t=time.perf_counter()
            try:
                with st.spinner('Validating stress telemetry → extracting physics features → estimating fatigue damage…'):
                    pred=run_shm(uploads)
                st.session_state.shm_pred=pred;st.session_state.shm_bytes={u.name:u.getvalue() for u in uploads};st.session_state.shm_elapsed=time.perf_counter()-t
                for _,r in pred.iterrows(): add_history('SHM',r.file_id,f'D={float(r.prediction):.5f}')
            except Exception as e:error_box(e)
    if 'shm_pred' in st.session_state:
        p=st.session_state.shm_pred; success(f'{len(p)} structural file(s) analyzed',st.session_state.shm_elapsed)
        a,b,c=st.columns(3);a.metric('Files analyzed',len(p));b.metric('Mean cumulative damage',f'{p.prediction.mean():.5f}');c.metric('Highest predicted damage',f'{p.prediction.max():.5f}')
        selected=st.selectbox('Inspect result',p.file_id.tolist(),key='shmsel'); row=p[p.file_id==selected].iloc[0]; assessment_card(shm_assessment(float(row.prediction)))
        rf=rainflow_diagnostics(st.session_state.shm_bytes.get(selected,b''))
        if not rf.empty:
            fig=px.histogram(rf,x='Stress amplitude',y='Cycle count',nbins=32,histfunc='sum',title=f'Rainflow stress-amplitude distribution · {selected}')
            st.plotly_chart(style_fig(fig),use_container_width=True,config=PLOT_CFG)
        st.dataframe(p,use_container_width=True,hide_index=True)
        if len(p)>=2:
            with st.expander('⚖ Compare analyses',expanded=False):
                x,y=st.columns(2);left=x.selectbox('Baseline',p.file_id,key='shml');right=y.selectbox('Comparison',p.file_id,index=1,key='shmr')
                if left!=right: st.plotly_chart(style_fig(shm_compare(p,left,right)),use_container_width=True,config=PLOT_CFG)
        technical_details('Hybrid physics + ML fatigue regression',len(p),st.session_state.shm_elapsed,{'Selected cumulative damage':f'{float(row.prediction):.6f}'})
        st.download_button('↓ Download shm_predictions.csv',csv_bytes(p),'shm_predictions.csv','text/csv',use_container_width=True)

with rail_tab:
    intro('Rail Corrugation','Classify axle-box vibration as Normal, Side I or Side II corrugation, with bilateral evidence and direct file-to-file comparison.')
    uploads=st.file_uploader('Upload one or more axle-box vibration CSV files',type=['csv'],accept_multiple_files=True,key='rail')
    if uploads and st.button('Analyze rail condition',type='primary',key='railgo'):
        t=time.perf_counter()
        try:
            with st.spinner('Validating vibration channels → extracting bilateral features → classifying corrugation…'): p,d=run_rail(uploads)
            st.session_state.rail_pred=p;st.session_state.rail_diag=d;st.session_state.rail_elapsed=time.perf_counter()-t
            for _,r in p.iterrows():add_history('Rail',r.file_id,r.prediction)
        except Exception as e:error_box(e)
    if 'rail_pred' in st.session_state:
        p,d=st.session_state.rail_pred,st.session_state.rail_diag; success(f'{len(p)} rail file(s) analyzed',st.session_state.rail_elapsed)
        vc=p.prediction.value_counts();a,b,c=st.columns(3);a.metric('Files analyzed',len(p));b.metric('Normal',int(vc.get('Normal',0)));c.metric('Corrugation alerts',int(len(p)-vc.get('Normal',0)))
        sel=st.selectbox('Inspect recording',p.file_id.tolist(),key='railsel');cls=str(p[p.file_id==sel].prediction.iloc[0]);assessment_card(rail_assessment(cls))
        dd=d.get(sel,{}); keys=[k for k in dd if ('rms' in k.lower() or 'energy' in k.lower())]
        if keys:
            vals={k.replace('_',' ').title():float(dd[k]) for k in keys[:10]};fig=px.bar(x=list(vals),y=list(vals.values()),labels={'x':'Bilateral feature','y':'Magnitude'},title=f'Bilateral vibration indicators · {sel}')
            st.plotly_chart(style_fig(fig),use_container_width=True,config=PLOT_CFG)
        st.dataframe(p,use_container_width=True,hide_index=True)
        if len(p)>=2:
            with st.expander('⚖ Compare analyses',expanded=True):
                x,y=st.columns(2);left=x.selectbox('Recording A',p.file_id,key='raill');right=y.selectbox('Recording B',p.file_id,index=1,key='railr')
                if left!=right:
                    st.plotly_chart(style_fig(rail_compare(p,d,left,right)),use_container_width=True,config=PLOT_CFG)
                    ca=str(p[p.file_id==left].prediction.iloc[0]);cb=str(p[p.file_id==right].prediction.iloc[0]);st.info(f'**Diagnosis comparison:** {left}: {ca}  ↔  {right}: {cb}')
        technical_details('Final calibrated voting ensemble + physics-grounded features',len(p),st.session_state.rail_elapsed)
        st.download_button('↓ Download rail_predictions.csv',csv_bytes(p),'rail_predictions.csv','text/csv',use_container_width=True)

with door_tab:
    intro('Passenger Door Monitoring','Detect door cycles in continuous controller telemetry and identify abnormal mechanical resistance. Upload multiple recordings to compare operating behaviour.')
    uploads=st.file_uploader('Upload one or more continuous Door telemetry CSV files',type=['csv'],accept_multiple_files=True,key='door')
    if uploads and st.button('Analyze door telemetry',type='primary',key='doorgo'):
        t=time.perf_counter();items={};raws={}
        try:
            with st.spinner('Segmenting cycles → extracting motor/position features → classifying resistance…'):
                for u in uploads:
                    p,r=run_door(u);items[u.name]=p;raws[u.name]=r
            st.session_state.door_items=items;st.session_state.door_raws=raws;st.session_state.door_elapsed=time.perf_counter()-t
            for name,p in items.items():
                ab=int((p.prediction=='Abnormal resistance').sum());add_history('Door',name,f'{ab}/{len(p)} flagged')
        except Exception as e:error_box(e)
    if 'door_items' in st.session_state:
        items=st.session_state.door_items;raws=st.session_state.door_raws;success(f'{len(items)} door recording(s) analyzed',st.session_state.door_elapsed)
        sel=st.selectbox('Inspect recording',list(items),key='doorsel');p=items[sel];raw=raws[sel];ab=int((p.prediction=='Abnormal resistance').sum());normal=len(p)-ab
        a,b,c,d=st.columns(4);a.metric('Cycles detected',len(p));b.metric('Normal',normal);c.metric('Abnormal resistance',ab);d.metric('Flagged-cycle rate',f'{100*ab/len(p):.1f}%' if len(p) else '—')
        assessment_card(door_assessment(ab,len(p)))
        st.plotly_chart(door_timeline(p),use_container_width=True,config=PLOT_CFG)
        filt=st.radio('Cycle table',['Abnormal only','All cycles'],horizontal=True,key='doorfilter');show=p[p.prediction=='Abnormal resistance'] if filt=='Abnormal only' else p
        st.dataframe(show,use_container_width=True,hide_index=True)
        with st.expander('Why was this flagged? · Sensor evidence',expanded=False):
            fig=raw_door_chart(raw,f'Electrical & position telemetry · {sel}')
            if fig:st.plotly_chart(fig,use_container_width=True,config=PLOT_CFG)
            st.caption('Use the timeline and raw telemetry together to inspect current, voltage, back-EMF and position behaviour. Senlytics does not infer a specific physical cause from these signals alone.')
        if len(items)>=2:
            with st.expander('⚖ Compare recordings',expanded=True):
                names=list(items);x,y=st.columns(2);left=x.selectbox('Recording A',names,key='doorl');right=y.selectbox('Recording B',names,index=1,key='doorr')
                if left!=right:
                    comp=door_summary_compare(items,left,right);st.dataframe(comp,use_container_width=True,hide_index=True)
                    fig=px.bar(comp,x='Recording',y='Flagged rate (%)',color='Recording',title='Abnormal-resistance rate comparison');st.plotly_chart(style_fig(fig),use_container_width=True,config=PLOT_CFG)
        technical_details('Temporal segmentation + resistance classifier',len(items),st.session_state.door_elapsed,{'Selected recording':sel})
        st.download_button('↓ Download door_predictions.csv',csv_bytes(p),'door_predictions.csv','text/csv',use_container_width=True)

with acv_tab:
    intro('ACV Refrigerant Monitoring','Rank cars by refrigerant-leak likelihood using genuine ensemble inference and compare thermal behaviour across consists.')
    uploads=st.file_uploader('Upload one or more consist workbooks',type=['xlsx'],accept_multiple_files=True,key='acv')
    if uploads and st.button('Analyze ACV consist',type='primary',key='acvgo'):
        t=time.perf_counter();rows=[];raws={};diags={}
        try:
            with st.spinner('Parsing consist telemetry → extracting cross-car features → ranking refrigerant-leak likelihood…'):
                for u in uploads:
                    p,raw,d=run_acv(u);rows.append(p);raws[u.name]=raw;diags[u.name]=d
            out=pd.concat(rows,ignore_index=True);st.session_state.acv_pred=out;st.session_state.acv_raws=raws;st.session_state.acv_diag=diags;st.session_state.acv_elapsed=time.perf_counter()-t
            for _,r in out.iterrows():add_history('ACV',r.file_id,f'Primary: Car {str(r.ranked_cars).split("|")[0]}')
        except Exception as e:error_box(e)
    if 'acv_pred' in st.session_state:
        p=st.session_state.acv_pred;raws=st.session_state.acv_raws;success(f'{len(p)} consist workbook(s) analyzed with live models',st.session_state.acv_elapsed)
        sel=st.selectbox('Inspect consist',p.file_id.tolist(),key='acvsel');ranked=str(p[p.file_id==sel].ranked_cars.iloc[0]).replace(' ','');cars=ranked.split('|')
        a,b,c=st.columns(3);a.metric('Cars ranked',len(cars));b.metric('Primary suspect',f'Car {cars[0]}');c.metric('Ensemble models','5')
        assessment_card(acv_assessment(ranked))
        st.markdown('**Suspect ranking · most likely → least likely**');st.markdown(''.join([f"<span class='rank {'top' if i==0 else ''}'>#{i+1} · {car}</span>" for i,car in enumerate(cars)]),unsafe_allow_html=True)
        temp=acv_temperature_series(raws[sel])
        if not temp.empty:
            x=temp['Time'] if 'Time' in temp else temp.index;fig=go.Figure()
            for col in [c for c in temp.columns if c!='Time'][:24]:
                width=3 if f'Car {cars[0]}' in str(col) else 1;opacity=1 if width==3 else .45;fig.add_trace(go.Scatter(x=x,y=temp[col],name=str(col),mode='lines',line={'width':width},opacity=opacity))
            fig.update_layout(title=f'Thermal telemetry · primary suspect Car {cars[0]} highlighted',hovermode='x unified');st.plotly_chart(style_fig(fig),use_container_width=True,config=PLOT_CFG)
        st.dataframe(p,use_container_width=True,hide_index=True)
        if len(p)>=2:
            with st.expander('⚖ Compare consists',expanded=True):
                x,y=st.columns(2);left=x.selectbox('Consist A',p.file_id,key='acvl');right=y.selectbox('Consist B',p.file_id,index=1,key='acvr')
                if left!=right:
                    ra=str(p[p.file_id==left].ranked_cars.iloc[0]);rb=str(p[p.file_id==right].ranked_cars.iloc[0]);st.info(f'**Primary suspect:** {left} → Car {ra.split("|")[0]}  ·  {right} → Car {rb.split("|")[0]}')
                    ca,cb=st.columns(2);ca.write('**Ranking A**');ca.code(ra);cb.write('**Ranking B**');cb.code(rb)
        technical_details('5-model ACV rank ensemble',len(p),st.session_state.acv_elapsed,{'Selected ranking':ranked})
        st.download_button('↓ Download acv_predictions.csv',csv_bytes(p),'acv_predictions.csv','text/csv',use_container_width=True)

with history_tab:
    intro('Recent Analyses','Session-only history for quick operator reference. Nothing is persisted after the session ends.')
    if st.session_state.history:
        h=pd.DataFrame(st.session_state.history);st.dataframe(h,use_container_width=True,hide_index=True)
        if st.button('Clear session history'):st.session_state.history=[];st.rerun()
    else:st.info('No analyses have been run in this session yet.')

st.markdown("<div class='footerx'><b>Senlytics</b> · Intelligent Train Condition Monitoring<br>Cache Me If You Can · LTA NebulaX 2026</div>",unsafe_allow_html=True)
