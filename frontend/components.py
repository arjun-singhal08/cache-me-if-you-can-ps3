import streamlit as st

def assessment_card(a):
    st.markdown(f"""<div class='assess {a['level']}'><div class='assess-title'>{a['title']}</div><div class='assess-grid'><div><div class='assess-k'>Area</div><div class='assess-v'>{a['area']}</div></div><div><div class='assess-k'>What Senlytics detected</div><div class='assess-v'>{a['detected']}</div></div><div><div class='assess-k'>Recommended follow-up</div><div class='assess-v'>{a['follow']}</div></div></div></div>""", unsafe_allow_html=True)

def intro(title, subtitle):
    st.markdown(f"## {title}"); st.markdown(f"<div class='section-note'>{subtitle}</div>", unsafe_allow_html=True)

def error_box(exc):
    st.error("Analysis could not be completed. Check that the uploaded file matches the expected subsystem format.")
    with st.expander("Technical details for developers"):
        st.code(str(exc))

def chart_config(wheel=False):
    return {"displaylogo":False,"scrollZoom":bool(wheel),"responsive":True,"modeBarButtonsToRemove":["lasso2d","select2d"]}

def technical_details(model, files, elapsed=None, extra=None):
    with st.expander("Engineering / technical details"):
        st.write(f"**Inference:** {model}")
        st.write(f"**Files processed:** {files}")
        if elapsed is not None: st.write(f"**Processing time:** {elapsed:.2f} s")
        if extra:
            for k,v in extra.items(): st.write(f"**{k}:** {v}")
