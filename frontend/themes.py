THEMES = {
    "Operations Dark": {"bg":"#071019","bg2":"#050b11","panel":"#0d1822","panel2":"#101e2a","line":"#203342","text":"#edf6fb","muted":"#8ca2b3","accent":"#2dd4bf","accent2":"#38bdf8","green":"#34d399","amber":"#fbbf24","red":"#fb7185","plot":"plotly_dark"},
    "Operations Light": {"bg":"#f5f8fa","bg2":"#edf3f6","panel":"#ffffff","panel2":"#f7fafb","line":"#d6e0e6","text":"#10212c","muted":"#5d7280","accent":"#087f72","accent2":"#087ca7","green":"#087f5b","amber":"#b36b00","red":"#c92a45","plot":"plotly_white"},
    "Midnight Teal": {"bg":"#061418","bg2":"#031012","panel":"#0a2025","panel2":"#0d292e","line":"#174047","text":"#e8ffff","muted":"#8eb9bd","accent":"#00d7c0","accent2":"#4bd7ff","green":"#5ce0a2","amber":"#ffc857","red":"#ff6b7a","plot":"plotly_dark"},
    "High Contrast": {"bg":"#000000","bg2":"#000000","panel":"#0a0a0a","panel2":"#111111","line":"#777777","text":"#ffffff","muted":"#d1d1d1","accent":"#00ffff","accent2":"#7dd3fc","green":"#00ff88","amber":"#ffdd00","red":"#ff5268","plot":"plotly_dark"},
}

def css(t):
    return f"""
<style>
:root{{--bg:{t['bg']};--bg2:{t['bg2']};--panel:{t['panel']};--panel2:{t['panel2']};--line:{t['line']};--text:{t['text']};--muted:{t['muted']};--accent:{t['accent']};--accent2:{t['accent2']};--green:{t['green']};--amber:{t['amber']};--red:{t['red']};}}
.stApp{{background:radial-gradient(circle at 88% -5%,color-mix(in srgb,var(--accent) 11%,transparent),transparent 30%),linear-gradient(180deg,var(--bg),var(--bg2));color:var(--text)}}
.block-container{{max-width:1480px;padding-top:1.1rem;padding-bottom:4rem}} #MainMenu,footer,header{{visibility:hidden}}
h1,h2,h3{{letter-spacing:-.025em;color:var(--text)}} p,label,[data-testid="stMarkdownContainer"]{{color:var(--text)}}
[data-testid="stTabs"] [data-baseweb="tab-list"]{{gap:.35rem;background:var(--panel);border:1px solid var(--line);padding:.4rem;border-radius:15px}}
[data-testid="stTabs"] [data-baseweb="tab"]{{border-radius:11px;padding:.7rem 1rem;font-weight:750;color:var(--muted)}}
[data-testid="stFileUploaderDropzone"]{{background:var(--panel2);border:1px dashed color-mix(in srgb,var(--accent) 45%,var(--line));border-radius:18px}}
[data-testid="stMetric"]{{background:linear-gradient(145deg,var(--panel2),var(--panel));border:1px solid var(--line);padding:1rem 1.1rem;border-radius:16px}}
[data-testid="stMetricLabel"]{{color:var(--muted)}} [data-testid="stMetricValue"]{{color:var(--text)}}
.stButton>button,.stDownloadButton>button{{border-radius:11px;border:1px solid var(--line);font-weight:750}}
.hero{{padding:1.1rem 1.35rem 1.25rem;border:1px solid var(--line);border-radius:22px;background:linear-gradient(120deg,var(--panel2),var(--panel));margin-bottom:.8rem;box-shadow:0 18px 55px rgba(0,0,0,.12)}}
.brand{{font-size:2.35rem;font-weight:900;letter-spacing:-.055em;line-height:1;color:var(--text)}} .brand-dot{{color:var(--accent)}}
.tag{{color:var(--muted);margin-top:.45rem}} .status{{display:inline-block;color:var(--green);font-size:.78rem;font-weight:800;border:1px solid color-mix(in srgb,var(--green) 40%,var(--line));background:color-mix(in srgb,var(--green) 9%,var(--panel));padding:.32rem .62rem;border-radius:999px}}
.section-note{{color:var(--muted);margin-top:-.45rem;margin-bottom:1rem}} .soft{{color:var(--muted);font-size:.88rem}}
.assess{{border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:16px;background:var(--panel);padding:1rem 1.15rem;margin:.7rem 0 1rem}} .assess.red{{border-left-color:var(--red)}} .assess.green{{border-left-color:var(--green)}} .assess.amber{{border-left-color:var(--amber)}}
.assess-title{{font-size:1.1rem;font-weight:850;margin-bottom:.55rem}} .assess-grid{{display:grid;grid-template-columns:1fr 1.7fr 1.7fr;gap:1rem}} .assess-k{{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);font-weight:800}} .assess-v{{margin-top:.2rem;color:var(--text)}}
.rank{{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:12px;padding:.58rem .82rem;margin:.18rem;font-weight:850}} .rank.top{{border-color:var(--amber);color:var(--amber)}}
.footerx{{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem;text-align:center}}
@media(max-width:800px){{.assess-grid{{grid-template-columns:1fr}}.brand{{font-size:1.9rem}}}}
</style>"""
