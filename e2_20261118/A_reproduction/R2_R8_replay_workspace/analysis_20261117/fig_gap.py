"""Dot-and-whisker figure of the evaluator generalization gap (consulted family minus other families, standardized)."""
rows=[('h','LLM 권고 · 초안 대비 개선'),
 ('c','GPT · 검증기 선택, α·β·γ로 고름',0.136,0.093,0.184),
 ('c','GPT · 검증기 선택, δ·ε로 고름',0.101,0.053,0.149),
 ('c','Gemini · 검증기 선택, α·β·γ로 고름',0.176,0.134,0.221),
 ('c','Gemini · 검증기 선택, δ·ε로 고름',0.175,0.128,0.221),
 ('f','GPT · 강화학습 참조',0.022,-0.018,0.063),
 ('f','Gemini · 강화학습 참조',0.014,-0.016,0.046),
 ('h','정책 · Heuristic 대비 기업별 몫'),
 ('c','9개 후보에서 α·β·γ로 고름',0.306,0.256,0.357),
 ('c','9개 후보에서 δ·ε로 고름',0.316,0.257,0.373),
 ('c','9개 후보에서 α로 고름',0.436,0.366,0.506),
 ('f','강화학습 정책',-0.004,-0.054,0.046)]
X0,X1,VMIN,VMAX=262,566,-0.1,0.55
def x(v): return X0+(v-VMIN)/(VMAX-VMIN)*(X1-X0)
def build(lit):
    tp='#0b0b0b' if lit else 'var(--text-primary)'; ts='#52514e' if lit else 'var(--text-secondary)'; tm='#898781' if lit else 'var(--text-muted)'
    grid='#e1e0d9' if lit else 'var(--border)'; base='#c3c2b7' if lit else 'var(--border-stronger)'
    cc='#eb6834' if lit else 'var(--series-2, #eb6834)'; cf='#2a78d6' if lit else 'var(--series-1, #2a78d6)'
    y=20; pos=[]
    for r in rows:
        if r[0]=='h': y+=12; pos.append((r,y)); y+=20
        else: pos.append((r,y)); y+=24
    ybot=y+4; H=ybot+40
    o=[f'<svg viewBox="0 0 680 {H}" width="100%" role="img" aria-labelledby="ggt ggd" style="font-family: {"sans-serif" if lit else "var(--font-sans)"}; display: block;">',
       '<title id="ggt">평가모형 일반화 격차: 참고한 평가모형 계열에서 잰 개선과 다른 계열에서 잰 개선의 차이</title>',
       '<desc id="ggd">평가모형으로 고른 방법은 LLM 수준(0.10~0.18)과 정책 수준(0.31~0.44)에서 모두 격차가 크고 95% 신뢰구간이 0을 제외한다. 강화학습 참조와 강화학습 정책은 격차가 0에 가깝고 신뢰구간이 0을 포함한다.</desc>']
    if lit: o.append(f'<rect width="680" height="{H}" fill="#fcfcfb"/>')
    for v in [0.0,0.1,0.2,0.3,0.4,0.5]:
        o.append(f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="14" y2="{ybot}" style="stroke: {base if v==0 else grid}; stroke-width: {1.5 if v==0 else 1};"/>')
        o.append(f'<text x="{x(v):.1f}" y="{ybot+16}" text-anchor="middle" style="fill: {tm}; font-size: 11px; font-variant-numeric: tabular-nums;">{"0" if v==0 else f"+{v:.1f}"}</text>')
    hits=[]
    for r,yy in pos:
        if r[0]=='h':
            o.append(f'<text x="0" y="{yy+4}" style="fill: {tp}; font-size: 13px; font-weight: 500;">{r[1]}</text>'); continue
        k,lab,m,lo,hi=r; c=cc if k=='c' else cf
        o.append(f'<text x="0" y="{yy+4}" style="fill: {ts}; font-size: 12px;">{lab}</text>')
        o.append(f'<line x1="{x(lo):.1f}" x2="{x(hi):.1f}" y1="{yy}" y2="{yy}" style="stroke: {c}; stroke-width: 2; stroke-linecap: round;"/>')
        if k=='c': o.append(f'<circle cx="{x(m):.1f}" cy="{yy}" r="5" style="fill: {c}; stroke: {"#fcfcfb" if lit else "var(--surface-1, #fcfcfb)"}; stroke-width: 1.5;"/>')
        else: o.append(f'<rect x="{x(m)-5:.1f}" y="{yy-5}" width="10" height="10" style="fill: {c}; stroke: {"#fcfcfb" if lit else "var(--surface-1, #fcfcfb)"}; stroke-width: 1.5;"/>')
        o.append(f'<text x="680" y="{yy+4}" text-anchor="end" style="fill: {tp}; font-size: 12px; font-variant-numeric: tabular-nums;">{f"{m:+.3f}".replace("-","−")}</text>')
        hits.append((yy,f'{lab}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]'))
    o.append(f'<text x="{(X0+X1)/2:.1f}" y="{ybot+30}" text-anchor="middle" style="fill: {tm}; font-size: 11px;">격차 = 참고한 계열에서 잰 개선 − 다른 계열에서 잰 개선 (평가모형별 표준화)</text>')
    for yy,t in hits: o.append(f'<rect class="ghit" x="0" y="{yy-12}" width="680" height="24" data-t="{t}" style="fill: transparent;"/>')
    o.append('</svg>'); return '\n'.join(o),H
svg_l,_=build(True); open('fig_gap_preview.svg','w').write(svg_l)
svg,_=build(False)
leg=('<div style="display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 8px; font-size: 12px; color: var(--text-secondary);">'
     '<span style="display: flex; align-items: center; gap: 5px;"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><circle cx="7" cy="7" r="5" style="fill: var(--series-2, #eb6834);"/></svg>평가모형으로 고른 방법</span>'
     '<span style="display: flex; align-items: center; gap: 5px;"><svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><rect x="2" y="2" width="10" height="10" style="fill: var(--series-1, #2a78d6);"/></svg>평가모형 없이 학습한 정책과 그 참조</span>'
     '<span>가로선: 95% 신뢰구간</span></div>')
html=('<h2 class="sr-only">평가모형으로 고른 방법은 참고한 계열에서만 개선이 크게 나타나고, 평가모형 없이 학습한 강화학습 정책과 그 참조는 계열 간 차이가 없다.</h2>\n'
      '<div style="position: relative; padding: 0.5rem 0;">\n'+leg+'\n'+svg+'\n'
      '<div id="ggtip" style="position: absolute; display: none; pointer-events: none; background: var(--surface-2); border: 0.5px solid var(--border-strong); border-radius: var(--radius); padding: 4px 8px; font-size: 12px; color: var(--text-primary); white-space: nowrap; font-variant-numeric: tabular-nums;"></div>\n</div>\n'
      '<script>\n(function(){var tip=document.getElementById("ggtip");var box=tip.parentElement;document.querySelectorAll("rect.ghit").forEach(function(r){r.addEventListener("mousemove",function(ev){var b=box.getBoundingClientRect();tip.textContent=r.getAttribute("data-t");tip.style.display="block";var x=ev.clientX-b.left+12;var w=tip.offsetWidth;if(x+w>b.width){x=ev.clientX-b.left-w-12;}tip.style.left=Math.max(0,x)+"px";tip.style.top=(ev.clientY-b.top-30)+"px";});r.addEventListener("mouseleave",function(){tip.style.display="none";});});})();\n</script>')
open('fig_gap_widget.html','w').write(html); print(len(html))
