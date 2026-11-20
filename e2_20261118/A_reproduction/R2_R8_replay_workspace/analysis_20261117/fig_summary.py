"""Build the summary figure: (left) reference increment on 7 evaluators, (right) level differences against the Heuristic.
Writes a literal-colour SVG for visual checking and an HTML fragment that uses theme variables."""
import json, sys
J = json.load(open('e17_summary.json'))
EV = ['α', 'β', 'γ', 'δ1', 'δ2', 'ε1', 'ε2']
A = [('GPT', 1, J['GPT C6-E − GPT C4R']), ('Gemini', 2, J['Gemini C6-E − Gemini C4R'])]
B = [('강화학습 정책', 3, J['RL − OE']), ('GPT', 1, J['GPT C4 − OE']), ('Gemini', 2, J['Gemini C4 − OE'])]
W, H = 680, 352
Y0, Y1, VMAX, VMIN = 44, 284, 0.45, -0.15
def y(v): return Y0 + (VMAX - v) / (VMAX - VMIN) * (Y1 - Y0)
PA = (46, 314); PB = (392, 672)
def build(mode):
    lit = mode == 'literal'
    C = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}
    col = lambda s: C[s] if lit else f'var(--series-{s}, {C[s]})'
    tp = '#0b0b0b' if lit else 'var(--text-primary)'; ts = '#52514e' if lit else 'var(--text-secondary)'; tm = '#898781' if lit else 'var(--text-muted)'
    grid = '#e1e0d9' if lit else 'var(--border)'; base = '#c3c2b7' if lit else 'var(--border-stronger)'
    o = []; hits = []
    o.append(f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-labelledby="fgt fgd" style="font-family: {"sans-serif" if lit else "var(--font-sans)"}; display: block;">')
    o.append('<title id="fgt">일곱 평가모형에서 본 참조정보 추가의 증분과 Heuristic 대비 수준 차이</title>')
    o.append('<desc id="fgd">왼쪽: 강화학습 참조를 받은 수정에서 자기 재검토를 뺀 정책가치 차이는 두 모델, 일곱 평가모형 14칸 모두 0보다 크다. 오른쪽: 강화학습 정책과 LLM 직접 작성에서 Heuristic을 뺀 차이는 β에서 음수가 되는 등 평가모형에 따라 부호가 바뀐다.</desc>')
    if lit: o.append(f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>')
    def panel(x0, x1, title, series, ticks_left):
        sw = (x1 - x0) / 7
        for v in [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
            yy = y(v); o.append(f'<line x1="{x0}" x2="{x1}" y1="{yy:.1f}" y2="{yy:.1f}" style="stroke: {base if v == 0 else grid}; stroke-width: {1.5 if v == 0 else 1};"/>')
            o.append(f'<text x="{ticks_left}" y="{yy + 4:.1f}" text-anchor="end" style="fill: {tm}; font-size: 11px; font-variant-numeric: tabular-nums;">{("+" if v > 0 else "") + ("0" if v == 0 else f"{v:.1f}")}</text>')
        o.append(f'<text x="{x0}" y="26" style="fill: {tp}; font-size: 13px; font-weight: 500;">{title}</text>')
        k = len(series); off = {2: [-8, 8], 3: [-11, 0, 11]}[k]
        for j, e in enumerate(EV):
            cx = x0 + sw * (j + .5)
            o.append(f'<text x="{cx:.1f}" y="{Y1 + 20}" text-anchor="middle" style="fill: {ts}; font-size: 12px;">{e}</text>')
            for (nm, s, d), dx in zip(series, off):
                m, lo, hi = d[j]; sig = lo > 0 or hi < 0; x = cx + dx; ym, yl, yh = y(m), y(lo), y(hi); c = col(s); r = 4.5
                if sig: o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{yh:.1f}" y2="{yl:.1f}" style="stroke: {c}; stroke-width: 1.5;"/>')
                else:
                    if ym - r - yh > 0.5: o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{yh:.1f}" y2="{ym - r:.1f}" style="stroke: {c}; stroke-width: 1.5;"/>')
                    if yl - (ym + r) > 0.5: o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{ym + r:.1f}" y2="{yl:.1f}" style="stroke: {c}; stroke-width: 1.5;"/>')
                st = f'fill: {c};' if sig else f'fill: none; stroke: {c}; stroke-width: 1.5;'
                rr = r if sig else r - .75
                if s == 1: o.append(f'<circle cx="{x:.1f}" cy="{ym:.1f}" r="{rr}" style="{st}"/>')
                elif s == 2: o.append(f'<rect x="{x - rr:.1f}" y="{ym - rr:.1f}" width="{2 * rr}" height="{2 * rr}" style="{st}"/>')
                else:
                    q = rr + 1.2; o.append(f'<path d="M{x:.1f} {ym - q:.1f} L{x + q:.1f} {ym:.1f} L{x:.1f} {ym + q:.1f} L{x - q:.1f} {ym:.1f} Z" style="{st}"/>')
                hw = (sw / k) if k == 3 else 16
                hits.append((x - hw / 2, min(yh, ym - 8) - 4, hw, max(yl, ym + 8) - min(yh, ym - 8) + 8, f'{e} · {nm}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}]'))
        for a, b, lab in [(0, 3, '논문의 평가모형'), (3, 7, '변수가 다른 평가모형')]:
            xa, xb = x0 + sw * a + 6, x0 + sw * b - 6
            o.append(f'<line x1="{xa:.1f}" x2="{xb:.1f}" y1="{Y1 + 30}" y2="{Y1 + 30}" style="stroke: {base}; stroke-width: 1;"/>')
            o.append(f'<text x="{(xa + xb) / 2:.1f}" y="{Y1 + 46}" text-anchor="middle" style="fill: {tm}; font-size: 11px;">{lab}</text>')
    panel(*PA, '참조정보 추가의 증분', A, PA[0] - 8)
    panel(*PB, 'Heuristic 대비 수준 차이', B, PB[0] - 8)
    for (hx, hy, hw, hh, t) in hits:
        o.append(f'<rect class="hit" x="{hx:.1f}" y="{hy:.1f}" width="{hw:.1f}" height="{hh:.1f}" data-t="{t}" style="fill: transparent;"/>')
    o.append('</svg>')
    return '\n'.join(o)
open('fig_summary_preview.svg', 'w').write(build('literal'))
svg = build('theme')
def mk(s, shape, filled=True):
    c = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}[s]; v = f'var(--series-{s}, {c})'
    st = f'fill: {v};' if filled else f'fill: none; stroke: {v}; stroke-width: 1.5;'
    g = {'c': f'<circle cx="7" cy="7" r="4.5" style="{st}"/>', 'r': f'<rect x="2.5" y="2.5" width="9" height="9" style="{st}"/>', 'd': f'<path d="M7 1.3 L12.7 7 L7 12.7 L1.3 7 Z" style="{st}"/>'}[shape]
    return f'<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" style="flex: none;">{g}</svg>'
leg = ('<div style="display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 6px; font-size: 12px; color: var(--text-secondary);">'
       f'<span style="display: flex; align-items: center; gap: 5px;">{mk(1, "c")}GPT</span>'
       f'<span style="display: flex; align-items: center; gap: 5px;">{mk(2, "r")}Gemini</span>'
       f'<span style="display: flex; align-items: center; gap: 5px;">{mk(3, "d")}강화학습 정책</span>'
       f'<span style="display: flex; align-items: center; gap: 5px;">{mk(1, "c", False)}빈 표식: 95% 신뢰구간이 0을 포함</span>'
       '<span>세로선: 95% 신뢰구간 · 단위: 정책가치 차이</span></div>')
html = ('<h2 class="sr-only">일곱 평가모형에서 참조정보 추가의 증분은 14칸 모두 양수이고, Heuristic 대비 수준 차이는 평가모형에 따라 부호가 바뀐다.</h2>\n'
        '<div style="position: relative; padding: 0.5rem 0;">\n' + leg + '\n' + svg + '\n'
        '<div id="fgtip" style="position: absolute; display: none; pointer-events: none; background: var(--surface-2); border: 0.5px solid var(--border-strong); border-radius: var(--radius); padding: 4px 8px; font-size: 12px; color: var(--text-primary); white-space: nowrap; font-variant-numeric: tabular-nums;"></div>\n</div>\n'
        '<script>\n(function(){var tip=document.getElementById("fgtip");var box=tip.parentElement;document.querySelectorAll("rect.hit").forEach(function(r){r.addEventListener("mousemove",function(ev){var b=box.getBoundingClientRect();tip.textContent=r.getAttribute("data-t");tip.style.display="block";var x=ev.clientX-b.left+12;var w=tip.offsetWidth;if(x+w>b.width){x=ev.clientX-b.left-w-12;}tip.style.left=Math.max(0,x)+"px";tip.style.top=(ev.clientY-b.top-30)+"px";});r.addEventListener("mouseleave",function(){tip.style.display="none";});});})();\n</script>')
open('fig_summary_widget.html', 'w').write(html)
print(len(html), 'chars')
