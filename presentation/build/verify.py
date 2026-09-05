"""Render every slide and check it fits the 1920x1080 stage.

Checks per slide:
  * content taller/wider than the 1920x1080 canvas (overflow)
  * any element whose box escapes the slide bounds
  * text smaller than a comfortable reading size
  * overlapping sibling panels in the main content grid
Writes a PNG per slide to presentation/build/shots/<deck>/ for eyeballing.

Usage: python verify.py [signal|cobalt|swiss|all]
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE.parent
SHOTS = HERE / "shots"

PROBE = """
() => {
  const slide = document.querySelector('.slide.active');
  if (!slide) return {error: 'no active slide'};
  const SB = slide.getBoundingClientRect();
  const issues = [];

  // 1. overall overflow of the slide's own scroll box
  if (slide.scrollHeight > 1081) issues.push(`slide scrollHeight ${slide.scrollHeight} > 1080`);
  if (slide.scrollWidth  > 1921) issues.push(`slide scrollWidth ${slide.scrollWidth} > 1920`);

  // 2. elements escaping the canvas
  const tooSmall = [];
  slide.querySelectorAll('*').forEach(el => {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) return;
    const tag = el.className && typeof el.className === 'string'
        ? el.tagName.toLowerCase() + '.' + el.className.split(' ')[0] : el.tagName.toLowerCase();
    if (r.bottom > SB.bottom + 1.5) issues.push(`${tag} bottom overflows by ${(r.bottom-SB.bottom).toFixed(0)}px`);
    if (r.right  > SB.right  + 1.5) issues.push(`${tag} right overflows by ${(r.right-SB.right).toFixed(0)}px`);
    if (r.left   < SB.left   - 1.5) issues.push(`${tag} left overflows by ${(SB.left-r.left).toFixed(0)}px`);
    // readable text size
    const txt = (el.textContent || '').trim();
    if (txt && el.children.length === 0) {
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (fs < 14) tooSmall.push(`${tag} @ ${fs.toFixed(0)}px`);
    }
    // internal scrolling = hidden content
    if (el.scrollHeight > el.clientHeight + 3 && getComputedStyle(el).overflowY !== 'visible')
      issues.push(`${tag} clips content (${el.scrollHeight} > ${el.clientHeight})`);
  });
  if (tooSmall.length) issues.push('text under 14px: ' + [...new Set(tooSmall)].slice(0,4).join(', '));

  // 3. overlapping siblings inside grid/flex content regions
  const regions = slide.querySelectorAll('.s-body, .step-grid, .fig-grid, .err-grid, .res-grid, .col2, .rel-grid, .mdl-grid, .close-grid, .fnd-row, .ev-grid');
  regions.forEach(reg => {
    const kids = [...reg.children].filter(k => k.getBoundingClientRect().height > 4);
    for (let i = 0; i < kids.length; i++)
      for (let j = i + 1; j < kids.length; j++) {
        const a = kids[i].getBoundingClientRect(), b = kids[j].getBoundingClientRect();
        const ox = Math.min(a.right,b.right) - Math.max(a.left,b.left);
        const oy = Math.min(a.bottom,b.bottom) - Math.max(a.top,b.top);
        if (ox > 6 && oy > 6) issues.push(`overlap: ${kids[i].className||kids[i].tagName} / ${kids[j].className||kids[j].tagName} (${ox.toFixed(0)}x${oy.toFixed(0)}px)`);
      }
  });
  // 4. decorative overlays must not sit on top of readable content.
  //    A full-bleed .deco is a background wash (e.g. a radial gradient), not an overlay —
  //    only sub-region decorations (glitch columns, colour blocks) can actually obscure text.
  const deco = slide.querySelector('.deco');
  const decoArea = deco ? deco.getBoundingClientRect().width * deco.getBoundingClientRect().height : 0;
  if (deco && decoArea < 0.8 * SB.width * SB.height) {
    const d = deco.getBoundingClientRect();
    slide.querySelectorAll('h1,h2,h3,p,td,th,li,.g-row,.g-cap,.stat,.chip,.cover-names').forEach(el => {
      const r = el.getBoundingClientRect();
      if (r.width === 0) return;
      const ox = Math.min(d.right,r.right) - Math.max(d.left,r.left);
      const oy = Math.min(d.bottom,r.bottom) - Math.max(d.top,r.top);
      if (ox > 4 && oy > 4)
        issues.push(`decoration covers ${el.tagName.toLowerCase()}.${(el.className||'').split(' ')[0]}`);
    });
  }

  return {issues, h: slide.scrollHeight, w: slide.scrollWidth};
}
"""


def verify(deck):
    path = OUT / f"deck-{deck}.html"
    if not path.exists():
        print(f"  deck-{deck}.html not found — skipping")
        return 0
    shots = SHOTS / deck
    shots.mkdir(parents=True, exist_ok=True)
    total_issues = 0

    with sync_playwright() as p:
        b = p.chromium.launch()
        # 1920x1080 so the stage renders 1:1 and measurements are exact
        pg = b.new_page(viewport={"width": 1920, "height": 1080})
        pg.goto(path.as_uri())
        pg.wait_for_timeout(2200)          # webfonts
        n = pg.evaluate("() => document.querySelectorAll('.slide').length")
        print(f"\n=== {deck.upper()} — {n} slides ===")

        for i in range(n):
            pg.evaluate(f"() => deck.show({i})")
            pg.wait_for_timeout(430)
            r = pg.evaluate(PROBE)
            issues = r.get("issues", [])
            # de-duplicate noisy repeats
            seen, uniq = set(), []
            for it in issues:
                k = it.split("(")[0]
                if k not in seen:
                    seen.add(k); uniq.append(it)
            tag = "OK  " if not uniq else "FAIL"
            print(f"  [{tag}] slide {i+1:02d}  {r.get('w')}x{r.get('h')}")
            for it in uniq[:6]:
                print(f"         - {it}")
            total_issues += len(uniq)
            pg.screenshot(path=str(shots / f"{i+1:02d}.png"))

        # phone check: stage must letterbox, never reflow
        pg.set_viewport_size({"width": 390, "height": 844})
        pg.wait_for_timeout(500)
        box = pg.evaluate("""() => {const s=document.querySelector('.deck-stage').getBoundingClientRect();
            return {w:Math.round(s.width),h:Math.round(s.height),r:+(s.width/s.height).toFixed(3)};}""")
        ok = abs(box["r"] - 1.7778) < 0.02
        print(f"  [{'OK  ' if ok else 'FAIL'}] phone 390x844 -> stage {box['w']}x{box['h']} ratio {box['r']}")
        if not ok:
            total_issues += 1
        b.close()
    return total_issues


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    decks = ["signal", "cobalt", "swiss"] if which == "all" else [which]
    bad = sum(verify(d) for d in decks)
    print(f"\n{'ALL CLEAN' if bad == 0 else str(bad) + ' ISSUE(S) FOUND'}")
