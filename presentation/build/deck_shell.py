"""Shared deck shell: the mandatory fixed-stage CSS and the presentation controller JS.

BASE_CSS is the full contents of the frontend-slides `viewport-base.css` and must be
included verbatim in every generated deck.
"""

BASE_CSS = """
/* ===========================================
   FIXED 16:9 STAGE: MANDATORY BASE STYLES
   Slides are authored at 1920x1080 and scaled as a whole.
   =========================================== */

/* 1. Lock the browser viewport */
html,
body {
    width: 100%;
    height: 100%;
    margin: 0;
    overflow: hidden;
    background: var(--stage-bg, #000);
}

/* 2. Full-window deck viewport */
.deck-viewport {
    position: fixed;
    inset: 0;
    overflow: hidden;
    background: var(--stage-bg, #000);
}

/* 3. Fixed 16:9 design canvas.
   JavaScript sets transform: translate(...) scale(...). */
.deck-stage {
    position: absolute;
    left: 0;
    top: 0;
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    transform-origin: 0 0;
    background: var(--slide-bg, #fff);
}

/* 4. Slides stack inside the fixed stage. */
.slide {
    position: absolute;
    inset: 0;
    width: 1920px;
    height: 1080px;
    overflow: hidden;
    display: block;
    visibility: hidden;
    opacity: 0;
    pointer-events: none;
    background: var(--slide-bg, #fff);
}

.slide.active,
.slide.visible {
    visibility: visible;
    opacity: 1;
    pointer-events: auto;
    z-index: 1;
}

/* 5. Keep media inside authored slide bounds */
img,
video,
canvas,
svg {
    max-width: 100%;
    max-height: 100%;
}

/* 6. Presentation chrome stays outside the slide design system */
.deck-controls {
    position: fixed;
    left: 50%;
    bottom: 22px;
    transform: translateX(-50%);
    z-index: 1000;
}

/* 7. Print one fixed-size slide per page */
@media print {
    html,
    body {
        width: 1920px;
        height: auto;
        overflow: visible;
        background: #fff;
    }

    .deck-viewport {
        position: static;
        overflow: visible;
        background: #fff;
    }

    .deck-stage {
        position: static;
        width: auto;
        height: auto;
        transform: none !important;
        background: none;
    }

    .slide {
        position: relative;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        width: 1920px;
        height: 1080px;
        break-after: page;
        page-break-after: always;
    }

    .slide:last-child {
        break-after: auto;
        page-break-after: auto;
    }

    .deck-controls {
        display: none !important;
    }
}

/* 8. Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.2s !important;
    }
}
"""

# Progress bar + edit affordance chrome, themed by CSS vars each deck defines.
CHROME_CSS = """
/* === DECK CHROME (outside the slide design system) === */
.deck-progress{position:fixed;left:0;bottom:0;height:3px;background:var(--chrome-accent,#888);
  z-index:1000;transition:width .45s cubic-bezier(.22,1,.36,1);pointer-events:none}
.deck-counter{position:fixed;right:26px;bottom:20px;z-index:1000;
  font-family:var(--font-mono,monospace);font-size:13px;letter-spacing:.16em;
  color:var(--chrome-fg,#888);opacity:.75;pointer-events:none}
.edit-hotzone{position:fixed;top:0;left:0;width:80px;height:80px;z-index:10000;cursor:pointer}
.edit-toggle{position:fixed;top:18px;left:18px;z-index:10001;opacity:0;pointer-events:none;
  transition:opacity .3s ease;border:0;border-radius:50%;width:44px;height:44px;font-size:19px;
  cursor:pointer;background:var(--chrome-accent,#888);color:#fff;line-height:1}
.edit-toggle.show,.edit-toggle.active{opacity:1;pointer-events:auto}
.edit-toggle.active{outline:3px solid rgba(255,255,255,.35)}
.editing [contenteditable="true"]{outline:1px dashed var(--chrome-accent,#888);outline-offset:4px}
.edit-hint{position:fixed;top:20px;left:74px;z-index:10001;font-family:var(--font-mono,monospace);
  font-size:12px;letter-spacing:.1em;padding:8px 12px;border-radius:4px;
  background:rgba(0,0,0,.82);color:#fff;opacity:0;transition:opacity .3s;pointer-events:none}
.edit-hint.show{opacity:1}
"""

CHROME_HTML = """
<div class="deck-progress" id="deckProgress"></div>
<div class="deck-counter" id="deckCounter">01 / 13</div>
<div class="edit-hotzone"></div>
<button class="edit-toggle" id="editToggle" title="Edit mode (E)">&#9998;</button>
<div class="edit-hint" id="editHint">EDIT MODE &mdash; CLICK TEXT &middot; CTRL+S SAVES</div>
"""

CONTROLLER_JS = r"""
/* ===========================================
   SLIDE PRESENTATION CONTROLLER
   Keyboard / wheel / touch nav + fixed-stage scaling + inline editing
   =========================================== */
class SlidePresentation {
    constructor() {
        this.slides  = Array.from(document.querySelectorAll('.slide'));
        this.stage   = document.getElementById('deckStage');
        this.bar     = document.getElementById('deckProgress');
        this.counter = document.getElementById('deckCounter');
        this.current = 0;
        this.setupStageScale();
        this.setupKeyboardNav();
        this.setupWheelNav();
        this.setupTouchNav();
        this.restore();
        this.show(0);
    }

    /* --- scale the whole 1920x1080 stage into the viewport --- */
    setupStageScale() {
        const scale = () => {
            const f = Math.min(window.innerWidth / 1920, window.innerHeight / 1080);
            const x = (window.innerWidth  - 1920 * f) / 2;
            const y = (window.innerHeight - 1080 * f) / 2;
            this.stage.style.transform = `translate(${x}px, ${y}px) scale(${f})`;
        };
        scale();
        window.addEventListener('resize', scale);
    }

    setupKeyboardNav() {
        document.addEventListener('keydown', (e) => {
            if (document.body.classList.contains('editing') && e.key !== 'Escape'
                && !(e.ctrlKey || e.metaKey)) return;
            switch (e.key) {
                case 'ArrowRight': case 'ArrowDown': case ' ': case 'PageDown':
                    e.preventDefault(); this.next(); break;
                case 'ArrowLeft': case 'ArrowUp': case 'PageUp':
                    e.preventDefault(); this.prev(); break;
                case 'Home': e.preventDefault(); this.show(0); break;
                case 'End':  e.preventDefault(); this.show(this.slides.length - 1); break;
                case 'e': case 'E':
                    if (!e.ctrlKey && !e.metaKey) { e.preventDefault(); editor.toggle(); }
                    break;
                case 'Escape':
                    if (document.body.classList.contains('editing')) editor.toggle();
                    break;
            }
        });
    }

    setupWheelNav() {
        let lock = false;
        window.addEventListener('wheel', (e) => {
            if (document.body.classList.contains('editing')) return;
            if (lock || Math.abs(e.deltaY) < 24) return;
            lock = true;
            e.deltaY > 0 ? this.next() : this.prev();
            setTimeout(() => lock = false, 620);
        }, { passive: true });
    }

    setupTouchNav() {
        let x0 = null, y0 = null;
        window.addEventListener('touchstart', (e) => {
            x0 = e.changedTouches[0].clientX; y0 = e.changedTouches[0].clientY;
        }, { passive: true });
        window.addEventListener('touchend', (e) => {
            if (x0 === null) return;
            const dx = e.changedTouches[0].clientX - x0;
            const dy = e.changedTouches[0].clientY - y0;
            if (Math.abs(dx) > 55 && Math.abs(dx) > Math.abs(dy)) dx < 0 ? this.next() : this.prev();
            x0 = y0 = null;
        }, { passive: true });
    }

    next() { this.show(this.current + 1); }
    prev() { this.show(this.current - 1); }

    show(i) {
        this.current = Math.max(0, Math.min(i, this.slides.length - 1));
        this.slides.forEach((s, k) => {
            s.classList.toggle('active',  k === this.current);
            s.classList.toggle('visible', k === this.current);
        });
        const pct = (this.current + 1) / this.slides.length * 100;
        if (this.bar) this.bar.style.width = pct + '%';
        if (this.counter) this.counter.textContent =
            String(this.current + 1).padStart(2, '0') + ' / ' + String(this.slides.length).padStart(2, '0');
        location.hash = 'slide-' + (this.current + 1);
    }

    restore() {
        const m = /slide-(\d+)/.exec(location.hash);
        if (m) this.current = Math.min(parseInt(m[1], 10) - 1, this.slides.length - 1);
    }
}

/* ===========================================
   INLINE EDITING
   Hover the top-left hotzone or press E. Ctrl+S saves to localStorage.
   =========================================== */
class InlineEditor {
    constructor(key) {
        this.key = key;
        this.on = false;
        this.targets = 'h1,h2,h3,h4,p,li,td,th,span.ed,div.ed';
        this.load();
        this.wire();
    }

    wire() {
        const btn  = document.getElementById('editToggle');
        const zone = document.querySelector('.edit-hotzone');
        const hint = document.getElementById('editHint');
        let t = null;

        btn.addEventListener('click', () => this.toggle());

        /* JS hover with a grace period — a CSS ~ selector breaks on pointer-events:none */
        zone.addEventListener('mouseenter', () => { clearTimeout(t); btn.classList.add('show'); });
        zone.addEventListener('mouseleave', () => { t = setTimeout(() => {
            if (!this.on) btn.classList.remove('show'); }, 400); });
        btn.addEventListener('mouseenter',  () => clearTimeout(t));
        btn.addEventListener('mouseleave',  () => { t = setTimeout(() => {
            if (!this.on) btn.classList.remove('show'); }, 400); });

        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
                e.preventDefault(); this.save();
                hint.textContent = 'SAVED'; hint.classList.add('show');
                setTimeout(() => hint.classList.remove('show'), 1400);
            }
        });
    }

    toggle() {
        this.on = !this.on;
        const btn = document.getElementById('editToggle');
        const hint = document.getElementById('editHint');
        document.body.classList.toggle('editing', this.on);
        btn.classList.toggle('active', this.on);
        btn.classList.toggle('show', this.on);
        hint.textContent = 'EDIT MODE — CLICK TEXT · CTRL+S SAVES';
        hint.classList.toggle('show', this.on);
        document.querySelectorAll(this.targets).forEach((el, i) => {
            el.setAttribute('contenteditable', this.on ? 'true' : 'false');
            if (this.on && !el.dataset.edid) el.dataset.edid = 'n' + i;
        });
        if (!this.on) { this.save(); hint.classList.remove('show'); }
    }

    save() {
        const out = {};
        document.querySelectorAll('[data-edid]').forEach(el => out[el.dataset.edid] = el.innerHTML);
        try { localStorage.setItem(this.key, JSON.stringify(out)); } catch (e) {}
    }

    load() {
        let saved;
        try { saved = JSON.parse(localStorage.getItem(this.key) || '{}'); } catch (e) { return; }
        if (!saved || !Object.keys(saved).length) return;
        const els = document.querySelectorAll(this.targets);
        els.forEach((el, i) => {
            const id = 'n' + i;
            if (saved[id] !== undefined) { el.dataset.edid = id; el.innerHTML = saved[id]; }
        });
    }
}

const deck   = new SlidePresentation();
const editor = new InlineEditor('__EDITKEY__');
"""
