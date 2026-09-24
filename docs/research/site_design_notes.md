# Public site — design research and brand direction

> For research use only. Not for diagnostic use.

Date: 2026-09-24. Scope: the public informational site in `services/site/`.
Method: fetched the marketing home pages of seven radiology-AI companies and two
design-forward science/AI product sites (text and markup only — motion was
inferred from markup, asset names and embedded players, so "no motion visible"
means none was detectable, not none exists). Findings below are observations
about other people's sites, not claims about their products.

## 1. Reference sites

### Lunit (lunit.io)
- **Hero:** mission headline ("Conquer cancer through cutting-edge AI") plus a
  three-word tagline; single CTA "Speak with a specialist"; abstract tech
  background, clinician testimonial photo below the fold.
- **Motion:** none detectable; static image assets.
- **Colour:** white/light neutral ground, teal-turquoise accent for buttons and
  icons.
- **Type:** one sans family; large metric numbers, corporate hierarchy.
- **Evidence / regulatory:** adoption counters (sites, countries, publications,
  partnerships); a named clinician quote; no FDA/CE language on the home page.
- **Credible:** partner names, publication hub. **Salesy:** aspirational
  adjectives, speed/efficiency framing without outcome data up front.

### Qure.ai (qure.ai)
- **Hero:** superlative headline ("world's most adopted…") over impact counters
  (lives, countries, sites); "Contact us" CTA; full-width photo banner; partner
  logo carousel.
- **Motion:** use-case carousel and expandable "see how" panels.
- **Colour:** white ground, blue/teal accent; regulatory badges rendered as
  neutral grey shields.
- **Type:** sans; oversized numerals for metrics.
- **Evidence / regulatory:** FDA-cleared / CE / GDPR / HIPAA badge row; a
  peer-reviewed study with named authors; partner logos.
- **Credible:** badges, named researchers, a linked study. **Salesy:**
  superlatives, rotating testimonials that blur attribution.

### Annalise.ai → Harrison.ai (harrison.ai)
- **Hero:** "Reimagining the future of healthcare with AI" + "unlocking
  infinite medical capacity"; CTAs "Get started" / "Book a demo"; clinical
  environment photography.
- **Motion:** logo and case-card carousels; otherwise static.
- **Colour:** white ground, blues and greys.
- **Type:** clean sans, high contrast.
- **Evidence / regulatory:** deployment share (">35 % of NHS England CXRs"),
  "130 findings", site and country counts, funding, "40+ countries with
  clearance" (no product-level clearance detail on the home page).
- **Credible:** specific deployment numbers, "decision-support / assistive"
  wording. **Salesy:** "game-changer", "infinite capacity".

### Aidoc (aidoc.com)
- **Hero:** all-caps "ALWAYS ON AI" + platform subline; "Book a demo" repeated;
  text-led with small icons, embedded YouTube stroke-case video.
- **Motion:** video embed only; no scroll animation detectable.
- **Colour:** white ground; bright orange for alerts, blue for links — a
  deliberate "you can't miss it" accent.
- **Type:** bold sans headlines, institutional tone.
- **Evidence / regulatory:** "60 million scans / year", "230+ clinical
  studies", CE / UKCA marks, FDA Breakthrough designations, named hospital
  logos, footnoted outcome metrics ("5.2 days reduction in turnaround").
- **Credible:** footnotes, named radiologists, third-party algorithm partners.
  **Salesy:** ROI language, stacked testimonials, "award-winning".

### Gleamer (gleamer.ai)
- **Hero:** "Among the world's most adopted AI solutions" + exam/site/country
  counters; CTAs "Discover Copilot", "Get a demo".
- **Motion:** case-study video thumbnails; nothing else detectable.
- **Colour / type:** not exposed in markup (CDN imagery); conventional
  healthcare-SaaS blues/greys assumed.
- **Evidence / regulatory:** "140+ clinical studies", "60+ publications" with
  journal names, clinician quotes with credentials, institution logos, outcome
  claims ("70 % reduction in reading time"); no regulatory marks visible on
  the home page.
- **Credible:** journal-named publications. **Salesy:** "pioneering the
  future", logo walls.

### Viz.ai (viz.ai)
- **Hero:** emotional headline ("This is the face of better outcomes"), subline
  that front-loads "over 50 FDA-cleared algorithms"; two audience CTAs
  (providers / life sciences); patient photograph rather than tech imagery.
- **Motion:** lazy-loaded SVG placeholders; no animation detectable.
- **Colour:** dark/neutral ground, high-contrast text, restrained accent.
- **Type:** sans; hierarchy by size.
- **Evidence / regulatory:** clearance count in the subline; "clinically
  validated / real-world performance" section; named physician quotes; "See
  the data" link to publications.
- **Credible:** regulatory status stated up front, publication link.
  **Salesy:** "revolutionizing", pun taglines.

### Rad AI (radai.com)
- **Hero:** "AI solutions for modern radiology" / "built by radiologists, for
  radiologists"; "Request a demo" three times above the fold; abstract
  geometric background, team photography — no actual radiographs.
- **Motion:** continuous client-logo carousel.
- **Colour:** off-white ground, teal/cyan accent, dark grey text.
- **Type:** modern sans; bold phrases inline.
- **Evidence / regulatory:** "200+ health organizations", SOC 2 Type II and
  HIPAA, RSNA Ventures partnership, user counts, funding; founder-physician
  narrative; seven named executive testimonials. No peer-reviewed studies
  linked from the home page.
- **Credible:** compliance specifics, named people. **Salesy:** "up to 50 %
  faster" with no denominator, "dramatic improvement".

### Isomorphic Labs (isomorphiclabs.com) — design-forward science
- **Hero:** two-word mission ("Solve all disease") then one paragraph of
  context; CTAs "Work with us" and section navigation; almost no imagery.
- **Motion:** none detectable; content-first.
- **Colour / type:** minimal palette, large bold statement type against plain
  explanatory paragraphs.
- **Credibility:** scientific lineage (AlphaFold), a named leader's quote,
  concrete artefact (the "Drug Design Engine"), news feed.
- **Lesson:** a very short headline plus one honest paragraph reads as more
  confident than a wall of counters. Treats the reader as intelligent.

### Owkin (owkin.com) — design-forward medical AI
- **Hero:** "Building the autonomous AI Scientist" + vision subline; arrow
  CTAs; layered product-UI graphics with gradient stripes.
- **Motion:** gradient stripe elements and layered UI; interactive states
  implied.
- **Colour:** dark ground, white type, blue gradient accents.
- **Type:** sans; hierarchy by size only.
- **Credibility:** named pharma partners, publications section, concrete case
  studies (TLS detection on H&E slides).
- **Lesson:** dark ground suits imaging content and reads as "reading room";
  superlatives ("SuperIntelligence") undercut it.

## 2. Patterns across the set

| Pattern | Seen at | Effect on a research reader |
|---|---|---|
| Adoption counters as the primary evidence (sites, countries, scans/year) | Qure, Gleamer, Harrison, Aidoc, Rad AI | Signals scale, not accuracy; reads as sales when there is no per-finding metric next to it |
| Logo walls and carousels | Qure, Gleamer, Harrison, Rad AI | Low information density; the only motion most sites use |
| Regulatory marks as badges, not sentences | Qure, Aidoc | Credible when accurate; Viz.ai's "50 FDA-cleared algorithms" in the subline is the clearest version |
| Named clinicians with credentials | Aidoc, Gleamer, Viz, Rad AI | Strong; anonymous or rotating quotes are weak |
| Footnoted numbers | Aidoc | The single most credibility-building device; rare |
| No radiograph in the hero | Lunit, Rad AI, Harrison, Isomorphic | The product's actual input is hidden behind photography or abstraction |
| "Revolutionary / game-changer / infinite" | Harrison, Viz, Gleamer, Owkin | Uniformly reads as hype |
| Demo CTA repeated ≥ 3× above the fold | Aidoc, Rad AI | Conversion-page feel, not a research page |

Nobody in the set shows the product doing its job in the hero — a
radiograph, a region of attention, a calibrated probability with a threshold,
and an explicit "cannot assess" state. That is the gap our cover fills.

## 3. Our brand direction

We are a research prototype. Credibility over sizzle. The cover should show
exactly what the software outputs, honestly, and stop.

### Palette — clinical, high-contrast, one accent
Cover (dark, "reading-room" ground so the radiograph reads correctly):

| Token | Value | Use | Contrast on ground |
|---|---|---|---|
| `--c-bg` | `#0a0e11` | page ground | — |
| `--c-bg-2` | `#121a1e` | panels | — |
| `--c-ink` | `#eef3f3` | headline / body | 17.3 : 1 |
| `--c-ink-2` | `#b7c3c4` | secondary text | 10.7 : 1 (9.7 : 1 on panels) |
| `--c-muted` | `#8a999e` | captions, axis labels | 6.6 : 1 (6.0 : 1 on panels) |
| `--c-accent` | `#3fd6c4` | one accent: links, primary button, scan line, probability fill | 10.7 : 1 (button text `#0a0e11` on accent: 10.7 : 1) |
| `--c-warn` | `#f2b134` | "unable to assess" and banner mark only | 10.3 : 1 |

Details page (light, paper ground for long reading):

| Token | Value | Use | Contrast on ground |
|---|---|---|---|
| `--p-bg` | `#f6f8f8` | page ground | — |
| `--p-surface` | `#ffffff` | cards, tables | — |
| `--p-ink` | `#16211f` | body | 15.6 : 1 |
| `--p-ink-2` | `#3f4d4c` | secondary | 8.6 : 1 |
| `--p-muted` | `#5b6a69` | captions | 5.3 : 1 |
| `--p-accent` | `#0b6e68` | links, section numbers, buttons | 5.7 : 1 (6.1 : 1 on white; white on accent 6.1 : 1) |
| `--p-warn` | `#8a3b0d` | notices | 7.3 : 1 |

All pairs were checked with the WCAG relative-luminance formula; the lowest
in use is muted text on a dark panel at 6.0 : 1, above the 4.5 : 1 floor.

Heat-map ramp (data colour, not brand): transparent → `#ffb000` → `#ff3d00`,
alpha-capped at 0.65 so the anatomy underneath stays legible. The same teal
accent is used on both pages so the two feel like one site; the ground flips
from dark (image) to light (reading).

### Type pairing — Google Fonts only
- **Manrope** 400 / 500 / 700 for headline, body and UI. Geometric, modern,
  even colour at small sizes; reads "engineering" rather than "wellness".
- **IBM Plex Mono** 400 / 600 for every number that is a measurement:
  probabilities, thresholds, AUROC, CIs, hashes, model versions. Numbers
  always in mono so a reader can tell data from prose at a glance.
- Fallback stacks: `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`
  and `ui-monospace, SFMono-Regular, Menlo, Consolas, monospace`.
- Scale: h1 `clamp(2.2rem, 4.8vw, 3.6rem)` weight 700 letter-spacing −0.02em;
  body 1.0625rem / 1.6; captions 0.85rem; mono data 0.9em.

### Motion principles — subtle, purposeful, reduced-motion respected
1. **One animation on the site**, the hero scan. Nothing else moves except
   focus/hover states.
2. **Every movement encodes a step in the pipeline**: sweep = image is being
   read; heat-map fade-in = region of attention; probabilities counting up =
   calibrated output; the occasional "unable to assess" cycle = the gate
   refusing. No decorative particles, gradients or parallax.
3. **Slow and quiet**: 12 s loop, ease-in-out, ≤ 0.65 overlay alpha, no
   flashing, sweep speed well under 3 Hz.
4. **Cheap**: `requestAnimationFrame` only; paused on `visibilitychange` and
   when the hero scrolls out of view (IntersectionObserver); one 512 px canvas.
5. **`prefers-reduced-motion: reduce`** renders the final frame once (heat-map
   on, probabilities at their final values) and never animates.
6. Every animated value is also present as text for screen readers; the
   canvas has a descriptive `aria-label` and the panel is a live region with
   `aria-live="off"` so it does not chatter.

### Tone — evidence-first, no hype
- One headline, one sentence, two buttons. No counters of sites or scans
  (we have none), no logos, no testimonials, no demo CTA.
- Numbers only with their denominator, CI and dataset; every result on the
  details page keeps its source, and the internal-test caveat sits above the
  tables, not below them.
- The status notice ("research prototype, not a cleared device, no clinical
  performance claim") is on the cover, above the fold's end, in plain words.
- Illustrative values in the hero are labelled "illustration — not a model
  output" so the animation cannot be mistaken for a result.
- Vocabulary: "second read", "calibrated probability", "unable to assess",
  "subgroup", "internal test". Never "revolutionary", "detects cancer",
  "faster", "accurate".
