# Final ivory logo / November 6, 2026

This is a standalone release, not a study gallery. Nothing outside this folder
is needed except Python and the dependencies in requirements.txt.

## Files and commands

- assets\logo.png: the exact approved shaded standard, 1078 x 1142, transparent.
- assets\logo-flat-color.png: the same geometry with one solid color per region.
- assets\logo-inner-rim-hdr.jpg: only the pale inner rim illuminates.
- assets\logo-inner-rim-symbols-hdr.jpg: the pale inner rim and both symbols illuminate.
- generate.py: all geometry, deterministic material shading, flat rendering,
  Ultra HDR encoding, and validation. No raster source or reference image is used.
- logo.json: approved placement, stroke widths, HDR settings, and standard checksum.
- assets\manifest.json: generated checksums, rendering environment, palette, and HDR readings.

Tested with Python 3.12.10 on Windows. In this folder, run:

    python -m pip install -r requirements.txt
    python generate.py
    python generate.py --validate-only

To regenerate without touching the supplied assets:

    python generate.py --output regenerated

## Fidelity requirements

Keep the approved 115% glyph size relative to October 31, the 117px vertical
gap, the approximately 32.8px strokes of the 4, the approximately 29.5px currency
ring, the constant 44px dark outer border, current ivory shading, fold, and
transparency. This is the currency-down layout with the 4 moved 6px down and
the currency 6px up from that study. Generate geometry before supersampling; never resize
or repaint glyph rasters. The material distance field intentionally retains
the original shading reference even though the perimeter is now concentric.
The standard SHA-256 in logo.json is an approval invariant, not a value to
silently refresh after a rendering mismatch.

The flat variant must use the new standard's masks, including its enlarged,
repositioned symbols. Fully covered regions must have single colors;
antialiasing is allowed only at boundaries.

The illuminated HDR border is the PALE INNER RIM, never the dark outer outline.
Both HDR files must be real Ultra HDR JPEGs with a gain map, based on the
shaded standard, not the flat alternative or an older logo. Their ordinary
SDR JPEG fallback is the unlit standard composited onto black. JPEG has no
transparency. An HDR-capable display AND viewer are needed to see the glow;
ordinary screenshots and SDR viewers are not proof of HDR presentation.

Do not add reference photos, historical variants, caches, virtual environments,
installed dependencies, preview experiments, or temporary files to the release.
Run CPU-intensive generation and packaging below normal priority with restricted
CPU affinity when the platform supports it.
