# Generator Field Guide

Current shop guidance for preparing source plates, prompting image generators,
and comparing image-to-3D routes. This guide complements the queue and route
contract in the root README; it does not replace each model repository's setup
and backend documentation.

The rules below are empirical. Preserve the named route, version, and evidence
scope when applying them. A result from one source class or local port is not a
universal model claim.

## Before every run

Record enough information to replay and interpret the result:

- exact source image bytes, dimensions, and digest;
- complete prompt, seed, steps, guidance, and other generation settings;
- requested route and the effective runner, backend, device, and model version;
- output bytes and dimensions, including partial or failed output;
- source, prompt, settings, and output adjacent in the evidence sheet.

Greenroom completion proves that the effective command finished. It does not
prove that the source plate was suitable, the prompt isolated the intended
variable, or the output supports a scientific or production claim.

## Source plates

Use a source plate at the target aspect ratio and resolution whenever source
geometry matters. In the observed `mflux` 0.16.9 edit route, each reference is
scaled directly to the requested dimensions without preserving aspect ratio,
cropping, or padding. A non-square source sent to a square run is therefore
anisotropically stretched before inference.

Treat these as first-class experimental variables:

- camera view and projection;
- framing and crop;
- lighting and background;
- shading and material contrast;
- reference order in a multi-reference run.

When exact camera authority is available upstream, render the desired camera
view into the source plate instead of asking the text prompt to transpose it.
For multi-reference routes, do not assume references are exchangeable: order
and position may carry different conditioning roles.

## FLUX and MFLUX prompting

Start with short, literal language. Five to twenty words has been a productive
working range in current assays, not a hard cap. Long prompts can overpower the
image prior; an observed roughly 90-word prompt materially displaced source
conditioning.

Prompts are causal interventions, not captions. Apply these checks after every
prompt edit:

1. Classify the edit as grammar-only, nuisance constraint, or target-semantic
   intervention.
2. Check it against known generator behavior before adopting it.
3. Ask whether any noun, species name, body part, material, mood, or camera term
   could directly select or overwrite the outcome being measured.
4. Rerun the previous prompt when an apparently cleaner rewrite changes target
   semantics.

Specific class names can dominate an emergent class. Specific body-part wording
can selectively rewrite that region. Use ordinary visual language in prompts;
keep assay jargon such as "biological tripod" in records rather than sending it
to the generator. Negative-clause laundry lists are not a neutral default and
should be introduced only when their effect is itself being tested.

Source authority is nonlinear. A crude or semantically ambiguous carrier can
be ignored or heavily regularized, while a plausible, overdetermined source may
be followed very closely. Do not infer a smooth response curve from one strong
or weak result.

## Assay design

Use the same source, prompt, seed, and settings when comparing routes. When
testing source authority, hold prompt and seed fixed and use source plates with
matched camera, raster, framing, lighting, and preprocessing. When testing
prompt authority, hold the exact source bytes fixed.

One seed establishes one basin sample. Use multiple seeds to distinguish a
stable source-conditioned effect from stochastic basin selection. Paired same-
seed comparisons remain useful, but they do not by themselves establish a
general effect.

Add a control only when it can discriminate a live hypothesis, such as proving
that an external reference route is healthy. A ceremonial control that cannot
change the interpretation only consumes compute and creates false rigor.

## TRELLIS.2 and Stable Fast 3D

Treat TRELLIS.2 and Stable Fast 3D as source-dependent competitors. Neither is
the universal default winner:

- TRELLIS.2 often preserves sharper geometry on current creature sources;
- Stable Fast 3D can produce stronger textures and can occasionally produce the
  more useful geometry;
- compare both on the exact same admitted source whenever the reconstruction
  choice matters.

TRELLIS.2 is sensitive to projected view and source lighting. Deliberate camera
or lighting variants can be useful assays, but label them as distinct source
conditions rather than route repeats.

Keep route failures separate from model-ceiling claims. The official CUDA route
has produced healthy reference geometry where Apple Silicon ports produced
degenerate output. MLX or MPS failure is evidence about that effective route
until the same source is exercised on the reference implementation.

Fur and other dense repeated structures are hostile cases for the current Mac
TRELLIS routes. They can produce malformed relief or extremely large meshes.
Prefer geometry-only generation or a headless render of the unsimplified result
for diagnosis before starting an expensive simplification path. Record the raw
vertex and face counts before attributing a hang to inference.

## Updating this guide

Add a rule when it has replayable evidence and changes how another user should
prepare, run, or interpret a job. Name the effective route and version when the
behavior may be implementation-specific. When later evidence disagrees, narrow
or supersede the rule; do not silently preserve the stronger wording.

Keep implementation details in the owning generator README:

- `~/dev/mlx-ideogram4/README.md` for FLUX/MFLUX;
- `~/dev/trellis2mlx/README.md` for TRELLIS.2;
- `~/dev/sf3d/README.md` for Stable Fast 3D.

Keep queueing, lease, bump, route-identity, and receipt behavior in Greenroom's
root README. This document owns the cross-route operating layer between those
surfaces.
