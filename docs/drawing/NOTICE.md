# The images in this folder

Every image here shows digits drawn by a model trained on MNIST, which
Yann LeCun and Corinna Cortes make available under the Creative Commons
Attribution-Share Alike 3.0 licence (CC BY-SA 3.0); MNIST is derived from
collections of the National Institute of Standards and Technology (NIST).
The images are offered under the same licence, with that credit.

- `pixels-*.png`: drawn by this project's pixel model (PLAN.md, Drawing,
  part 1), at the guidance scale in the name; `make draw-pixels-grid`
  and `make draw-figures` remake and copy them.
- `codebook-*.png`: drawn by this project's codebook model (PLAN.md,
  Drawing, part 2), at the guidance scale in the name; `make
  draw-codebook-grid` and `make draw-figures` remake and copy them.
- `pixels-misses.png` and `.txt`: the first 20 of its digits the
  classifier doesn't name as asked, each beside its top half
  (`make draw-pixels-misses`).
- `diffusion-w1.png`, `diffusion-w5.png`: drawn by tiny-diffusion-in-rust's
  conditional diffusion model (its stage 4), copied from that project's
  `docs/digits-w1.png` and `docs/digits-w5.png` for comparison.
