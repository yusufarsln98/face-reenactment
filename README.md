# Face Reenactment: Exploring DeepFakes

A comparative exploration of facial motion transfer and deepfake architectures for face reenactment.

---

## Repository Structure

* **`autoencoders/`**: Method 1 — Autoencoder-based face swapping architecture.
* **`fomm/`**: Method 2 — First Order Motion Model (GAN-based relative keypoint motion transfer).
  * **Experiment & Research Notes:** Detailed technical notes, implementation timeline, and failure mode analyses are located in [`fomm/notes/`](fomm/notes/).
  * **Inference Runner:** Run `python run.py <input_folder>` inside `fomm/`.
* **`rectified_flow/`**: Method 3 — Rectified flow head swap.
* **`docs/`**: Shared group deliverables:
  * **[`docs/slides/`](docs/slides/)**: LaTeX Beamer presentation slides (including interactive YouTube demo links).
  * **[`docs/report/`](docs/report/)**: LaTeX academic project report.

---

## Quick Start (FOMM)

```bash
cd fomm
python run.py bella_poarch
```
Outputs are generated in `fomm/outputs/<target>/` with side-by-side comparison videos.
