# FruitFly student research data — published experiment

Run: hall-20260922-replay1, 2026-09-22. Approved for publication by the user.

Public site: https://xiaoyh-code.github.io/fruitfly-flight-lab/

On the public site, the pipeline, future tasks and other scenes are linked separately. The 3DGS replay requires the online site and the author's remote Hall scene. The ZIP link is available on the hosted page; its payload is already extracted here.

Open index.html for the offline data viewer; open training-data.xlsx for typed, filterable tables. Figures, tables and MuJoCo videos can be inspected offline.

## Contents

- data/: original NPZ, exact CSV exports, sanitized saved JSON, seed manifest, dictionaries and hashes.
- figures/: four PNG/SVG plots, bilingual captions and their source hashes.
- models/: source, selected and rejected action heads with identical tensors and sanitized metadata.
- media/: independent MuJoCo third-person replay; the Hall scan and its RGB footage are excluded.
- source/: project code snapshot and pinned dependencies for method inspection, not a standalone install.
- student-report-guide.md: teaching prompts, qualified results and repeat-experiment command.
- manifest.json: size and SHA-256 of every packaged payload (manifest itself excluded).

## Interpretation

2,400 collected examples; selected round 0 fitted 1,600. All 64 evaluations are retained. Repeated validation seeds and paired control seeds mean these are not 64 independent trials. 20/20 audit successes apply only to the one approximate obstacle, static Hall and goal/velocity-assisted planar task. This does not demonstrate real depth, full-scene safety or real flight. Only two final-epoch MSE values exist; do not invent a learning curve.

## Attribution and scene boundary

Crazyflie model: Google DeepMind MuJoCo Menagerie / whoenig, MIT (license included). Flyvis: https://github.com/TuragaLab/flyvis ; MuJoCo: https://mujoco.org/ ; Gymnasium: https://gymnasium.farama.org/ . Hall source: https://huggingface.co/datasets/amacati/splats . The Hall scene is All Rights Reserved and is not redistributed in this package. Citing its source does not grant redistribution permission. Frozen Flyvis pretrained model files are not redistributed.
