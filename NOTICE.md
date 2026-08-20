# Third-Party Notices

GenPrior includes or interoperates with code, models, features, and datasets
from third parties. Those materials remain subject to their respective terms.
This notice is attribution and dependency documentation; it does not replace
any third-party license.

## Source-code dependencies

- **Shift-GCN**: the generated-motion encoder and CUDA temporal-shift operator
  are adapted from [Shift-GCN](https://github.com/kchengiva/Shift-GCN) and have
  been modified for GenPrior. Those files remain subject to the upstream
  Creative Commons Attribution-NonCommercial 4.0 International license, which
  is included at `genprior/models/shiftgcn/LICENSE`.
- **CLIP**: semantic text features are produced with a CLIP text encoder. Model
  files are not included in this repository and remain subject to the model
  provider's terms.
- **PyTorch, Transformers, NumPy, pandas, PyYAML, and tqdm**: runtime
  dependencies installed through `environment.yml`, each under its own
  license.

## Data and derived assets

- **NTU RGB+D 60 and NTU RGB+D 120**: raw data must be obtained by users under
  the dataset provider's access terms.
- **PKU-MMD**: raw data must be obtained by users under the dataset provider's
  access terms.
- **SynSE and SA-DVAE features**: pre-extracted feature archives are derived
  from the corresponding research code and datasets. Their redistribution
  terms must be checked before publishing the archives.
- **Generated motions and generated-motion encoder**: release links and
  checksums are forthcoming. Their archive will carry its own provenance and
  usage notice.

## Paper

`paper/GenPrior.pdf` is the camera-ready research paper accepted by ACM MM
2026. The paper is not covered by third-party software licenses.
