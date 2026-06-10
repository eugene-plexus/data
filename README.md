# eugene-plexus-data

Dataset preparation and tokenizer training engine for [Eugene Plexus](https://github.com/eugene-plexus).

## What this is

The data component of Eugene Plexus. It owns dataset acquisition and
preparation for the local-LLM-training platform: import (local path / web
scrape / HuggingFace Hub), cleaning + dedupe, train/val/test splitting,
**tokenizer training**, and **pretokenization** into fixed-size token blocks
(memory-mapped Arrow shards). It serves dataset manifests and tokenizers to
the `trainer` and `eval` components, which reference them by id + vocab
fingerprint. It does NOT run training — that's the `trainer`.

```
GET    /v1/data/datasets                          list dataset manifests
POST   /v1/data/datasets                          register a new (empty) dataset
GET    /v1/data/datasets/{datasetId}              read one dataset manifest
DELETE /v1/data/datasets/{datasetId}              delete a dataset and its shards
POST   /v1/data/datasets/{datasetId}/import       import raw data (path / URL / HF Hub)
POST   /v1/data/datasets/{datasetId}/pretokenize  pretokenize into fixed-size blocks
GET    /v1/data/tokenizers                         list tokenizers
POST   /v1/data/tokenizers/train                   train a new BPE tokenizer
```

Plus the standard Eugene Plexus config trio (`GET /v1/config`,
`GET /v1/config/schema`, `PATCH /v1/config`), `POST /v1/config/test`,
`POST /v1/admin/restart`, and `GET /healthz`.

## v0.3 skeleton status

This repo currently ships the **control-plane skeleton**: the HTTP wire
shape (routes + generated models + config + auth + health + safe mode) is
complete, but the actual data-preparation engine is **not implemented
yet**. Mutating endpoints (`import`, `pretokenize`, `tokenizers/train`) and
the per-dataset create/get/delete operations return `501 Not Implemented`
with a standard `Problem` body explaining that the preparation engine is
future work. `GET /v1/data/datasets` and `GET /v1/data/tokenizers` return
empty lists.

## Quick start

```bash
pip install -e ".[dev]"
python -m eugene_plexus_data
# default port 8088; override via PATCH /v1/config or the config file
```

The first run creates a `config.yaml` in the working directory with the
component's defaults. Edit through the UI, through `PATCH /v1/config`, or
by hand.

## Degraded-mode startup

Per the project-wide rule (`feedback_degraded_mode_required.md`), a bad
config never prevents the component from starting. Config endpoints stay
reachable so operators can fix the broken setting through the UI;
domain endpoints behave according to the skeleton (501) until the
preparation engine lands.

## Codegen

Pydantic models for the data component and shared schemas are generated
from the pinned `eugene-plexus/specs` commit:

```bash
python scripts/codegen.py
```

`SPECS_REF` records the commit SHA. Bump it to track a newer specs
release; CI re-runs codegen and fails if the working tree drifts.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (DCO sign-off required).
