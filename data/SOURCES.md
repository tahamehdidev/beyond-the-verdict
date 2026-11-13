# Data sources, licence and integrity

Downloaded 12-13 Nov 2026. The data itself is NOT committed (large; CC BY-NC 4.0 non-commercial).

## Dataset
AVeriTeC: Schlichtkrull, Guo, Vlachos (NeurIPS 2023 Datasets and Benchmarks); FEVER-8 shared task release (Schlichtkrull et al. 2024).

- Claim files and knowledge stores: Hugging Face repository `chenxwh/AVeriTeC` (linked from https://fever.ai/dataset/averitec.html)
  - https://huggingface.co/chenxwh/AVeriTeC (`data/dev.json`, `data/train.json`, `data_store/knowledge_store/...`)
- Original GitHub repository (format description, licence badge): https://github.com/MichSchli/AVeriTeC
- **Licence:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). The Hugging Face metadata lists `cc-by-nc-4.0` and the GitHub README carries the same licence. Use here is non-commercial coursework with attribution. The data is not redistributed in the submission; obtain it from the sources above.

## Files used

| File (in `data/raw/`) | Source path in HF repo | Bytes | SHA-256 |
|---|---|---|---|
| dev.json | data/dev.json | 1,785,475 | 499793726b4a5406780928a3d9dedc48d6dd53de778f22437d129cacdb08e300 |
| train.json | data/train.json | 10,184,813 | ae5eda7c42ddf1695ef185a7ba1bc716928f5adf57103e4f78aae5f9afe00f9c |
| dev_knowledge_store.zip | data_store/knowledge_store/dev_knowledge_store.zip | 11,537,899,362 | 021e258cd6fb5fe6d627a4667d663e95c184c966939c15124df9206142fc2212 |
| train_0_999.zip | data_store/knowledge_store/train/train_0_999.zip | 20,730,339,734 | 389d2284c5f25410205ec530c2101c3e6d575eae460731e1909ad19192dcb810 |

Not downloaded: `train_1000_1999.zip`, `train_2000_3067.zip` (together ~42.8 GB), the hidden test sets, the shared-task test stores.

## Verified facts (checked against the downloaded files)
- dev.json: 500 claims: Refuted 305, Supported 122, Conflicting Evidence/Cherrypicking 38, Not Enough Evidence 35.
- train.json: 3,068 claims: Refuted 1,742, Supported 849, Not Enough Evidence 282, Conflicting Evidence/Cherrypicking 195.
- Claim fields: cached_original_claim_url, claim, claim_date, claim_types, fact_checking_article, fact_checking_strategies, justification, label, location_ISO_code, original_claim_url, questions, reporting_source, required_reannotation, speaker.
- Knowledge store: dev zip has 501 entries (`output_dev/<id>.json` x500 plus a folder); train zip has 1,000 entries (`<id>.json`, ids 0-999). Each file is JSON-lines: one record per document with keys `claim_id`, `type`, `query`, `url`, `url2text` (list of sentences). There is no date field. The `type` field marks gold documents and must not be used by the pipeline.

## Model
Ollama 0.40.16, model `qwen2.5:7b-instruct`, digest `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`, 7.6B parameters, Q4_K_M (4-bit), context length 32768 (a smaller `num_ctx` is used; see config.yaml).
