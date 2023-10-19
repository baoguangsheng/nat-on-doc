# Non-Autoregressive Document-Level Machine Translation
**This code is for EMNLP 2023 Findings long paper "Non-Autoregressive Document-Level Machine Translation".**

[Paper](https://arxiv.org/abs/2305.12878) 

## Brief Intro
NAT models achieve high acceleration on documents, and sentence alignment significantly enhances their performance. 
However, current NAT models still have a significant performance gap compared to their AT counterparts. 
Our investigation shows that NAT models suffer more from the multi-modality and misalignment issues in the context of document-level MT than sentence-level MT,
and current NAT models face challenges on handling document context and discourse phenomena.


## Prepare Raw Data & Knowledge Distilled Data
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_data/run-all-data.sh iwslt17 exp_root
```

## AT Baselines
Training and testing of Transformer and G-Transformer:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_at/run-all-at.sh iwslt17 exp_root raw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_at/run-all-at.sh iwslt17 exp_root kd
```

## NAT Models
Training and testing of GLAT, GLAT+CTC, and G-Trans+GLAT+CTC:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_nat/run-all-glat.sh iwslt17 exp_root raw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_nat/run-all-glat.sh iwslt17 exp_root kd
```

Training and testing of DA-Transformer and G-Trans+DAT:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_nat/run-all-dat.sh iwslt17 exp_root raw
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_nat/run-all-dat.sh iwslt17 exp_root kd
```

## Discourse Evaluation
Run both AT and NAT models on discourse phenomena testsuite, where the data is saved to ./data-cadec and the experiments are saved to ./exp_disc_raw. 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts_disc/run-all.sh
```