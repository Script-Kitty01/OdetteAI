# FluxGuard experiment summary

| run | kind | metric | model | baseline | leakage gate |
|---|---|---|---|---|---|
| cmapss-rul-fd001\20261012T111302Z | rul | mae (lower better) | 9.634 | 18.399 | True |
| cwru-anomaly-baseline\20261012T111002Z | anomaly | roc_auc | 0.824 | — | True |
| cwru-classifier-cnn\20261012T111016Z | classifier | accuracy | 0.878 | 0.958 | True |
| cwru-classifier-cwt\20261012T111249Z | classifier | accuracy | 0.958 | 0.958 | True |
| cwru-classifier-setcn\20261012T111205Z | classifier | accuracy | 0.757 | 0.958 | True |
| cwru-classifier-stft\20261012T111218Z | classifier | accuracy | 0.958 | 0.958 | True |
