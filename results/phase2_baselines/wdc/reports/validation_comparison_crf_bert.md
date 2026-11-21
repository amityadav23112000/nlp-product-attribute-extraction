# CRF–BERT validation comparison

All models passed identical validation IDs, tokens, gold labels and metadata checks. Training/validation product and cluster IDs do not overlap. No official test predictions were loaded.

| Model | Precision | Recall | Micro F1 | Macro F1 | Invalid I-tags |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRF | 0.8340 | 0.7494 | 0.7894 | 0.7596 | 1 |
| BERT_initial | 0.5942 | 0.6717 | 0.6306 | 0.4047 | 155 |
| BERT_lr1e-5 | 0.6635 | 0.7425 | 0.7008 | 0.5452 | 136 |
| BERT_lr2e-5 | 0.7556 | 0.8095 | 0.7816 | 0.6834 | 92 |
| BERT_seed42 | 0.7796 | 0.8202 | 0.7994 | 0.7764 | 59 |
| BERT_seed43 | 0.7903 | 0.8333 | 0.8112 | 0.7808 | 60 |
| BERT_seed44 | 0.7909 | 0.8246 | 0.8074 | 0.7830 | 65 |

## Errors

| Group | CRF | BERT_initial | BERT_lr1e-5 | BERT_lr2e-5 | BERT_seed42 | BERT_seed43 | BERT_seed44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| attribute_confusion | 58 | 204 | 148 | 85 | 64 | 56 | 59 |
| boundary_error | 76 | 168 | 144 | 109 | 110 | 95 | 94 |
| complete_miss | 266 | 152 | 119 | 110 | 113 | 115 | 127 |
| spurious_prediction | 104 | 360 | 309 | 224 | 196 | 202 | 195 |

## Source

| Group | CRF | BERT_initial | BERT_lr1e-5 | BERT_lr2e-5 | BERT_seed42 | BERT_seed43 | BERT_seed44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| description | 0.7789 | 0.5854 | 0.6638 | 0.7596 | 0.7784 | 0.7901 | 0.7902 |
| title | 0.8014 | 0.6853 | 0.7459 | 0.8080 | 0.8243 | 0.8362 | 0.8274 |

## Category

| Group | CRF | BERT_initial | BERT_lr1e-5 | BERT_lr2e-5 | BERT_seed42 | BERT_seed43 | BERT_seed44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Computers And Accessories | 0.8837 | 0.7971 | 0.8423 | 0.8932 | 0.9132 | 0.8995 | 0.9043 |
| Grocery And Gourmet Food | 0.5739 | 0.3974 | 0.4595 | 0.6338 | 0.6575 | 0.6757 | 0.7143 |
| Home And Garden | 0.7338 | 0.4861 | 0.6116 | 0.7141 | 0.7439 | 0.7624 | 0.7535 |
| Jewelry | 0.9327 | 0.7685 | 0.8394 | 0.9202 | 0.9245 | 0.9409 | 0.9340 |
| Office Products | 0.6628 | 0.5359 | 0.5770 | 0.6557 | 0.6658 | 0.7011 | 0.6844 |

## Frequency group

| Group | CRF | BERT_initial | BERT_lr1e-5 | BERT_lr2e-5 | BERT_seed42 | BERT_seed43 | BERT_seed44 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| extreme_tail | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| head | 0.7859 | 0.6554 | 0.7118 | 0.7942 | 0.8009 | 0.8167 | 0.8123 |
| medium | 0.8196 | 0.4242 | 0.6592 | 0.7400 | 0.7981 | 0.7852 | 0.7836 |
| tail | 0.7458 | 0.2564 | 0.2500 | 0.4314 | 0.7419 | 0.7500 | 0.7576 |

## Protocol notes

Errors use the existing one-to-one taxonomy; reconstructed FP/FN counts are checked against exact entity matching. Illegal predicted I-tags begin entities under the historical evaluator; changing that rule is a separate ablation.

Frequency buckets count positive training entity spans: head >100, medium 20–100, tail 5–19, extreme tail <5. Zero-support groups have F1=0 by convention; consult JSON support counts before interpreting them.

This evaluates the preserved processed BIO benchmark, whose token boundaries use gold spans. It does not measure annotation-independent raw-text inference.
