# Product attribute extraction: validation comparison

All models passed identical validation IDs, tokens, gold labels and metadata checks. Training/validation product and cluster IDs do not overlap. No official test predictions were loaded.

| Model | Precision | Recall | Micro F1 | Macro F1 | Invalid I-tags |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRF | 0.8340 | 0.7494 | 0.7894 | 0.7596 | 1 |
| BERT_seed42 | 0.7796 | 0.8202 | 0.7994 | 0.7764 | 59 |
| DeBERTa_seed42 | 0.8298 | 0.8678 | 0.8484 | 0.8235 | 40 |

## Errors

| Group | CRF | BERT_seed42 | DeBERTa_seed42 |
| --- | ---: | ---: | ---: |
| attribute_confusion | 58 | 64 | 45 |
| boundary_error | 76 | 110 | 85 |
| complete_miss | 266 | 113 | 81 |
| spurious_prediction | 104 | 196 | 154 |

## Source

| Group | CRF | BERT_seed42 | DeBERTa_seed42 |
| --- | ---: | ---: | ---: |
| description | 0.7789 | 0.7784 | 0.8271 |
| title | 0.8014 | 0.8243 | 0.8736 |

## Category

| Group | CRF | BERT_seed42 | DeBERTa_seed42 |
| --- | ---: | ---: | ---: |
| Computers And Accessories | 0.8837 | 0.9132 | 0.9127 |
| Grocery And Gourmet Food | 0.5739 | 0.6575 | 0.6438 |
| Home And Garden | 0.7338 | 0.7439 | 0.7986 |
| Jewelry | 0.9327 | 0.9245 | 0.9476 |
| Office Products | 0.6628 | 0.6658 | 0.8000 |

## Frequency group

| Group | CRF | BERT_seed42 | DeBERTa_seed42 |
| --- | ---: | ---: | ---: |
| extreme_tail | 0.0000 | 0.0000 | 0.0000 |
| head | 0.7859 | 0.8009 | 0.8502 |
| medium | 0.8196 | 0.7981 | 0.8387 |
| tail | 0.7458 | 0.7419 | 0.8358 |

## Protocol notes

Errors use the existing one-to-one taxonomy; reconstructed FP/FN counts are checked against exact entity matching. Illegal predicted I-tags begin entities under the historical evaluator; changing that rule is a separate ablation.

Frequency buckets count positive training entity spans: head >100, medium 20–100, tail 5–19, extreme tail <5. Zero-support groups have F1=0 by convention; consult JSON support counts before interpreting them.

This evaluates the preserved processed BIO benchmark, whose token boundaries use gold spans. It does not measure annotation-independent raw-text inference.
