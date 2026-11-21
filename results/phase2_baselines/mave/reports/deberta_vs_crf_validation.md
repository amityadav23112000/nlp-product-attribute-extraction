# Product attribute extraction: validation comparison

All models passed identical validation IDs, tokens, gold labels and metadata checks. Training/validation product and cluster IDs do not overlap. No official test predictions were loaded.

| Model | Precision | Recall | Micro F1 | Macro F1 | Invalid I-tags |
| --- | ---: | ---: | ---: | ---: | ---: |
| MAVE_CRF | 0.8013 | 0.8092 | 0.8053 | 0.6880 | 1 |
| MAVE_DeBERTa | 0.7948 | 0.8652 | 0.8285 | 0.7087 | 12 |

## Errors

| Group | MAVE_CRF | MAVE_DeBERTa |
| --- | ---: | ---: |
| attribute_confusion | 47 | 43 |
| boundary_error | 21 | 26 |
| complete_miss | 242 | 150 |
| spurious_prediction | 258 | 294 |

## Source

| Group | MAVE_CRF | MAVE_DeBERTa |
| --- | ---: | ---: |
| brand | 0.8000 | 0.8333 |
| description | 0.7573 | 0.7899 |
| feature | 0.7013 | 0.7292 |
| price | 0.0000 | 0.0000 |
| title | 0.8969 | 0.9122 |

## Category

| Group | MAVE_CRF | MAVE_DeBERTa |
| --- | ---: | ---: |
| Books | 0.7822 | 0.8136 |
| Dresses | 0.8060 | 0.8105 |
| Shirts & Tops | 0.8618 | 0.8881 |
| Shoes | 0.7322 | 0.7757 |
| Watches | 0.8564 | 0.8726 |

## Frequency group

| Group | MAVE_CRF | MAVE_DeBERTa |
| --- | ---: | ---: |
| extreme_tail | 0.0000 | 0.0000 |
| head | 0.8076 | 0.8348 |
| medium | 0.7860 | 0.7730 |
| tail | 0.4000 | 0.2857 |

## Protocol notes

Errors use the existing one-to-one taxonomy; reconstructed FP/FN counts are checked against exact entity matching. Illegal predicted I-tags begin entities under the historical evaluator; changing that rule is a separate ablation.

Frequency buckets count positive training entity spans: head >100, medium 20–100, tail 5–19, extreme tail <5. Zero-support groups have F1=0 by convention; consult JSON support counts before interpreting them.

This evaluates the preserved processed BIO benchmark, whose token boundaries use gold spans. It does not measure annotation-independent raw-text inference.
