# EDA Noise-Distribution Report (PRD §9)

## Drift checks against PRD §9's measured numbers

- **Country split**: PASS -- matches PRD §9. `{'train': {'India': 883188, 'US': 1323633}, 'test': {'US': 663106, 'France': 259452, 'India': 809986}}`
- **Singleton rate**: PASS -- matches PRD §9. `{'total': 2206821, 'singletons': 123247}`
- **Match-count distribution**: PASS -- matches PRD §9. `{1: 119157, 2: 375212, 3: 530841, 4: 484115, 5: 321957, 6: 164868, 7: 63968, 8: 18680, 9: 4205, 10: 534, 11: 37}`

## New measurements (not previously in the PRD)

### Address-component completeness by source x country

| source | country | n | blank_address_pct | missing_postal_heuristic_pct | missing_state_heuristic_pct |
| --- | --- | --- | --- | --- | --- |
| source1 | France | 259452 | 0.00 | 98.88 | 99.84 |
| source1 | India | 1693174 | 0.00 | 91.54 | 1.56 |
| source1 | US | 1986739 | 0.00 | 42.84 | 0.00 |
| source2 | France | 703378 | 3.06 | 97.50 | 69.17 |
| source2 | India | 4330364 | 2.55 | 90.26 | 27.01 |
| source2 | US | 4888147 | 3.40 | 48.02 | 3.89 |
| source3 | France | 731615 | 2.94 | 97.56 | 99.94 |
| source3 | India | 4520547 | 2.75 | 90.74 | 76.41 |
| source3 | US | 5115757 | 3.25 | 47.51 | 3.29 |

### Non-Latin-script fraction, India-labeled business_name

| source | n_india | non_ascii_name_pct |
| --- | --- | --- |
| source1 | 1693174 | 0.00 |
| source2 | 4330364 | 27.74 |
| source3 | 4520547 | 18.34 |

### S1 vs. S2/S3 garbled-name heuristic rate

| source | garbled_name_heuristic_pct |
| --- | --- |
| source1 | 0.39 |
| source2 | 12.33 |
| source3 | 7.66 |

### Typo vs. word-order-transposition frequency (sampled true positives)

- **n_true_positive_pairs_total**: 7638365
- **n_sampled**: 50000
- **typo_like**: 20720
- **transposition_like**: 2922
- **other**: 26358
- **typo_like_pct**: 41.44
- **transposition_like_pct**: 5.844
- **other_pct**: 52.716
- **thresholds**: {'token_set_high': 85.0, 'lev_high': 85.0, 'lev_low': 70.0}
