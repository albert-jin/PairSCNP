# Reference results

`reference_metrics.json` contains the original per-image A/B evaluation records, selected checkpoint hashes and steps. `reported_evaluation_ids.json` names the images underlying those records: 70 TopoMortar images and 50 Crack500 images. They are evaluation subsets of the full test splits and must not be represented as full-test means. These records were copied and not recomputed during release packaging.

The public evaluator defaults to every image in the test split (350 TopoMortar, 200 Crack500). Use `--reported-ids` explicitly for comparison with these archived records. Individual examples in `Figures/` are selected qualitative cases; their Dice labels refer to full source images, not display crops or dataset averages.

Checkpoints were chosen by validation Dice at steps 9600/1200 for TopoMortar baseline/PairSCNP and 2400/3600 for Crack500. All runs completed 12000 training steps. Released metadata does not include checkpoint binaries; retrain using the supplied commands to create a new checkpoint. No separate photometric-only, occlusion-only, or consistency-function ablation runs are supplied here.
