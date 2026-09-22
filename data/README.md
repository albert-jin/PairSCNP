# Data preparation

Only manifests, download metadata and instructions are tracked. Raw images, masks and derived caches remain local. Follow the dataset-specific guides:

- [TopoMortar](topomortar/README.md)
- [Crack500](crack500/README.md)

Each manifest records IDs, split membership, relative layout, label thresholds and SHA256 hashes. `scripts/prepare_data.py` locates files beneath a supplied directory by filename and validates their content hash. It writes a local `prepared_manifest.json` and computes training/validation geometry before spatial cropping. Test masks are used for evaluation only and are not used to construct inference inputs.
