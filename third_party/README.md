# Third-party code

`pairscnp/author_deeplab` retains the SCNP loss and DeepLab head used by the experiments, together with the required Detectron2 DeepLab support modules. Original source headers are retained.

- [SCNP](https://github.com/jmlipman/SCNP-SameClassNeighborPenalization): MIT; see [SCNP_LICENSE](SCNP_LICENSE).
- [Detectron2 / DeepLab](https://github.com/facebookresearch/detectron2/tree/main/projects/DeepLab): Apache-2.0; see [DETECTRON2_LICENSE](DETECTRON2_LICENSE).

The release wrapper supplies the `SCNPCEDice` dispatch alias for `SCNPCEDiceLoss`; the vendored head and loss files are otherwise unchanged. Public configuration files derive from the Detectron2 DeepLab R103-OS16 configuration, with local base-file paths. The name R103 denotes the DeepLab stem variant; the configuration's ResNet depth field is 101.

Datasets and pretrained model downloads retain their upstream terms. The repository license does not grant rights to third-party data.
