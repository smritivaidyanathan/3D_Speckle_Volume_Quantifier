# 3D Speckle Volume Quantifier

This repository contains a custom Python pipeline for 3D quantification of nuclear speckles from fluorescent confocal image stacks. Developed as part of my undergraduate honors thesis at Brown University, the tool was used to analyze how the GA-binding transcription factor CLAMP affects the formation and morphology of splicing condensates in *Drosophila* embryos.

## Overview

- Calculates per-cell 3D speckle volumes across z-stack image data
- Designed for batch processing of multiple nuclei
- Produces output CSVs with per-speckle and per-cell measurements
- Used in: *Computational Analysis of the Impact of GA-binding Transcription Factor CLAMP on Splicing Condensate Dynamics in Drosophila*

## Directory

- `scripts/`: contains all analysis scripts, including speckle segmentation and volume quantification
- `p_and_t_values.csv`, `progress.csv`: example CSV outputs from experiments

## Getting Started

The core logic is embedded in the `.py` files under the `scripts/` directory. Comments within the scripts provide guidance on usage and customization. Scripts assume access to z-stack `.tif` images and region masks.

## Contact

For questions, reach out to: smriti_vaidyanathan@brown.edu
