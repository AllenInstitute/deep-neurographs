# GraphTrace

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-37.5%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)

## Overview

GraphTrace is a Python library that leverages machine learning to automatically correct splits in fragmented neuron segmentations from whole-brain images.

<p>
  <img src="imgs/result.png" width="900" alt="">
  <br>
   <b> Figure: </b>Neuron fragments corrected by using GraphTrace.
</p>

Briefly describe inference pipeline, to do...

- Graph Construction: Reads neuron fragments stored as swc files and loads them into a Networkx graph.
- Proposals: Generates potential connections between nearby fragments to correct false splits in the segmentation
- Feature Generation: Extracts geometric and image-based features from the graph to be utilized by a machine learning model that classifies the proposals.
- Graph Neural Network (GNN) Inference: Predicts whether to accept or reject proposals based on the generated features and graphical structure.
- Graph Update: Integrates inference results by merging fragments corresponding to an accepted proposal.

<p align="center">
  <img src="imgs/pipeline.png" width="800" alt="pipeline">
    <br>
  <b> Figure: </b>Visualization of split correction pipeline, see Inference section for description of each step.
</p>

## Inference

### Step 1: Graph Construction

To do...

### Step 2: Proposal Generation

To do...

### Step 3: Proposal Classification

To do...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
GraphTrace is licensed under the MIT License.
