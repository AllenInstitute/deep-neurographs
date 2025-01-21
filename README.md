# GraphTrace

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-37.5%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)

## Overview

GraphTrace is a Python library that leverages machine learning to automatically correct splits in fragmented neuron segmentations from whole-brain images. A demo Jupyter notebook is provided to showcase the process: it loads fragmented neuron segments from a predicted segmentation and corrects them using GraphTrace. You can also use the provided code to train a deep learning model and evaluate it against ground truth data.

<p>
  <img src="imgs/result.png" width="900" alt="Example of before and after run obtained with GraphTrace">
</p>

Briefly describe inference pipeline, to do...

- Graph Construction: Reads neuron fragments stored as swc files and loads them into a Networkx graph
- Proposals: Generates potential connections between nearby fragments to correct false splits in the segmentation
- Feature Generation: Extracts geometric and image-based features from the graph to be utilized by a machine learning model that classifies the proposals.
- Graph Neural Network (GNN) Inference: Predicts whether to accept or reject proposals based on the generated features and graphical structure.
- Graph Update: Integrates inference results by merging fragments corresponding to an accepted proposal.

<p style="text-align: center;">
  <img src="imgs/pipeline.png" width="700" alt="Visualization of split correction pipeline. See Inference section for description of each step.">
</p>

## Inference

To do...

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

## License
GraphTrace is licensed under the MIT License.
