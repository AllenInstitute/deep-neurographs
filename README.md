# GraphTrace

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-37.5%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)


<b> GraphTrace</b> is a Python library that automatically corrects split errors in fragmented neuron segmentations from whole-brain images. It takes SWC files as input and uses a graph-based neural network pipeline to propose, score, and merge candidate connections between neuron fragments. GraphTrace efficiently handles datasets with millions of fragments across whole-brain volumes, enabling high-throughput proofreading and reconstruction at scale.

<p>
  <img src="imgs/result.png" width="900" alt="">
  <br>
   <b> Figure: </b>GraphTrace reconnects fragmented neuron segments into coherent traces.
</p>

## Overview

The neuron fragment split correction pipeline consists of three main steps:

<blockquote>
  <p>a. <strong>Graph Construction</strong>: Reads neuron fragments stored as SWC files and loads them into a Networkx graph.</p>
  <p>b. <strong>Proposal Generation</strong>: Generates potential connections between nearby fragments.</p>
  <p>c. <strong>GNN-Based Inference</strong>: Predicts whether to accept or reject proposals based on the geometric and image-based features.</p>
</blockquote>
<br>

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

## Usage

To do...

## Contact Information
For any inquiries, feedback, or contributions, please do not hesitate to contact us. You can reach us via email at anna.grim@alleninstitute.org or connect on [LinkedIn](https://www.linkedin.com/in/anna-m-grim/).

## License
GraphTrace is licensed under the MIT License.
