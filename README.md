# GraphTrace

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-37.5%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)


GraphTrace is a Python library that utilizes machine learning to perform automated reconstruction of a fragmented neuron segmentation from whole brain images.

<p>
  <img src="imgs/result.png" width="400" alt="Example of before and after run obtained with GraphTrace">
</p>

## Key Features

- Graph Construction: Reads neuron fragments stored as swc files and loads them into a Networkx graph
- Proposals: Generates potential connections between nearby fragments to correct false splits in the segmentation
- Feature Generation: Extracts geometric and image-based features from the graph to be utilized by a machine learning model for classifying proposals.
- Graph Neural Network (GNN) Inference: Predicts whether to accept or reject proposals based on the generated features and graphical structure.
- Graph Update: Integrates inference results by merging fragments corresponding to an accepted proposal, resulting in a more accurate representation of neuron structures.

## Installation
To use the software, in the root directory, run
```bash
pip install -e .
```

To develop the code, run
```bash
git clone 
pip install -e .[dev]
```

## Contributing
We welcome contributions from the community! If you have suggestions, improvements, or bug reports, please open an issue or submit a pull request.

## License
GraphTrace is licensed under the MIT License.
