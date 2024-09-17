# GraphTrace Toolbox

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-37.5%25-red)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.7-blue?logo=python)


GraphTrace is a Python library that utilizes machine learning to perform automated reconstruction of fragmented neuron segments from whole brain images.

## Key Features

- Graph Construction: Converts neuron fragments into a graph structure, where nodes represent individual fragments and edges denote potential connections.
- Connection Proposals: Generates and evaluates potential connections between graph nodes to improve segmentation accuracy.
- Feature Generation: Extracts and processes features from the graph, providing valuable input for machine learning models.
- Graph Neural Network (GNN) Inference: Employs graph neural networks to predict and refine connections based on the generated features.
- Graph Update: Integrates inference results to update and merge fragments, resulting in a more accurate representation of neuron structures.

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
