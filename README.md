SAIL - **S**imple **AI** **L**ibrary

SAIL is a python package designed for speed and simplicity when developing and running deep learning models. Built on top of a c++ library with python bindings, SAIL is currently in development, changes are being released daily with new features and bug fixes.

SAIL's api is based on influences from [PyTorch](https://github.com/pytorch/pytorch), [Tensorflow](https://github.com/tensorflow/tensorflow), and [NumPy](https://github.com/numpy/numpy). Implementation of different features and modules are influenced by [PyTorch](https://github.com/pytorch/pytorch), [NumPy](https://github.com/numpy/numpy), and [Chainer](https://github.com/chainer/chainer)

SAIL only works on linux for now.

Click [here](https://sail-ml.github.io/) to view the documentation.

<!-- toc -->
- [Roadmap](#roadmap)
- [Installation](#installation)
  - [From Source](#from-source)
- [Releases and Contributing](#releases-and-contributing)
- [The Team](#the-team)
- [License](#license)

<!-- tocstop -->
## Roadmap
**NOT IN ORDER**
 - Matrix ops
 - Documentation
 - More dtype support
 - GPU support
 - Tighter integration with python

## Installation

### PIP Installation

Just run `pip install sail-ml`
Note: SAIL only works on linux

### From Source

Right now, **YOU MUST HAVE NVIDIA TOOLKIT AND AVX2 COMPATIBLE CPU TO RUN**. This is being built to a specific system for now, and in the future that will no longer be and issue.

SAIL is built using GCC and G++ 8.0. To install, you need Python >= 3.8 and NumPy.  Then just run 
`python setup.py install`

That is a super simple explanation but the project is new so  ¯\_(ツ)_/¯

## Releases and Contributing

To learn more about making a contribution to SAIL, please see our [Contribution page](CONTRIBUTING.md). No concrete policies are set yet, as the base of the project is still being setup.

## The Team

SAIL is currently maintained solely by [Tucker Siegel](https://linkedin.com/in/tucker-siegel) ([Github Profile](https://github.com/tgs266/))

## License

SAIL has a BSD License found in the [LICENSE](LICENSE) file.
