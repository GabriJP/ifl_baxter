# Informed federated learning to train robotic arm controllers

This is the code used for "Informed federated learning to train robotic arm controllers". It is based on [Baxter_Dynamic_Model](https://github.com/EduardoRosLab/Baxter_Dynamic_Model).

## Running on Jetson Nano

Because of Cuda compatibility, at most Torch<2 and Tensorflow<2.5 can be used.

Also, these Tensorflow releases require an older release of Numpy, incompatible with used releases of Flwr.

Keras from 3.2 requires a data type from PyTorch that is not implemented in these releases.

For compatibility with Numpy and Keras 3 (which allows using Torch as backend), the working environment is as follows:

Jetson Nano 4GB
Ubuntu 20.04 bare image from [here](https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image)
Python 3.10 following [these instructions](https://forums.developer.nvidia.com/t/python-venv-on-jetson-nano/291510)
Pytorch 1.13 following [these instructions](https://qengineering.eu/install-pytorch-on-jetson-nano.html)
Keras 3 <3.2 and edit `$HOME/.local/lib/python3.10/site-packages/keras/src/backend/torch/random.py`:
Remove lines 2 and 16, as PyTorch introduced Dynamo in version 2 and any reference to it will result in error.

## Citation

Please cite this work as follows:

```bibtex
@article{jimenez2025ifl_baxter,
  title={Informed federated learning to train robotic arm controllers},
  author={Jimenez-Perera, Gabriel and Valencia-Vidal, Brayan and R. Luque, Niceto and Ros, Eduardo and Barranco, Francisco},
  journal={arXiv preprint arXiv:[TODO]},
  year={2025}
}
```
