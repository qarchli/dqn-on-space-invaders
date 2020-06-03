# dqn-on-space-invaders

## Overview

This is a PyTorch implementation of a Deep Q-Network agent trained to play the Atari 2600 game of Space Invaders. The related paper is the following: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602v1.pdf), published in 2014 by Google Deepmind. 

This repository also corresponds to the source code for this post (LINK TO BE ADDED LATER) I have written on the subject.

## Dependencies

Install the requirements using this command:

```bash
pip install -r requirements.txt
```

There is one more thing to install to have access to the Atari environment. In fact, OpenAI gym library does not support by default the Atari environment. 

### Linux users

Simply run the following command:
```bash
pip install atari-py
```
### Windows users

Start by running the same as Linux users, if you have some errors popping up then detailed instructions to install Atari environments in Windows platforms are given [here](https://github.com/Kojoley/atari-py).    

## Usage

Once dependencies are installed, you can open `main.py` and decide whether you want to train or test the agent. This can be done by setting the `TRAIN` variable to either `True`or `False`. Other hyper-parameters are to be specified in the same file.

If trained, the agent's weights are saved in `./train`. Otherwise, videos of the agent playing are stored in `./test/`.

## TODO
- [ ] Add the possibility of hyper-parameters tuning.
- [ ] TensorBoard support.
- [ ] Add a run manager.
