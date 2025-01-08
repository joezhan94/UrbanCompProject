# Simple U-Net model for pixel-based semantic segmentation

This is a simple U-net model for pixel-based semantic segmentation, that we use as a baseline for our work. The model is implemented in PyTorch, and was developed using Python 3.8.

Due to GitHub's file size limit, the trained model weights are not included in this repository. However, you can download them from [Google Drive](https://drive.google.com/file/d/1A4HwC8maj-Irvjaq8GVf6oxjyWu-BTGM/view?usp=sharing).

The following command can be used to replicate our work:

`cd /path/to/URBANCOMPPROJECT`

`mkdir data`

`python3 -m pip install -r requirements.txt`

`python3 get_masks.py`

`python3 model.py`

