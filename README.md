# torch-ngp-semantic
This is a rough implementation of semantic-NeRF based on torch-ngp

### Achievement:

The learning of semantic neural field can be achieved in a very short time.(！！！about 20 minutes)

### Limitation:

Because the rendering process of semantic category prediction volume rendering has not been completed yet, it cannot be accelerated by cuda. And, the implementation content of this project is relatively limited, which can only achieve the semantic field learning and limited visual display. 

# Results show
![image](https://github.com/Clear-3d/torch-ngp-semantic/blob/9f3115edcd420a914f65dc889b9590a8bec6e413/assets/show1.gif)


# Install

```bash
git clone https://github.com/Clear-3d/torch-ngp-semantic.git
cd torch-ngp-semantic 
```

### Install with pip

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```

### Build extension (optional)


```bash
# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
cd raymarching
python setup.py build_ext --inplace # build ext only, do not install (only can be used in the parent directory)
pip install . # install to python path (you still need the raymarching/ folder, since this only install the built extension.)
```

## Tested environments

* Ubuntu 18.04 with torch 1.10 & CUDA 11.3 on an A100-PCIE-40GB.

# Usage

I modified some DMSR dataset format to make it easier to use, you can download it here.

Train code：

```python
python main_nerf.py /data/dmsr/bathroom/train/ --workspace /data/dmsr/bathroom/train_rebuild/test --path_sem /data/dmsr/bathroom/train/semantic/ 
```

If your GPU memory is enough, you can add --preload

```
python main_nerf.py /data/dmsr/bathroom/train/ --workspace /data/dmsr/bathroom/train_rebuild/test --path_sem /data/dmsr/bathroom/train/semantic/ --preload
```

During training, you can check the current learning results of the scene in **workspace**.

After finishing the previous step, if you have a graphical display, you can run below to see the result in the GUI:

```
python main_nerf.py /data/dmsr/bathroom/train/ --workspace /data/dmsr/bathroom/train_rebuild/test --path_sem /data/dmsr/bathroom/train/semantic/ --preload --gui
```

Do not use -O for training, because this repository does not implement the cuda part, and other test code may break as well.

Since I didn't have time to complete the whole project, I can only make sure that the above training and display process is correct. If you are interested in improving the whole project, feel free to ask questions in the issue.

# Acknowledgement

Thanks to torch-ngp for a very nice pytorch implementation of Instant-ngp:

```
@misc{torch-ngp,
    Author = {Jiaxiang Tang},
    Year = {2022},
    Note = {https://github.com/ashawkey/torch-ngp},
    Title = {Torch-ngp: a PyTorch implementation of instant-ngp}
}

@article{tang2022compressible,
    title = {Compressible-composable NeRF via Rank-residual Decomposition},
    author = {Tang, Jiaxiang and Chen, Xiaokang and Wang, Jingbo and Zeng, Gang},
    journal = {arXiv preprint arXiv:2205.14870},
    year = {2022}
}
```

Thanks to semantic-nerf for the code ideas:

```
@inproceedings{Zhi:etal:ICCV2021,
  title={In-Place Scene Labelling and Understanding with Implicit Scene Representation},
  author={Shuaifeng Zhi and Tristan Laidlow and Stefan Leutenegger and Andrew J. Davison},
  booktitle=ICCV,
  year={2021}
}
```

Thanks to DM-NeRF for providing the dataset:

```
@article{wang2022dmnerf,
  title={DM-NeRF: 3D Scene Geometry Decomposition and Manipulation from 2D Images},
  author={Bing, Wang and Chen, Lu and Yang, Bo},
  journal={arXiv preprint arXiv:2208.07227},
  year={2022}
}
```

