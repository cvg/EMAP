<p align="center">
  <h1 align="center">3D Neural Edge Reconstruction</h1>
  <p align="center">
    <a href="https://github.com/rayeeli/"><strong>Lei Li</strong></a>
    ·
    <a href="https://pengsongyou.github.io/"><strong>Songyou Peng</strong></a>
    ·
    <a href="https://niujinshuchong.github.io/"><strong>Zehao Yu</strong></a>
    ·
    <a href="http://b1ueber2y.me/"><strong>Shaohui Liu</strong></a>
    ·
    <a href="https://rpautrat.github.io/"><strong>Rémi Pautrat</strong></a>
    <br>
    <a href=""><strong>Xiaochuan Yin</strong></a>
    ·
    <a href="https://people.inf.ethz.ch/pomarc/"><strong>Marc Pollefeys</strong></a>
  </p>
  <h2 align="center">CVPR 2024</h2>
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2024/papers/Li_3D_Neural_Edge_Reconstruction_CVPR_2024_paper.pdf">Paper</a> | <a href="https://youtu.be/ONXfu2b4Nug">Video</a> | <a href="https://neural-edge-map.github.io/">Project Page</a></h3>
</p>

<p align="center" style="display: flex; justify-content: center;">
  <img src="./media/replica.gif" style="width: 80%;" />
</p>

<p align="center">
EMAP enables 3D edge reconstruction from multi-view 2D edge maps.  
</p>
<br>

## Installation

```
git clone https://github.com/cvg/EMAP.git
cd EMAP

conda create -n emap python=3.8
conda activate emap

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Datasets
Download datasets:
```
python scripts/download_data.py 
```
The data is organized as follows:

```
<scan_id>
|-- meta_data.json      # camera parameters
|-- color               # images for each view
    |-- 0_colors.png
    |-- 1_colors.png
    ...
|-- edge_DexiNed        # edge maps extracted from DexiNed
    |-- 0_colors.png
    |-- 1_colors.png
    ...
|-- edge_PidiNet        # edge maps extracted from PidiNet
    |-- 0_colors.png
    |-- 1_colors.png
    ...
```

## Training and Edge Extraction
To train and extract edges on different datasets, use the following commands:

#### ABC-NEF_Edge Dataset
```
bash scripts/run_ABC.bash
```

#### Replica_Edge Dataset
```
bash scripts/run_Replica.bash
```

#### DTU_Edge Dataset
```
bash scripts/run_DTU.bash
```

### Checkpoints
We have uploaded the model checkpoints on [Google Drive](https://drive.google.com/file/d/1kU87MqDv5IvwjCt8I8KecTlIok39fuws/view?usp=sharing). 

## Evaluation
To evaluate extracted edges on ABC-NEF_Edge dataset, use the following commands:

#### ABC-NEF_Edge Dataset
```
python src/eval/eval_ABC.py
```

## Code Release Status
- [x] Training Code
- [x] Inference Code
- [x] Evaluation Code
- [ ] Custom Dataset Support

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of EMAP is licensed under a [MIT License](LICENSE.txt).

## <a name="CitingEMAP"></a>Citing EMAP

If you find the code useful, please consider the following BibTeX entry.

```BibTeX
@InProceedings{li2024neural,
  title={3D Neural Edge Reconstruction},
  author={Li, Lei and Peng, Songyou and Yu, Zehao and Liu, Shaohui and Pautrat, R{\'e}mi and Yin, Xiaochuan and Pollefeys, Marc},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```

## Contact
If you encounter any issues, you can also contact Lei through lllei.li0386@gmail.com.

## Acknowledgement

This project is built upon [NeuralUDF](https://github.com/xxlong0/NeuralUDF), [NeuS](https://github.com/Totoro97/NeuS) and [MeshUDF](https://github.com/cvlab-epfl/MeshUDF). We use pretrained [DexiNed](https://github.com/xavysp/DexiNed) and [PidiNet](https://github.com/hellozhuo/pidinet) for edge map extraction. We thank all the authors for their great work and repos.
