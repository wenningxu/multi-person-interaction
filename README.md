# Multi-Person Interaction Generation from Two-Person Motion Priors

## Getting started

This code was tested on `Ubuntu 20.04.1 LTS` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment

```shell
conda create --name intergen
conda activate intergen
pip install -r requirements.txt
```

### 2. Get data


Download the data from [webpage](https://tr3e.github.io/intergen-page/). And put them into ./data/.



### 1. Download the checkpoint
Run the shell script:

```shell
./prepare/download_pretrain_model.sh
```

### 2. Modify the configs
Modify config files ./configs/model.yaml and ./configs/infer.yaml


### 3. Modify the input file ./prompts.txt like:

```sh
In an intense boxing match, one is continuously punching while the other is defending and counterattacking.
With fiery passion two dancers entwine in Latin dance sublime.
Two fencers engage in a thrilling duel, their sabres clashing and sparking as they strive for victory.
The two are blaming each other and having an intense argument.
Two good friends jump in the same rhythm to celebrate.
Two people bow to each other.
Two people embrace each other.
...
```

### 4. Run
```shell
python tools/infer.py
```
The results will be ploted and put in ./results/




## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{Xu2025,
  series = {SIGGRAPH Conference Papers ’25},
  title = {Multi-Person Interaction Generation from Two-Person Motion Priors},
  url = {http://dx.doi.org/10.1145/3721238.3730688},
  DOI = {10.1145/3721238.3730688},
  booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  publisher = {ACM},
  author = {Xu,  Wenning and Fan,  Shiyu and Henderson,  Paul and Ho,  Edmond S. L.},
  year = {2025},
  month = aug,
  pages = {1–11},
  collection = {SIGGRAPH Conference Papers ’25}
}
```




## Acknowledgement
This code is derived from InterGen
