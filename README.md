# DRE


## Installation

Clone repo with submodules:
```
git clone --recurse-submodules git@github.com:weifanjiang/DRE.git
```

Install the [modified CSSPy](https://github.com/weifanjiang/CSSPy/tree/361d18d7b9c08bcff11a18524a718b3522c48786) package:
```
cd DRE/CSSPy
pip install .
```

Unzip the Philly traces data (requires the git [LFS](https://git-lfs.github.com/) plug-in):
```
cd DRE/philly-traces
tar -xvf trace-data.tar.gz
```
