# 2024-nc-hackathon-ttfs

##Prerequisites:
Ubuntu20.04+, python3, pip, git, conda

For installing conda:
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate

Or follow: https://docs.anaconda.com/miniconda/


##Get Started

1. Clone reposity
2. Create conda environment: conda env create -f environment.yml
3. Activate conda environment: conda activate nc-hackathon
4. Step into repository: cd 2024-nc-hackathon-ttfs


##Test time-to-first-spike setup:

1. cd equivalent-training-ReLUnetwork-SNN/
2. python main.py --model_type=SNN --model_name=FC2_example_train --data_name=MNIST --epochs=1

##Test lava setup:

1. export PYTHONPATH=$PWD/lava/src
2. jupyter lab
3. Navigate to tutorials and run in_depth -> tutorial02

## Traffic Sign Dataset

Hints for GTSRB are available here: https://benchmark.ini.rub.de/gtsrb_dataset.html

Download Dataset from here: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
