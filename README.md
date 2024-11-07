# 2024-nc-hackathon-ttfs

prerequisites:
Ubuntu20.04+, python3, pip

Get Started

1. Clone reposity
2. Create virtual environment: python3 -m venv nc-hack-venv
3. Activate virtual environment: source nc-hack-venv/bin/activate
4. Step into repository: cd 2024-nc-hackathon-ttfs
5. Uncomment tensorflow option in requirements.txt
6. Install libraries: pip install -r requirements.txt


Test time-to-first-spike setup:
1. cd equivalent-training-ReLUnetwork-SNN/
2. python main.py --model_type=SNN --model_name=FC2_example_train --data_name=MNIST --epochs=1

Test lava setup:
1. export PYTHONPATH=$PWD/lava/src
2. jupyter notebook
3. Navigate to tutorials and run in_depth -> tutorial02


Hints for GTSRB are available here: https://benchmark.ini.rub.de/gtsrb_dataset.html

Download Dataset from here: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
