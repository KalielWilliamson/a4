git pull https://github.com/KalielWilliamson/a4.git

mkdir mdr_venv
conda create -n gym python=3 pip
source mdr_venv/Scripts/activate
pip install opencv-python
pip install gym[all]
pip install gym[atari]
pip install ale-py
pip install gym[accept-rom-license]

python experiment.py
