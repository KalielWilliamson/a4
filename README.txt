# Using python 3.x

# Download the project from github
git pull https://github.com/KalielWilliamson/a4.git

# Install the requirements
mkdir mdr_venv
source mdr_venv/Scripts/activate
pip install opencv-python
pip install gym[all]
pip install gym[atari]
pip install ale-py
pip install gym[accept-rom-license]
pip install -r requirements.txt


# Running the code
python experiment.py
