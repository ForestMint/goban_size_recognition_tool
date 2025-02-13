This project is about training a neural network for sorting goban pictures into groups based on their size (9x9, 13x13 or 19x19).
Thus, when a board is detected by the GoStream app, the right position model (9x9, 13x13 or 19x19) can be called without any action of the user.

# create Anaconda environment
$ conda env create -f environment.yml

# activate venv (virtual environment)
$ source .venv/bin/activate
$ pip install keras
$ pip install tensorflow
$ pip intall Pillow