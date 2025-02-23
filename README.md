This project is about training a neural network for sorting goban pictures into groups based on their size (9x9, 13x13 or 19x19).
Thus, when a board is detected by the GoStream app, the right position model (9x9, 13x13 or 19x19) can be called without any action of the user.

---

## 🐍 Run the script in an [Anaconda](https://anaconda.org/) environment
Install Anaconda

Change directory to the goban_size_regnition_tool directory

Create an environment
```sh
$ conda env create -f environment.yml
```
Activate the environment
```sh
$ conda activate goban-size-recognition-tool
```
Run the app
```sh
$ python my_script.py
```

---

## 🐧 Run script in Linux system
```sh
$ ./launch.sh
```