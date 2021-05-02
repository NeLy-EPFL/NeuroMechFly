## Installing NeuroMechFly
To avoid any conflicts of python packages with your existing python environment, we highly recommend to use virtualenv or conda env. To create a conda environment, follow the following steps: 

First, you can download the repository on your local machine by running the following line in the terminal:
```bash
$ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
```
After the download is complete, navigate to the NeuroMechFly folder:
```bash
$ cd NeuroMechFly
```
In this folder, run the following commands to create a virtual environment and activate it:
```bash
$ conda create -n neuromechfly python=3.6 numpy Cython
$ conda activate neuromechfly
```
First, install the FARMS Container by running:
```bash
$ pip install git+https://gitlab.com/FARMSIM/farms_container.git
```
Finally, install all the dependencies by running:
```bash
$ pip install -e .
```
Once you complete all the steps, NeuroMechFly is ready to use!

---
**NOTE:**
Each time you start using NeuroMechFly, please activate virtual environment by running: 
```bash
$ conda activate neuromechfly
```

---
Alternatively, you can use virtualenv. For instructions on how to setup and use virtual environments please refer to [Virtualenv](https://realpython.com/python-virtual-environments-a-primer).

After setting up your virtualenv, to install and use the NeuroMechFly library follow the abovementioned procedure in your active python environment.
