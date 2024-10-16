## Installing NeuroMechFly
To avoid any conflicts of python packages with your existing python environment, we highly recommend to use virtualenv or conda env. To create a conda environment, follow the following steps: 

**First, make sure that you have git-lfs (large file storage) installed on your local machine. Otherwise, please refer to this [link](https://github.com/git-lfs/git-lfs) to learn more about how to install the Git LFS.**

Then, you can download the repository on your local machine by running the following line in the terminal. If you are interested in downloading the blender model, run `git lfs clone` instead of `git clone`.
```bash
$ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
```
After the download is complete, navigate to the NeuroMechFly folder:
```bash
$ cd NeuroMechFly
```
In this folder, run the following commands to create a virtual environment and activate it:
```bash
$ conda create -n neuromechfly python=3.6 numpy Cython shapely
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

**NOTE:** Microsoft Visual Studio C++ 14.0 or better is required for the proper installation of farms_container.

---
**NOTE:**
Each time you start using NeuroMechFly, please activate virtual environment by running: 
```bash
$ conda activate neuromechfly
```

---
Alternatively, you can use virtualenv. For instructions on how to setup and use virtual environments please refer to [Virtualenv](https://realpython.com/python-virtual-environments-a-primer).

After setting up your virtualenv, to install and use the NeuroMechFly library follow the abovementioned procedure in your active python environment.
