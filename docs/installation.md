## Installing NeuroMechFly

First, you can download the repository on your local machine by running the following line in the terminal:
```bash
$ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git
```
After the download is complete, navigate to the NeuromMechFly folder:
```bash
$ cd NeuroMechFly
```
In this folder, run the following commands to create a virtual environment:
```bash
$ conda create -n neuromechfly python=3.6 numpy Cython
```
Finally, install all the dependencies by running:
```bash
$ pip install -e .
```
Once you complete all the steps, NeuroMechFly is ready to use!
