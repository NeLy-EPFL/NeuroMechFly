## Installing NeuroMechFly

To avoid any conflicts of python packages with your existing python
environment, we highly recommend to use virtualenv or conda env.

For instructions on how to setup and use virtual environments please
refer to
[Virtualenv](https://realpython.com/python-virtual-environments-a-primer)
and [Condaenv](https://docs.conda.io/projects/conda/en/latest/commands/create.html)

After setting up your environment, to install and use the NeuroMechFly
library run the following commands in your active python environment,

```bash
$ pip install git+https://gitlab.com/FARMSIM/farms_container.git
$ pip install git+https://github.com/NeLy-EPFL/NeuroMechFly.git@secondary-master#egg=NeuroMechFly
```

To install the library in developer mode use `-e` option,

```bash
$ pip install git+https://gitlab.com/FARMSIM/farms_container.git
$ pip install -e git+https://github.com/NeLy-EPFL/NeuroMechFly.git@secondary-master#egg=NeuroMechFly

<!-- First, you can download the repository on your local machine by running the following line in the terminal: -->
<!-- ```bash -->
<!-- $ git clone https://github.com/NeLy-EPFL/NeuroMechFly.git -->
<!-- ``` -->
<!-- After the download is complete, navigate to the NeuromMechFly folder: -->
<!-- ```bash -->
<!-- $ cd NeuroMechFly -->
<!-- ``` -->
<!-- In this folder, run the following commands to create a virtual environment: -->
<!-- ```bash -->
<!-- $ conda create -n neuromechfly python=3.6 numpy Cython -->
<!-- ``` -->
<!-- Finally, install all the dependencies by running: -->
<!-- ```bash -->
<!-- $ pip install -e . -->
<!-- ``` -->
<!-- Once you complete all the steps, NeuroMechFly is ready to use! -->
