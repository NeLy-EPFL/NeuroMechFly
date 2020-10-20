#!/usr/bin/env python3
"""Class to generate DAE system."""

from .table import Table
import farms_pylog as pylog
import shutil
import os
import pandas as pd


class Container:
    """ Main Data Container Class. """

    def __init__(self, max_iterations=1):
        """ Initialization. """
        super().__init__()
        self.__max_iterations = max_iterations
        if self.__max_iterations == 1:
            pylog.info("LOGGING of Data is Disbaled!")
            pylog.info("Max Iterations set to {}".format(
                self.__max_iterations))

    def __del__(self):
        print("Deleting container....")

    def add_namespace(self, name):
        """Add a new namespace to the container
        Parameters
        ----------
        name : <str>
            Name of the new namespace to be defined
        Returns
        -------
        namespace : <DataContainer>
            Return the created namespace
        """
        if hasattr(self, name):
            pylog.error(
                "Trying to recreate {} namespace that already exists!".format(name))
            raise ValueError
        setattr(self, name, DataTable(name, self.__max_iterations))
        pylog.info("Created new container namespace : {}".format(name))
        return getattr(self, name)

    def initialize(self):
        """Initialize container
        Keyword Arguments:
        self --
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, DataTable):
                attr.initialize()

    def update_log(self):
        """update container log
        Keyword Arguments:
        self --
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, DataTable):
                attr.update_log()

    def dump(self, dump_path='./Results', overwrite=None):
        """Dump all the data from the container to the disk.

        Parameters
        ----------
        dump_path : <str, optional>
        Path on the disk to dump all the data from the container.
        If not specified by default the functions tries to write to the
        current directory with Results as the folder. If the Results folder
        already exists then a user response is asked if it needs to be
        overwritten and save with new name or just quit

        Returns
        -------
        out : <bool>
         Output True if dumping is successfull
        """
        #: Check if the given folder is valid
        (path, folder_name) = os.path.split(dump_path)
        if not os.path.isdir(path):
            pylog.error("Provided path is invalid - {}".format(dump_path))
            raise ValueError
        if os.path.isdir(dump_path):
            pylog.info("Folder {} already exists.".format(folder_name))
            if overwrite is None:
                overwrite = input(
                    "Do you want overwrite the exisiting folder {}?"
                    "[y]/[n]\n".format(folder_name))
                overwrite = True if overwrite.lower()=='y' else False
            if overwrite:
                pylog.info("Deleting Folder {}".format(folder_name))
                shutil.rmtree(dump_path)
            else:
                folder_name = input(
                    'New name for the dump folder?\n')
        #: CREATE FOLDERS
        dump_path = os.path.join(path, folder_name)
        os.mkdir(dump_path)
        #: Create a sub-directory for each namespace in the container
        #: Save the data for each table in the container
        for nname, namespace in self.__dict__.items():
            if isinstance(namespace, DataTable):
                namespace_folder = os.path.join(dump_path, nname)
                os.mkdir(namespace_folder)
                for tname, table in namespace.__dict__.items():
                    if isinstance(table, Table):
                        data = pd.DataFrame(table.log)
                        data.columns = table.names
                        data.to_hdf(
                            os.path.join(namespace_folder, tname+'.h5'),
                            tname, mode='w')
        return True


class DataTable(dict):
    """Data Container class
    """

    def __init__(self, name, max_iterations):
        super().__init__()
        self.name = name
        self.max_iterations = max_iterations

    def add_table(self, name, table_type='VARIABLE'):
        """ Add new type parameter for book keeping

        Parameters
        ----------
        name : <str>
            Name of the new parameter
        TABLE_TYPE : <str> , optional
            Type of the table
            1. If set to VARIABLE then data logging is enabled given that
        maximum iterations is set to Non-Zero
           2. If set to CONSTANT then data logging is disabled irrespective
        that maximum iterations is set to zero or not
        Returns
        -------
        parameters : <Parameters>
            Return the parameters table
        """
        if hasattr(self, name):
            pylog.error(
                "Trying to recreate {} table that already exists!".format(name))
            raise ValueError
        setattr(
            self, name,
            Table(name, table_type, self.max_iterations))
        pylog.info(
            "Created new table {} in namespaces {}".format(name, self.name))
        return getattr(self, name)

    def initialize(self):
        """ Initialize all the tables  """
        for name, attr in self.__dict__.items():
            if isinstance(attr, Table):
                attr.initialize_table()

    def update_log(self):
        """update container log
        Keyword Arguments:
        self --
        """
        for name, attr in self.__dict__.items():
            if isinstance(attr, Table):
                attr.update_log()
