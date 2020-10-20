""" Generate a template network. """

import farms_pylog as pylog
import networkx as nx
from farms_network.neural_system import NeuralSystem
from farms_sdf.sdf import ModelSDF
from farms_sdf import utils as sdf_utils
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from farms_container import Container

pylog.set_level('debug')


class AgnosticBaseController:
    """Base class for generating model specific neural control.
    """

    def __init__(self, controller_type, sdf_model_path):
        super(AgnosticBaseController, self).__init__()
        self.controller_type = controller_type
        self.model_path = sdf_model_path
        self.network = nx.DiGraph()
        self.model = AgnosticBaseController.read_sdf(
            self.model_path)[0]

    @staticmethod
    def read_sdf(sdf_path):
        """Read an sdf model
        Parameters
        ----------
        sdf_path: <str>
            File path to the sdf model
        """
        return ModelSDF.read(sdf_path)

    def export_network(self, pathname='graph.graphml'):
        """
        Method to export the generated network
        Parameters
        ----------
        pathname : <str>
            File path to save the file


        Returns
        -------
        out : <None>

        """
        nx.write_graphml(self.network, pathname)

    def generate_edges(self):
        """Connect edges of neurons in the network.

        Parameters
        ---------

        Returns
        -------
        out :

        """
        pylog.error('generate_neurons method not implemented in child class')
        raise NotImplementedError

    def generate_neurons(self, neuron_type, neuron_defaults):
        """Abstract class to generate neurons for each joint in the model.

        Parameters
        ----------
        self :

        neuron_type :

        neuron_defaults :


        Returns
        -------
        out :

        """
        pylog.error('generate_neurons method not implemented in child class')
        raise NotImplementedError


class AgnosticPositionController(AgnosticBaseController):
    """Class to generate a position based oscillator controller.
    """

    def __init__(self, sdf_model_path):
        super(AgnosticPositionController, self).__init__(
            controller_type='POSITION_CONTROL',
            sdf_model_path=sdf_model_path
        )

    def generate_neurons(
            self, neuron_type='oscillator', **kwargs
    ):
        """ Generate neuron for the mode. """
        joints = self.model.joints
        links = self.model.links
        link_id = sdf_utils.link_name_to_index(self.model)
        for joint in joints:
            self.network.add_node(
                joint.name,
                model='oscillator',
                **kwargs,
                x=links[link_id[joint.child]].pose[0],
                y=links[link_id[joint.child]].pose[1],
                z=links[link_id[joint.child]].pose[2],
            )

    def generate_edges(
            self, couple_closest_neighbour=True, couple_base=True,
            **kwargs
    ):
        """

        Parameters
        ----------
        self :

        couple_closest_neighbor_oscillators :

        couple_base :


        Returns
        -------
        out :

        """
        if couple_closest_neighbour:
            AgnosticPositionController.couple_closest_neighbor(
                self.network,
                self.model,
                **kwargs
            )
        if couple_base:
            AgnosticPositionController.couple_base(
                self.network,
                self.model,
                **kwargs
            )

    @staticmethod
    def couple(network, node_1, node_2, **kwargs):
        """
        Add mutual connection between two nodes
        """

        weight = kwargs.pop('weight', 1.0)
        phi = kwargs.pop('phi', 0.0)

        network.add_edge(
            node_1,
            node_2,
            weight=weight,
            phi=phi
        )
        network.add_edge(
            node_2,
            node_1,
            weight=weight,
            phi=-1*phi
        )

    @staticmethod
    def couple_closest_neighbor(network, model, **kwargs):
        """ Add connections to closest neighbors. """
        weight = kwargs.pop('weight', 1.0)
        phi = kwargs.pop('phi', 0.0)

        for joint in model.joints:
            for conn in sdf_utils.find_neighboring_joints(
                    model, joint.name):
                AgnosticPositionController.couple(
                    network,
                    joint.name,
                    conn,
                    weight=weight,
                    phi=phi
                )

    @staticmethod
    def couple_base(network, model, **kwargs):
        """ Add connection between base nodes. """
        weight = kwargs.pop('weight', 1.0)
        phi = kwargs.pop('phi', 0.0)

        root_link = sdf_utils.find_root(model)
        base_joints = []
        for joint in model.joints:
            if joint.parent == root_link:
                base_joints.append(joint.name)
        for j1, j2 in itertools.combinations(base_joints, 2):
            AgnosticPositionController.couple(
                network,
                j1,
                j2,
                weight=weight,
                phi=phi
            )


class AgnosticController:
    """Base class for generating model specific neural control.
    """

    def __init__(
            self,
            sdf_path,
            connect_mutual=True,
            connect_closest_neighbors=True,
            connect_base_nodes=True,
            remove_joint_types=[]
    ):
        super().__init__()
        self.model = self.read_sdf(sdf_path)[0]
        #: Remove certain joint types
        for j in remove_joint_types:
            self.model.joints = sdf_utils.remove_joint_type(
                self.model, j)
        self.connect_flexion_extension = connect_mutual
        self.connect_closest_neighbors = connect_closest_neighbors
        self.connect_base_nodes = connect_base_nodes
        #: Define a network graph
        self.network = nx.DiGraph()
        #: Generate the basic network
        self.generate_network()

    @staticmethod
    def read_sdf(sdf_path):
        """Read sdf model
        Keyword Arguments:
        sdf_path --
        """
        return ModelSDF.read(sdf_path)

    @staticmethod
    def add_mutual_connection(network, node_1, node_2, weight, phi):
        """
        Add mutual connection between two nodes
        """
        network.add_edge(
            node_1,
            node_2,
            weight=weight,
            phi=phi
        )
        network.add_edge(
            node_2,
            node_1,
            weight=weight,
            phi=-1*phi
        )

    @staticmethod
    def add_connection_antagonist(network, node_1, node_2, **kwargs):
        """
        Add mutual connection between two nodes
        """
        weight = kwargs.pop('weight', 1.0)
        phi = kwargs.pop('phi', 0.0)        
        
        AgnosticController.add_mutual_connection(
            network,
            node_1 + '_flexion',
            node_2 + '_flexion',
            weight=weight,
            phi=phi
        )
        AgnosticController.add_mutual_connection(
            network,
            node_1 + '_extension',
            node_2 + '_extension',
            weight=weight,
            phi=phi
        )

    @staticmethod
    def add_connection_to_closest_neighbors(network, model, weight):
        """ Add connections to closest neighbors. """
        for joint in model.joints:
            for conn in sdf_utils.find_neighboring_joints(
                    model, joint.name):
                print("{} -> {}".format(joint.name, conn))
                AgnosticController.add_mutual_connection(
                    network,
                    joint.name + '_flexion',
                    conn + '_flexion',
                    weight=weight,
                    phi=np.pi/2
                )
                AgnosticController.add_mutual_connection(
                    network,
                    joint.name + '_extension',
                    conn + '_extension',
                    weight=weight,
                    phi=np.pi/2
                )

    @staticmethod
    def add_connection_between_base_nodes(network, model, weight):
        """ Add connection between base nodes. """
        root_link = sdf_utils.find_root(model)
        base_joints = []
        for joint in model.joints:
            if joint.parent == root_link:
                base_joints.append(joint.name)
        for j1, j2 in itertools.combinations(base_joints, 2):
            AgnosticController.add_mutual_connection(
                network,
                j1 + '_flexion',
                j2 + '_flexion',
                weight=weight,
                phi=0.0
            )
            AgnosticController.add_mutual_connection(
                network,
                j1 + '_extension',
                j2 + '_extension',
                weight=weight,
                phi=0.0
            )

    def generate_network(self):
        """Generate network
        Keyword Arguments:
        self --
        """
        links = self.model.links
        link_id = sdf_utils.link_name_to_index(self.model)
        weight = 5000.0
        #: Add two neurons to each joint and connect each other
        for joint in self.model.joints:
            self.network.add_node(
                joint.name + '_flexion',
                model='oscillator',
                f=3,
                R=1.0,
                a=25,
                x=links[link_id[joint.child]].pose[0]+0.001,
                y=links[link_id[joint.child]].pose[1] +
                links[link_id[joint.child]].pose[2],
                z=links[link_id[joint.child]].pose[2],
            )
            self.network.add_node(
                joint.name + '_extension',
                model='oscillator',
                f=3,
                R=1.0,
                a=25,
                x=links[link_id[joint.child]].pose[0]-0.001,
                y=links[link_id[joint.child]].pose[1] +
                links[link_id[joint.child]].pose[2],
                z=links[link_id[joint.child]].pose[2],
            )
            if self.connect_flexion_extension:
                AgnosticController.add_mutual_connection(
                    self.network,
                    joint.name + '_flexion',
                    joint.name + '_extension',
                    weight=weight,
                    phi=np.pi
                )

        #: Connect neurons to closest neighbors
        if self.connect_closest_neighbors:
            pylog.debug("Connecting closest neighbors")
            AgnosticController.add_connection_to_closest_neighbors(
                self.network,
                self.model,
                weight=weight
            )

        #: Connect neurons between the base nodes
        if self.connect_base_nodes:
            pylog.debug("Connecting base nodes")
            AgnosticController.add_connection_between_base_nodes(
                self.network,
                self.model,
                weight=weight
            )


def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../../farms_blender/animats/"
         "mouse_v1/design/sdf/mouse_locomotion.sdf"),
    )
    net_dir = "../config/mouse_locomotion.graphml"
    nx.write_graphml(controller_gen.network, net_dir)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 1
    time_vec = np.arange(0, dur, dt)  #: Time
    container = Container(dur/dt)
    net = NeuralSystem(
        "../config/mouse_locomotion.graphml",
        container)
    #: initialize network parameters
    container.initialize()
    net.setup_integrator()

    #: Integrate the network
    pylog.info('Begin Integration!')

    for t in time_vec:
        net.step(dt=dt)
        container.update_log()

    #: Results
    # container.dump()
    state = np.asarray(container.neural.states.log)
    neuron_out = np.asarray(container.neural.outputs.log)
    names = container.neural.outputs.names
    parameters = container.neural.parameters
    #: Show graph
    print(net.graph.number_of_edges())
    net.visualize_network(edge_labels=False)
    nosc = net.network.graph.number_of_nodes()
    plt.figure()
    for j in range(nosc):
        plt.plot((state[:, 2*j+1]*np.sin(neuron_out[:, j])))
    plt.legend(names)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
