""" CPG locomotion controller. """
import itertools
import os
from argparse import ArgumentParser
from pathlib import Path

import farms_pylog as pylog
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from farms_container import Container
from farms_network.networkx_model import NetworkXModel
from farms_network.neural_system import NeuralSystem

pylog.set_level("error")


def add_mutual_connection(network, node_1, node_2, weight, phi):
    """
    Add mutual connection between two nodes
    """
    network.add_edge(node_1, node_2, weight=weight, phi=phi)
    network.add_edge(node_2, node_1, weight=weight, phi=-1*phi)


def add_connection_antagonist(network, node_1, node_2, **kwargs):
    """
    Add mutual connection between two nodes
    """
    weight = kwargs.pop('weight', 1.0)
    phi = kwargs.pop('phi', 0.0)

    add_mutual_connection(
        network, f"{node_1}_flexion", f"{node_2}_flexion", weight=weight,
        phi=phi
    )
    add_mutual_connection(
        network, f"{node_1}_extension", f"{node_2}_extension", weight=weight,
        phi=phi
    )


def create_oscillator_network(export_path, **kwargs):
    """Create the drosophila reduced network.
    """
    # Network properties
    default_weight = kwargs.pop("default_weight", 100.0)
    default_phi = kwargs.pop("default_phi", 0.0)
    # Initialize di graph network
    network = nx.DiGraph()
    # Generate list of controlled joints in the model
    sides = ('L', 'R')
    positions = ('F', 'M', 'H')
    segments = ('Coxa', 'Femur', 'Tibia')
    nodes = [
        f"joint_{side}{position}{segment}_roll"
        if (position in ["M", "H"]) and (segment == "Coxa")
        else f"joint_{side}{position}{segment}"
        for side in sides
        for position in positions
        for segment in segments
    ]
    # Create flexion-extension oscillator for each node
    for node in nodes:
        network.add_node(f"{node}_flexion", model="oscillator", f=3.0,
                         R=1.0, a=1.0)
        network.add_node(f"{node}_extension", model="oscillator", f=3.0,
                         R=1.0, a=1.0)
    # Connect flexion-extension nodes
    for node in nodes:
        if node.split("_")[-1][2:] not in ['Femur', 'Tibia']:
            add_mutual_connection(
                network, f"{node}_flexion", f"{node}_extension",
                weight=default_weight, phi=np.pi
            )
    # Connect leg oscillators
    for side in sides:
        for position in positions:
            for j in range(len(segments[:-1])):
                node_1 = segments[j]
                node_2 = segments[j+1]
                if (position in ["M", "H"]) and (segments[j] == "Coxa"):
                    node_1 = "Coxa_roll"
                add_mutual_connection(
                    network, f"joint_{side}{position}{node_1}_flexion",
                    f"joint_{side}{position}{node_2}_flexion",
                    weight=default_weight, phi=np.pi/2
                )
                add_mutual_connection(
                    network, f"joint_{side}{position}{node_1}_extension",
                    f"joint_{side}{position}{node_2}_extension",
                    weight=default_weight, phi=np.pi/2
                )
    #: Connect base nodes
    base_connections = [
        ['LFCoxa', 'RFCoxa', {'weight': default_weight, 'phi': np.pi}],
        ['LFCoxa', 'RMCoxa_roll', {'weight': default_weight, 'phi': np.pi}],
        ['RMCoxa_roll', 'LHCoxa_roll', {'weight': default_weight, 'phi': 0.0}],
        ['RFCoxa', 'LMCoxa_roll', {'weight': default_weight, 'phi': np.pi}],
        ['LMCoxa_roll', 'RHCoxa_roll', {'weight': default_weight, 'phi': 0.0}],
    ]
    for n1, n2, data in base_connections:
        add_connection_antagonist(network, f"joint_{n1}", f"joint_{n2}",
                                  **data)
    # Update node positions for visualization
    with open('locomotion_network_node_positions.yaml', 'r') as file:
        node_positions = yaml.load(file, yaml.SafeLoader)
    for node, data in node_positions.items():
        network.nodes[node]['x'] = data[0]
        network.nodes[node]['y'] = data[1]
        network.nodes[node]['z'] = data[2]
    # Export graph
    print(export_path)
    nx.write_graphml(network, export_path)


def run_network(network_path):
    """ Run the network.

    Parameters
    ----------
    network_path : <Path>
        Path to the network config file
    """
    # Initialize network
    dt = 1e-3  #: Time step (1ms)
    duration = 2
    time_vec = np.arange(0, duration, dt)  #: Time
    container = Container(duration/dt)
    net = NeuralSystem(network_path, container)

    # initialize network parameters
    container.initialize()
    net.setup_integrator()

    #: Integrate the network
    pylog.debug('Begin Integration!')

    for t in time_vec:
        net.step(dt=dt)
        container.update_log()

    #: Results
    container.dump(overwrite=True)

    # Plot results
    neural_data = container.neural
    neural_outputs = neural_data.outputs.log
    neural_outputs_names = neural_data.outputs.names
    neural_outputs_name_id = neural_data.outputs.name_index
    # Plot Intra-limb activations
    for leg in ("RF", "RM", "RH", "LH", "LM", "LH"):
        leg_data = np.asarray(
            [
                neural_outputs[:, neural_outputs_name_id[name]]
                for name in neural_outputs_names
                if leg in name
            ]
        ).T
        leg_names = [
            name for name in neural_outputs_names
            if leg in name
        ]
        fig, axs = plt.subplots(nrows=3, ncols=1)
        axs[0].plot(time_vec, 1 + np.sin(leg_data[:, :2]))
        axs[1].plot(time_vec, 1 + np.sin(leg_data[:, 2:4]))
        axs[2].plot(time_vec, 1 + np.sin(leg_data[:, 4:]))
        axs[0].axes.xaxis.set_visible(False)
        axs[1].axes.xaxis.set_visible(False)
        axs[0].set_title(leg_names[0].split('_')[2])
        axs[1].set_title(leg_names[2].split('_')[2])
        axs[2].set_title(leg_names[4].split('_')[2])
        axs[2].set_xlabel("Time[s]")
    # Plot Inter-limb activations
    leg_data = np.asarray(
        [
            neural_outputs[:, neural_outputs_name_id[name]]
            for name in neural_outputs_names
            if "Coxa" in name and "flexion" in name
        ]
    ).T
    leg_names = [
        name for name in neural_outputs_names
        if "Coxa" in name
    ]
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(time_vec, 1 + np.sin(leg_data[:, :]))
    ax.set_title("Coxa")
    ax.set_xlabel("Time[s]")

    #: Show network
    net.visualize_network(edge_labels=False)
    plt.show()


def parse_args():
    """Parse command line arguments to generate and simulate the network.
    """
    parser = ArgumentParser("Network parser")
    parser.add_argument(
        "--export-path", required=False, type=str,
        default=(
            Path(__file__).parent.absolute()
        ).joinpath("../config/network/locomotion_network.graphml"),
        dest="export_path"
    )
    parser.add_argument(
        "--run-network", required=False, type=bool,
        default=True, dest="run_network"
    )
    return parser.parse_args()


if __name__ == '__main__':
    # main()
    clargs = parse_args()
    create_oscillator_network(clargs.export_path)
    if clargs.run_network:
        run_network(clargs.export_path)
