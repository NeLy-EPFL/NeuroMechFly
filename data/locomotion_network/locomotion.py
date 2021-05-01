""" CPG locomotion controller. """

import itertools
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import yaml

from farms_container import Container
from farms_network.utils.agnostic_controller import AgnosticController
from farms_network.neural_system import NeuralSystem
from NeuroMechFly.sdf.sdf import ModelSDF


def main():
    """ Main. """

    legs = ('LF','LM','LH','RF','RM','RH')

    joints_to_remove = [
        # Head
        *['joint_Head', 'joint_Head_yaw', 'joint_Head_roll',
          'joint_Proboscis', 'joint_Labellum',
          'joint_LAntenna', 'joint_RAntenna', 'joint_LEye',
          'joint_REye', 'joint_Haustellum', 'joint_Rostrum'
          ],
        # Abdomen
        *['joint_A1A2', 'joint_A3', 'joint_A4', 'joint_A5', 'joint_A6'],
        # Body support
        *['prismatic_support_1', 'prismatic_support_2', 'revolute_support_1'],
        # Wings and Haltere
        *[
            f'joint_{side}{joint}{node}'
            for node in ('', '_roll', '_yaw')
            for joint in ('Wing', 'Haltere')
            for side in ('L', 'R')
        ],
        # Remove Tarsus nodes
        *[
            f'joint_{leg}Tarsus{i}'
            for i in range(1,6)
            for leg in legs
        ]
    ]

    import pprint
    pprint.pprint(joints_to_remove)

    controller_gen = AgnosticController(
        ("../../design/sdf/neuromechfly_limitsFromData.sdf"),
        connect_mutual=False,
        connect_closest_neighbors=False,
        connect_base_nodes=False,
        remove_joints=joints_to_remove
    )
    net_dir = "../../config/locomotion_tripod.graphml"
    network = controller_gen.network
    #: EDIT THE GENERIC CONTROLLER

    #: Remove Coxa and femur extra DOF nodes
    for leg in legs:
        for segment in ['Coxa','Femur']:
            joint = f"joint_{leg}{segment}"
            if segment == 'Coxa':
                network.remove_nodes_from(
                    [joint+'_yaw_flexion', joint+'_yaw_extension'])
                if 'F' in leg:
                    network.remove_nodes_from(
                        [joint+'_roll_flexion', joint+'_roll_extension'])
                else:
                    network.remove_nodes_from(
                        [joint+'_flexion', joint+'_extension'])
            if segment == 'Femur':
                network.remove_nodes_from(
                    [joint+'_roll_flexion', joint+'_roll_extension'])


    with open('network_node_positions.yaml', 'r') as file:
        node_positions = yaml.load(file, yaml.SafeLoader)
    for node, data in node_positions.items():
        network.nodes[node]['x'] = data[0]
        network.nodes[node]['y'] = data[1]
        network.nodes[node]['z'] = data[2]

    #: EDIT CONNECTIONS FOR TRIPOD GAIT
    #: Connecting base nodes
    weight = 100.0
    base_connections = [
        ['LFCoxa', 'RFCoxa', {'weight':weight, 'phi': np.pi}],
        ['LFCoxa', 'RMCoxa_roll', {'weight':weight, 'phi': np.pi}],
        ['RMCoxa_roll', 'LHCoxa_roll', {'weight':weight, 'phi': 0.0}],
        ['RFCoxa', 'LMCoxa_roll', {'weight':weight, 'phi': np.pi}],
        ['LMCoxa_roll', 'RHCoxa_roll', {'weight':weight, 'phi': 0.0}],
    ]

    for n1, n2, data in base_connections:
        AgnosticController.add_connection_antagonist(
            network,
            'joint_{}'.format(n1),
            'joint_{}'.format(n2),
            **data
        )

    leg_connections = [
        ['Coxa', 'Femur', {'weight':weight, 'phi': np.pi}],
        ['Femur', 'Tibia', {'weight':weight, 'phi': np.pi}],
    ]

    for n1, n2, data in leg_connections:
        for pos in ['F', 'M', 'H']:
            for side in ['L', 'R']:
                if (pos == 'M' or pos == 'H') and (n1 == 'Coxa'):
                    n1 = 'Coxa_roll'
                AgnosticController.add_connection_antagonist(
                    network,
                    'joint_{}{}{}'.format(side, pos, n1),
                    'joint_{}{}{}'.format(side, pos, n2),
                    **data
                )

    coxa_connections = [
        ['Coxa', 'Coxa', {'weight':weight, 'phi': np.pi}],
    ]

    for n1, n2, data in coxa_connections:
        for pos in ['F', 'M', 'H']:
            for side in ['L', 'R']:
                if (pos == 'M' or pos == 'H'):
                    n1 = 'Coxa_roll'
                    n2 = 'Coxa_roll'
                AgnosticController.add_mutual_connection(
                    network,
                    'joint_{}{}{}_{}'.format(side, pos, n1, 'flexion'),
                    'joint_{}{}{}_{}'.format(side, pos, n2, 'extension'),
                    **data
                )

    nx.write_graphml(network, net_dir)

    #: Export position file to yaml
    # with open('../config/network_node_positions.yaml', 'w') as file:
    #     yaml.dump(node_positions, file, default_flow_style=True)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 2
    time_vec = np.arange(0, dur, dt)  #: Time
    container = Container(dur/dt)
    net = NeuralSystem(
        net_dir,
        container)

    #: initialize network parameters
    container.initialize()
    net.setup_integrator()

    #: Integrate the network
    print('Begin Integration!')

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
    for leg in legs:
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

if __name__ == '__main__':
    main()
