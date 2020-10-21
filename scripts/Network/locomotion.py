""" cpg locomotion controller. """

import networkx as nx
import os
from NeuroMechFly.network.neural_system import NeuralSystem
import numpy as np
import matplotlib.pyplot as plt
import itertools
from NeuroMechFly.container import Container
from NeuroMechFly.sdf.sdf import ModelSDF
from NeuroMechFly.network.agnostic_controller import AgnosticController
import yaml

def main():
    """ Main. """
    controller_gen = AgnosticController(
        ("../../design/sdf/drosophila_100x_noLimits.sdf"),
        connect_mutual=False,
        connect_closest_neighbors=False,
        connect_base_nodes=False
    )
    net_dir = "../../config/locomotion_test.graphml"
    network = controller_gen.network
    #: EDIT THE GENERIC CONTROLLER
    #: Remove Head nodes
    network.remove_nodes_from(['joint_Head_flexion',
                               'joint_Head_extension',
                               #'joint_HeadFake1_flexion',
                               #'joint_HeadFake1_extension',
                               'joint_Proboscis_flexion',
                               'joint_Proboscis_extension',
                               'joint_Labellum_flexion',
                               'joint_Labellum_extension',
                               'joint_LAntenna_flexion',
                               'joint_LAntenna_extension',
                               'joint_RAntenna_flexion',
                               'joint_RAntenna_extension',
                               'joint_LEye_flexion',
                               'joint_LEye_extension',
                               'joint_REye_flexion',
                               'joint_REye_extension'])

    #: Remove Abdomen nodes
    network.remove_nodes_from(['joint_A1A2_flexion',
                               'joint_A1A2_extension',
                               'joint_A3_flexion',
                               'joint_A3_extension',
                               'joint_A4_flexion',
                               'joint_A4_extension',
                               'joint_A5_flexion',
                               'joint_A5_extension',                               
                               'joint_A6_flexion',
                               'joint_A6_extension'])

    #: Remove node for supporting nodes
    network.remove_nodes_from(['prismatic_support_1_flexion',
                               'prismatic_support_1_extension',
                               'prismatic_support_2_flexion',
                               'prismatic_support_2_extension',
                               'revolute_support_1_flexion',
                               'revolute_support_1_extension'])

    #: Remove wings and haltere nodes
    for node in ['','_roll','_yaw']:
        LwingNode = 'joint_LWing'+node
        RwingNode = 'joint_RWing'+node
        LhaltereNode = 'joint_LHaltere'+node
        RhaltereNode = 'joint_RHaltere'+node
        
        network.remove_nodes_from([LwingNode+'_flexion',
                                   RwingNode+'_flexion',
                                   LwingNode+'_extension',
                                   RwingNode+'_extension',
                                   LhaltereNode+'_flexion',
                                   RhaltereNode+'_flexion',
                                   LhaltereNode+'_extension',
                                   RhaltereNode+'_extension'])

    #: Remove tarsi nodes
    for i in range(1,6):
        for tarsus in ['LF','LM','LH','RF','RM','RH']:
            tarsusNode = 'joint_'+tarsus+'Tarsus'+str(i)
            network.remove_nodes_from([tarsusNode+'_flexion',
                                       tarsusNode+'_extension'])

    #: Remove Coxa and femur extra DOF nodes
    for side in ['LF','LM','LH','RF','RM','RH']:
        for segment in ['Coxa','Femur']:
            pitchNode = 'joint_'+side+segment
            rollNode = 'joint_'+side+segment+'_roll'
            yawNode = 'joint_'+side+segment+'_yaw'
            if segment == 'Coxa':                 
                network.remove_nodes_from([yawNode+'_flexion',
                                           yawNode+'_extension'])
                if 'F' in side:
                    network.remove_nodes_from([rollNode+'_flexion',
                                           rollNode+'_extension'])
                else:
                    network.remove_nodes_from([pitchNode+'_flexion',
                                           pitchNode+'_extension'])
            if segment == 'Femur': 
                network.remove_nodes_from([rollNode+'_flexion',
                                           rollNode+'_extension'])          

    
    #: Connect limbs
    # AgnosticController.add_mutual_connection(
    #     network,
    #     'LHip_flexion',
    #     'RHip_flexion',
    #     weight=10.0,
    #     phi=np.pi
    # )

    with open('network_node_positions.yaml', 'r') as file:
        node_positions = yaml.load(file, yaml.SafeLoader)
    for node, data in node_positions.items():
        network.nodes[node]['x'] = data[0]
        network.nodes[node]['y'] = data[1]
        network.nodes[node]['z'] = data[2]

    #: EDIT CONNECTIONS FOR TRIPOD GAIT
    #: Connecting base nodes
    weight = 1000.0
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
    
    # for joint in controller_gen.model.joints:
    #     n1 = '{}_flexion'.format(joint.name)
    #     n2 = '{}_extension'.format(joint.name)        
    #     network.remove_edges_from([(n1, n2), (n2, n1)])
        
    nx.write_graphml(network, net_dir)
    
    #: Export position file to yaml
    # with open('../config/network_node_positions.yaml', 'w') as file:
    #     yaml.dump(node_positions, file, default_flow_style=True)

    # #: Initialize network
    dt = 0.001  #: Time step
    dur = 10
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

    #: Show network
    net.visualize_network(edge_labels=False)
    plt.show()

if __name__ == '__main__':
    main()
