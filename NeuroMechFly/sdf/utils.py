""" Utility classes and methods for farms_sdf """

import os
from dataclasses import dataclass
from treelib import Tree

def replace_file_name_in_path(file_path, new_name):
    """
    Replace a file name in a given path. File extension is retained
 
    Parameters
    ----------
    file_path : <str>
        Path to the file object
    new_name : <str>
        Name to replace the original file name
    Returns
    -------
    out : <str>
        New path with the replaced file name
    """
    full_path = os.path.split(file_path)[0]
    file_extension = (os.path.split(file_path)[-1]).split('.')[-1]
    new_name = new_name + '.' + file_extension
    return os.path.join(full_path, new_name)

def link_name_to_index(model):
    """ Generate a dictionary for link names and their indicies in the
    model. """
    return {
        link.name : index for index, link in enumerate(model.links)
    }

def joint_name_to_index(model):
    """ Generate a dictionary for link names and their indicies in the
    model. """
    return {
        joint.name : index for index, joint in enumerate(model.joints)
    }

def find_parent_joints(model, joint_name):
    """ Find all the joints parented to the given joint. """
    joint_id = joint_name_to_index(model)
    link_id = link_name_to_index(model)
    #: FUCK : Add exception to catch invalid joint names
    joint = model.joints[joint_id[joint_name]]
    plink = joint.parent
    return [
        j.name for j in model.joints if j.child == plink
    ]

def find_child_joints(model, joint_name):
    """ Find all the joints parented to the given joint. """
    joint_id = joint_name_to_index(model)
    link_id = link_name_to_index(model)
    #: FUCK : Add exception to catch invalid joint names
    joint = model.joints[joint_id[joint_name]]
    clink = joint.child
    return [
        j.name for j in model.joints if j.parent == clink
    ]

def find_neighboring_joints(model, joint):
    """ Find both parent and child neighboring joints. """
    return (
        find_parent_joints(model, joint) + \
        find_child_joints(model, joint)
    )

def find_link_joints(model, link_name):
    """Find the joints attached to a given link
 
    Parameters
    ----------
    model : <ModelSDF>
        SDF model

    link_name : <str>
        Name of the link in the sdf
    
    Returns
    -------
    out : <tuple>
        Tuple of joint names attached to the link
    """
    return tuple([
        joint.name
        for joint in model.joints
        if joint.parent == link_name
    ])

def find_root(model):
    """ Find the root link. """
    #: FUCK: CRUDE AND UNELEGANT SOLUTION
    for link in model.links:
        lname = link.name
        count = 0
        for joint in model.joints:
            if joint.child == lname:
                count += 1
        if count == 0:
            return link.name

@dataclass
class TreeData:
    """Data for SDF tree
    """
    joint: str
        
def add_nodes_to_tree(model, tree, links=None, joint_index=None):
    """ Add nodes to links """
    if joint_index is None:
        joint_index = joint_name_to_index(model)
    if len(links) == 0:
        return True
    #: New links to be added
    new_links = []
    for link in links:
        for joint_name in find_link_joints(model, link):
            joint = model.joints[joint_index[joint_name]]
            tree.create_node(
                tag=joint.child, identifier=joint.child,
                parent=joint.parent, data=TreeData(joint.name)
            )
            new_links.append(joint.child)
    add_nodes_to_tree(model, tree, new_links, joint_index)

def construct_tree(model) -> Tree:
    """ Construct tree. """   
    tree = Tree()
    root = find_root(model)
    tree.create_node(tag=root, identifier=root, parent=None, data=TreeData(""))
    joint_index = joint_name_to_index(model)
    add_nodes_to_tree(model, tree, links=[root], joint_index=joint_index)
    return tree

def get_all_subtrees(model):
    """ Get all the subtrees in the given model. """
    #: Construct the tree
    tree = construct_tree(model)
    #: Get the branch nodes
    branch_roots = [
        n.identifier
        for n in tree.all_nodes_itr()
        if len(tree.children(n.identifier)) > 1
    ]
    #: Get all subtrees    
    return [
        tree.remove_subtree(children.identifier)
        for root in branch_roots[::-1]
        for children in tree.children(root)
    ]
    
