# Biomechanical module

Links and joints are defined in the ```sdf``` configuration files located in ```data/design/sdf/```, for a complete guide about the simulation description format (sdf) please refer to its [main page](http://sdformat.org/).

In this tutorial you can leanr how to:
- [Modify the body segments](Modifying-the-body-segments)
- [Modify the joints](Modifying-joints)
- [Change the pose](Changing-the-pose)


## Modifying the body segments

Every body segment in the biomechanical model is called a link. Each link is defined in the sdf configuration files considering its physical properties, such as: *pose, inertia, collision shape, and visual shape*. All these properties where obtained from the rendered CT-scan model. The *pose* consists of the translation and orientation of each link. The *inertia* has the mass and the inertial moments defined. The *collision* and *visual* shapes are the actual meshes rendered from the CT-scan model. Modifying the *pose* or *inertial* values for any link directly on the sdf file can lead to unstable simulations, therefore we do not recommend it.

**Changing visual/collision shape**

On the other hand, changing the *visual* and *collision* shapes is fairly easy while keeping the mass and intertial measurements untouched. For that you have to change the ```geometry``` definition in the ```collision``` and ```visual``` attributes for the link as we did in our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2) to change the morphology of the NeuroMechfly antenna to a stick model.

<table>
<tr>
<th>NeuroMechFly model</th>
<th>Antenna stick model</th>
</tr>
<tr>
<td>
  
```html
<collision name="LAntenna_collision">
  <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
  <geometry>
    <mesh>
      <uri>../meshes/stl/LAntenna.stl</uri>
      <scale>1000.0 1000.0 1000.0</scale>
    </mesh>
  </geometry>
</collision>
<visual name="LAntenna_visual">
  <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
  <geometry>
    <mesh>
      <uri>../meshes/stl/LAntenna.stl</uri>
      <scale>1000.0 1000.0 1000.0</scale>
    </mesh>
  </geometry>
  <material/>
</visual>
```
  
</td>
<td>
  
```html
<collision name="LAntenna_collision">
  <pose>0.0 0.0 -0.1485 0 0 0</pose>
  <geometry>
    <cylinder>
      <radius>0.06</radius>
      <length>0.297</length>
    </cylinder>
  </geometry>
</collision>
<visual name="LAntenna_visual">
  <pose>0.0 0.0 -0.1485 0 0 0</pose>
  <geometry>
    <cylinder>
      <radius>0.06</radius>
      <length>0.297</length>
    </cylinder>
  </geometry>
  <material/>
</visual>
```

</td>
</tr>
</table>

## Modifying joints

We have defined 90 joints (degrees-of-freedom) in NeuroMechFly. They are listed in Table 3 in our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2). 
In every sdf configuration file, joints are defined as follows:

```html
<joint name="joint_name" type="revolute">
  <parent>parent_link_name</parent>
  <child>child_link_name</child>
  <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
  <axis>
    <xyz>0.0 0.0 1.0</xyz>
    <limit>
      <lower>-3.14</lower>
      <upper>3.14</upper>
    </limit>
  </axis>
</joint>
```

**Removing joints**

Intuitevely one can though of removing a joint's definition to stop using it, however, that could also affect other things. NeuroMechFly biomechanics model is build as a kinematic chain, therefore, any change in a joint will affect all the links bellow it in the chain. That's why you should make sure to conserve a valid kinematic chain after removing a joint completely. Alternatevely, you can remove the actuation of that joint by setting its ```type``` variable to *fixed*. This will keep the joint static at its zero pose. Furthermore, you can refer to the [changing pose section](#changing-the-pose) below to know how to keep a certain position (different from the zero pose) during the simulation.

**Adding joints**

For adding a new joint you can copy the snippet above and replace the variables ```"joint_name"```, ```parent_link_name```, and ```child_link_name``` at convinience. This will generate a hinge-type (revolute) joint between the *parent* and the *child* links rotating around the *z* axis without limits. However, any joint addition implies to also update the ```parent_link_name```, and ```child_link_name``` variables in the *parent* and *child* links to preserve the kinematic chain unaffected.

**Modifying range of motion**

You can specify the range of motion for any joint by changing its ```lower``` and ```upper``` values in the ```limit``` property. For example, the next lines would set the joint limits for the Tibia-Tarsus joint in the left front leg to +/- 90° from its zero-pose.

```html
<joint name="joint_LFTarsus1" type="revolute">
  <parent>LFTibia</parent>
  <child>LFTarsus1</child>
  <pose>0.0 0.0 0.0 0.0 0.0 0.0</pose>
  <axis>
    <xyz>0.0 1.0 0.0</xyz>
    <limit>
      <lower>-1.57</lower>
      <upper>1.57</upper>
    </limit>
  </axis>
</joint>
```

## Changing the pose

Poses are defined in ```data/config/pose``` as *yaml* files. They consist of a joint's list with its desired angle in degrees, if a joint is not modified here it will keep its zero pose. For example, the next lines generate the *stretched pose*.

<table>
<tr>
<th>Stretched pose definition</th>
<th>Stretched pose</th>
</tr>
<tr>
<td>
  
```html
joints:
  joint_LFCoxa: 19
  joint_LFFemur: -130
  
  joint_LMCoxa_roll: 90
  joint_LMFemur: -100
  
  joint_LHCoxa_roll: 150
  joint_LHFemur: -100
  
  joint_RFCoxa: 19
  joint_RFFemur: -130
  
  joint_RMCoxa_roll: -90
  joint_RMFemur: -100
  
  joint_RHCoxa_roll: -150
  joint_RHFemur: -100
```
  
</td>
<td>
  
<p align="center">
  <img src="images/stretched_pose.png" width="350" />
</p>

</td>
</tr>
</table>

**Initial pose**

Initial poses are the poses used for the simulation only in the first time step. They are used for avoiding unwanted collisions when the model is created.
If no initial pose is defined for the simulation it will use the zero pose from the model, shown in Fig S7 in our related [publication](https://www.biorxiv.org/content/10.1101/2021.04.17.440214v2). For example, we use the stretch pose (shown above) when runing the script ```run_kinematic_replay_ground``` for avoiding collisions with the floor before the simulation begins. We add this pose as the ```pose``` variable in the simulation options, please refer to the [environment tutorial](environment_tutorial.md) to learn how to incorporate a initial pose in your simulation.

**Resting pose during simulation**

The resting pose applies for those joints that should keep a certain position during the entire simulation. If the zero pose of any segment is the desirable resting pose, then, we recommend to define that joint as *fixed* (as explaned above in the *Removing joints* subsection) as we did for our optimization experiments. However, if you want to use a non-zero pose you have to actuate the joints to keep the desired angle for each time step. We used this kind of resting pose for many non-legs segments during our *kinematic replay* experiments. These joints are defined in a dictionary named ```fixed_positions``` and defined as the ```fixed_positions``` variable in the simulation inizialization, please refer to the [environment tutorial](environment_tutorial.md) to learn how to incorporate fixed positions in your simulation.

