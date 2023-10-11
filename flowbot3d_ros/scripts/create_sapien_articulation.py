#!/usr/bin/env python


import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np



def create_box(
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size,
        color=None,
        name='',
) -> sapien.Actor:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba
        name: name of the actor.

    Returns:
        sapien.Actor
    """
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)  # Add collision shape
    builder.add_box_visual(half_size=half_size, color=color)  # Add visual shape
    box: sapien.Actor = builder.build(name=name)
    # Or you can set_name after building the actor
    # box.set_name(name)
    box.set_pose(pose)
    return box

def create_table(
        scene: sapien.Scene,
        pose: sapien.Pose,
        size,
        height,
        thickness=0.1,
        color=(0.8, 0.6, 0.4),
        name='table',
) -> sapien.Actor:
    """Create a table (a collection of collision and visual shapes)."""
    builder = scene.create_actor_builder()
    
    # Tabletop
    tabletop_pose = sapien.Pose([0., 0., -thickness / 2])  # Make the top surface's z equal to 0
    tabletop_half_size = [size / 2, size / 2, thickness / 2]
    builder.add_box_collision(pose=tabletop_pose, half_size=tabletop_half_size)
    builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half_size, color=color)
    
    # Table legs (x4)
    for i in [-1, 1]:
        for j in [-1, 1]:
            x = i * (size - thickness) / 2
            y = j * (size - thickness) / 2
            table_leg_pose = sapien.Pose([x, y, -height / 2])
            table_leg_half_size = [thickness / 2, thickness / 2, height / 2]
            builder.add_box_collision(pose=table_leg_pose, half_size=table_leg_half_size)
            builder.add_box_visual(pose=table_leg_pose, half_size=table_leg_half_size, color=color)

    table = builder.build(name=name)
    table.set_pose(pose)
    return table



def render_object():

    engine = sapien.Engine()  # Create a physical simulation engine
    renderer = sapien.SapienRenderer()  # Create a renderer
    engine.set_renderer(renderer)  # Bind the renderer and the engine
    scene = engine.create_scene()  # Create an instance of simulation world (aka scene)
    scene.set_timestep(1 / 100.0)  # Set the simulation frequency

    # NOTE: How to build actors (rigid bodies) is elaborated in create_actors.py
    scene.add_ground(altitude=0)  # Add a ground
    # actor_builder = scene.create_actor_builder()
    # actor_builder.add_box_collision(half_size=[0.5, 0.5, 0.5])
    # actor_builder.add_box_visual(half_size=[0.5, 0.5, 0.5], color=[1., 0., 0.])
    # box = actor_builder.build(name='box1')  # Add a box
    # box.set_pose(sapien.Pose(p=[0, 0, 0.5]))

    # Add some lights so that you can observe the scene
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    viewer = Viewer(renderer)  # Create a viewer (window)
    viewer.set_scene(scene)  # Bind the viewer and the scene

    # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
    # The principle axis of the camera is the x-axis
    viewer.set_camera_xyz(x=-4, y=0, z=2)
    # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
    # The camera now looks at the origin
    viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)

    # box = create_box(
    #     scene,
    #     sapien.Pose(p=[0, 0, 1.5]),
    #     half_size=[0.1, 0.1, 0.1],
    #     color=[0., 1., 0.],
    #     name='box2',
    # )

    builder = scene.create_actor_builder()
    builder.add_collision_from_file(filename='/home/russell/karolinska_open.stl')
    builder.add_visual_from_file(filename='/home/russell/karolinska_open.stl')
    mesh = builder.build(name='mesh')
    mesh.set_pose(sapien.Pose(p=[-0.2, 0, 1.0 + 0.05]))



    table = create_table(
        scene,
        sapien.Pose(p=[0, 0, 0.5]),
        size = 1,
        height = 0.5,
        color=[0., 0., 1.],
        name='table',
    )

    while not viewer.closed:  # Press key q to quit
        scene.step()  # Simulate the world
        scene.update_render()  # Update the world to the renderer
        viewer.render()


if __name__ == '__main__':




    render_object()
