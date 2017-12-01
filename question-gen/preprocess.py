import json
import os

from home_platform.env import BasicEnvironment
from home_platform.semantic import MaterialColorTable, MaterialTable, SuncgSemantics

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests/data")
TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests/data", "suncg")


def compute_all_relationships(scene):
    """
    Computes relationships between all pairs of objects in the scene.
    """

    objects = scene.getAllObjects()
    directions = ['east', 'west', 'north', 'south']
    all_relationships = {x: [] for x in directions}

    for i, obj1 in enumerate(objects):
        related_objects = {x: [] for x in directions}
        for j, obj2 in enumerate(objects):
            if obj1 == obj2:
                continue
            diff = [obj1.getPos()[k] - obj2.getPos()[k] for k in [0, 1]]
            if diff[0] > 0:
                related_objects['east'].append(j)
            else:
                related_objects['west'].append(j)

            if diff[1] > 0:
                related_objects['north'].append(j)
            else:
                related_objects['south'].append(j)

        for d in directions:
            all_relationships[d].append(related_objects[d])

    return all_relationships


def get_object_properties(obj, semantic_world):
    """
    Sets category, color and material properties of an object
    """

    properties = {}

    semanticsNp = obj.find('**/semantics')
    if not semanticsNp.isEmpty():
        properties['size'] = semanticsNp.getTag('overall-size')
        properties['shape'] = semanticsNp.getTag('coarse-category')
        properties['color'] = semanticsNp.getTag('basic-colors')
        properties['material'] = semanticsNp.getTag('materials')

    return properties


def generate_scene_metadata(house_id):
    """
    Generates a JSON file that contains a list of objects, and directional relationships b/w each pair of objects
    :param house_id:
    :return:
    """
    env = BasicEnvironment(houseId=house_id, suncgDatasetRoot=TEST_SUNCG_DATA_DIR)
    scene = env.scene
    keys = ['shape', 'size', 'color', 'material']

    metadata = {
        'info': {'split': 'train'},
        'scenes': [
            {
                "split": "train",
                "image_index": 0,
                "image_filename": "SUNCG_env_000000.png",
                "objects": [],
                "relationships": {}
            }
        ]
    }


    semantic_world = SuncgSemantics(scene, TEST_SUNCG_DATA_DIR)
    for idx, obj in enumerate(scene.getAllObjects()):
        props = get_object_properties(obj, semantic_world)
        metadata['scenes'][0]['objects'].append({k: v for k, v in props.iteritems() if k in keys})

    metadata['scenes'][0]['relationships'] = compute_all_relationships(env.scene)

    return metadata

if __name__ == '__main__':
    house_id_list = ['0004d52d1aeeb8ae6de39d6bd993e992']
    for house_id in house_id_list:
        metadata = generate_scene_metadata(house_id)
        with open(house_id + '.metadata.json', 'wb') as fp:
            json.dump(metadata, fp)
