import collections
import numpy as np
import os
import json

from home_platform.semantic import SuncgSemantics

TEST_SUNCG_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "tests", "data", "suncg")


def get_instances_from_frames(images, scene, segmentation_renderer):
    instance_color_mapping = collections.OrderedDict(segmentation_renderer.instance_color_mapping)
    all_instances = instance_color_mapping.keys()
    all_colors = np.stack(instance_color_mapping.values())
    eps = 1e-2
    instances_found = set([])

    for image in images:
        unique_colors = np.unique(image.reshape((-1, image.shape[-1])), axis=0)
        for color in unique_colors:
            z = np.sum(np.abs(all_colors - color[:3]), axis=1)
            x = np.min(z)
            if x < eps:
                item_idx = np.argmin(z)
                # if (color[:3]==np.array(MODEL_CATEGORY_COLOR_MAPPING[item])).all():
                instance_id = all_instances[item_idx]
                if instance_id not in instances_found:
                    instances_found.add(instance_id)

    return instances_found


def get_object_properties(obj):
    """
    Sets category, color and material properties of an object
    """

    properties = {}
    semanticsNp = obj.find('**/semantics')
    if not semanticsNp.isEmpty():
        properties['size'] = semanticsNp.getTag('overall-size')
        if properties['size'] == 'normal':
            properties['size'] = 'small'
        properties['shape'] = semanticsNp.getTag('coarse-category')
        properties['color'] = semanticsNp.getTag('basic-colors').split(',')[0]
        properties['material'] = semanticsNp.getTag('materials')

    return properties


def compute_all_relationships(objects):
    directions = ['right', 'left', 'front', 'behind']
    all_relationships = {x: [] for x in directions}

    for i, obj1 in enumerate(objects):
        related_objects = {x: [] for x in directions}
        for j, obj2 in enumerate(objects):
            if obj1 == obj2:
                continue
            diff = [obj1.getPos()[k] - obj2.getPos()[k] for k in [0, 1]]
            if diff[0] > 0:
                related_objects['right'].append(j)
            else:
                related_objects['left'].append(j)

            if diff[1] > 0:
                related_objects['front'].append(j)
            else:
                related_objects['behind'].append(j)

        for d in directions:
            all_relationships[d].append(related_objects[d])

    return all_relationships


def generate_metadata_from_instances(scene, instances_found):
    """
    Generates a JSON file that contains a list of objects, and directional relationships b/w each pair of objects
    :param house_id:
    :return:
    """
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
    objects_seen = []
    for idx, obj in enumerate(scene.getAllObjects()):
        instance_id = obj.getTag('instance-id')
        model_id = obj.getTag('instance-id')

        # Ignore cellings, floors and walls
        if instance_id in instances_found:
            props = get_object_properties(obj)
            if not props:
                continue
            objects_seen.append(obj)
            metadata['scenes'][0]['objects'].append({k: v for k, v in props.iteritems() if k in keys})

    metadata['scenes'][0]['relationships'] = compute_all_relationships(objects_seen)
    print metadata
    return metadata


def create_metadata_from_frames(images, scene, segmentation_renderer):
    instances_found = get_instances_from_frames(images, scene, segmentation_renderer)
    metadata = generate_metadata_from_instances(scene, instances_found)
    with open('metadata.json', 'wb') as fp:
        json.dump(metadata, fp)