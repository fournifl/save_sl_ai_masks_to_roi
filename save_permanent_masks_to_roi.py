import pickle as pk
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import numpy as np
import json


def get_params():

    # construct an argument parser
    parser = argparse.ArgumentParser()

    # add argument to the parser
    parser.add_argument('config')

    # get arguments
    args = vars(parser.parse_args())
    config_file = args['config']
    with open(config_file, 'r') as json_file:
        data = json_file.read()
    settings = json.loads(data)
    project_dir = settings['project_dir']
    output_dir = Path(settings['output_dir'])
    camera = settings['camera']

    return project_dir, camera, output_dir

def get_permanent_masks(pk_file):
    permanent_mask = pk.load(open(pk_file, 'rb'))
    width, height = permanent_mask.shape
    land = 255 * (permanent_mask == 4)
    land = land.astype(np.uint8)
    water = 255 * (permanent_mask == 5)
    water = water.astype(np.uint8)
    return land, water, width, height

def find_contours(arr, visu=False):
    contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if visu:
        image_contours = np.zeros((arr.shape[0], arr.shape[1], 1), np.uint8)
        cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 3)
        plt.imshow(image_contours)
        plt.show()
    return contours

def create_rois(contours_land, contours_water):

    list_roi = []

    for i in range(len(contours_land)):
        contours = np.squeeze(contours_land[i])
        roi = {}
        roi['category'] = 'other-land-features'
        roi['points'] = contours.tolist()
        list_roi.append(roi)

    for i in range(len(contours_water)):
        contours = np.squeeze(contours_water[i])
        roi = {}
        roi['category'] = 'aquatic'
        roi['points'] = contours.tolist()
        list_roi.append(roi)

    return list_roi

# read json parameters
project_dir, camera, output_dir = get_params()


# loop through cameras
for cam in camera:
    id = cam['id']
    permanent_mask_file = cam['permanent_mask_file'].format(project_dir=project_dir)

    # permanent land, water
    land, water, width, height = get_permanent_masks(permanent_mask_file)

    # find contours
    contours_land = find_contours(land)
    contours_water = find_contours(water)

    # rois from contours
    list_roi = create_rois(contours_land, contours_water)

    # save rois and img_shape to dict
    dico = {}
    dico['img_shape'] = [width, height]
    dico['rois'] = list_roi

    # save dict to json
    json_str = json.dumps(dico, indent=2)
    with open(output_dir.joinpath(f'permanent_mask_cam{id}.json'), "w") as f:
        f.write(json_str)