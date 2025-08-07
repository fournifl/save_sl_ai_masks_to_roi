import pickle as pk
import matplotlib.pyplot as plt
import cv2
import argparse
from pathlib import Path
import numpy as np
import json


# pour visualiser les rois:
# uv tool install "roi_editor @ git+ssh://git@github.com/wavesnsee/roi_editor"
# roi_editor ./data/img.jpeg ./data/img_rois.json --read-only

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
    training_data_pk_dirs = settings['training_data_pk_dirs']
    output_dir = Path(settings['output_dir'])
    camera = settings['camera']

    return training_data_pk_dirs, output_dir, camera

def get_labelling_masks(training_data_mask):
    width, height = training_data_mask['labels'].shape
    sand = 255 * (training_data_mask['labels'] == 1)
    white_water = 255 * (training_data_mask['labels'] == 2)
    water = 255 * (training_data_mask['labels'] == 3)
    sand = sand.astype(np.uint8)
    white_water = white_water.astype(np.uint8)
    water = water.astype(np.uint8)
    return sand, white_water, water, width, height

def find_contours(arr, visu=False):
    contours, hierarchy = cv2.findContours(arr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if visu:
        image_contours = np.zeros((arr.shape[0], arr.shape[1], 1), np.uint8)
        cv2.drawContours(image_contours, contours, -1, (255, 255, 255), 3)
        plt.imshow(image_contours)
        plt.show()
    return contours

def create_rois(contours_sand, contours_whitewater, contours_water):

    list_roi = []

    for i in range(len(contours_sand)):
        contours = np.squeeze(contours_sand[i])
        if contours.shape[0] > 2:
            roi = {}
            roi['category'] = 'sand'
            roi['points'] = contours.tolist()
            list_roi.append(roi)

    for i in range(len(contours_whitewater)):
        contours = np.squeeze(contours_whitewater[i])
        if contours.shape[0] > 2:
            roi = {}
            roi['category'] = 'white-water'
            roi['points'] = contours.tolist()
            list_roi.append(roi)

    for i in range(len(contours_water)):
        contours = np.squeeze(contours_water[i])
        if contours.shape[0] > 2:
            roi = {}
            roi['category'] = 'water'
            roi['points'] = contours.tolist()
            list_roi.append(roi)

    return list_roi

# read json parameters
training_data_pk_dirs, output_dir, camera = get_params()

# loop through cameras
for cam in camera:
    id = cam['id']
    training_data_mask_dir = Path(cam['training_data_pk_dir'].format(training_data_pk_dirs=training_data_pk_dirs))

    # list pk files
    ls = training_data_mask_dir.glob('*.pkl')

    # loop through training_data files
    for pk_file in ls:
        training_data_mask = pk.load(open(pk_file, 'rb'))

        # labelling masks of sand, white-water, water
        sand, whitewater, water, width, height = get_labelling_masks(training_data_mask)

        # find contours
        contours_sand = find_contours(sand)
        contours_whitewater = find_contours(whitewater)
        contours_water = find_contours(water)

        # rois from contours
        list_roi = create_rois(contours_sand, contours_whitewater, contours_water)

        # save rois and img_shape to dict
        dico = {}
        dico['img_shape'] = [width, height]
        dico['rois'] = list_roi

        # save dict to json
        json_str = json.dumps(dico, indent=2)
        output_filename = f'training_data_masks_P_A_CAM{id}_2{pk_file.stem.split('_2')[1]}.json'
        with open(output_dir.joinpath(output_filename), "w") as f:
            f.write(json_str)