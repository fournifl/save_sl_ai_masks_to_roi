import pickle as pk
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
from httpx_auth import OAuth2ResourceOwnerPasswordCredentials
from wns_api_clients import Client as ApiClient
from wns_api_clients.api.camera_blobs import read_camera_blobs
import cv2
import os
import argparse
import pandas as pd
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

def create_df(resp_api):

    filenames_A = []
    dates = []
    blob_id_A = []

    for resp in resp_api:
        dates.append(resp.date)
        filenames_A.append(resp.filename)
        blob_id_A.append(resp.id)

    df = pd.DataFrame({'date': dates, 'filename': filenames_A, 'blob_id_A': blob_id_A})
    df.set_index('date', inplace=True)
    return df


# read json parameters
training_data_pk_dirs, output_dir, camera = get_params()

# create objet ApiCLient
api_url: str = "https://app.wavesnsee.com"
api_client = ApiClient(
        base_url=api_url,
        httpx_args={
            "auth": OAuth2ResourceOwnerPasswordCredentials(
                token_url=f"{api_url}/api/auth/access-token",
                username=os.getenv('user_wns_api_client'),
                password=os.getenv('passwd_wns_api_client'),
            ),
        },
    )

# loop through cameras
for cam in camera:
    id = cam['id']
    training_data_mask_dir = Path(cam['training_data_pk_dir'].format(training_data_pk_dirs=training_data_pk_dirs))

    # list pk files
    ls = sorted(training_data_mask_dir.glob('*.pkl'))

    # get start and end dates of pk files
    start_str = datetime.strptime(ls[0].stem.split('_')[-3], '%Y%m%d').strftime('%Y-%m-%d')
    end_str_tmp = datetime.strptime(ls[-1].stem.split('_')[-3],'%Y%m%d') + timedelta(1)
    end_str = end_str_tmp.strftime('%Y-%m-%d')

    # api request read_camera_blobs
    resp_api = read_camera_blobs.sync(
        client=api_client,
        type_id=3,
        camera_id=id,
        start=start_str,
        end=end_str
    )

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


        # get date ok pk file
        date_pk_file = datetime.strptime('-'.join(pk_file.stem.split('_')[-3:]), '%Y%m%d-%H-%M').strftime('%Y-%m-%d %H:%M')
        date_pk_file = pd.to_datetime(date_pk_file)

        # construct output json filename
        df = create_df(resp_api)
        df.index = df.index.tz_localize(None)
        iloc_idx = df.index.get_indexer([date_pk_file], method='nearest')
        A_name = df['filename'][iloc_idx[0]]
        # in a future verison of library, prefer probably (check it): A_name = df['filename'].iloc[iloc_idx[0]]
        output_filename = f'training_data_masks_P_{Path(A_name).stem}.json'

        # save dict to json
        json_str = json.dumps(dico, indent=2)
        with open(output_dir.joinpath(output_filename), "w") as f:
            f.write(json_str)

