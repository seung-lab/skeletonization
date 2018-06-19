# Import necessary packages
import numpy as np
import scipy.io as spio

from cloudvolume import CloudVolume

from multiprocessing import Pool
from functools import partial

import json

from time import time
from os import listdir


##### LOAD OBJECT #####
def list_chunks(json_file):
	if json_file[-4:] == 'json':
		file = open(json_file);
		data = json.load(file);
		fragments = data["fragments"];

		chunk_list = [];
		for i in range(len(fragments)):
			chunk = fragments[i];
			chunk_range = chunk.split(":")[2];
			chunk_list.append(chunk_range);

	else:
		fragments = json_file.split("[")[1][:-2]
		fragment_list = fragments.split(",")

		chunk_list = [];
		for i in range(len(fragment_list)):
			chunk = fragment_list[i].strip()[1:-1]
			chunk_range = chunk.split(":")[2]
			chunk_list.append(chunk_range)

	return chunk_list


def extract_points_chunk(input_dir, object_id, mip_level, chunk_range, overlap=0):
	# Extract chunk from google cloud
	range3 = chunk_range.split('_');
	ind = [];
	for i in range(3):
		range_i = range3[i];
		bound = range_i.split('-');
		ind.append(int(bound[0]) - overlap);
		ind.append(int(bound[1]) + overlap);

	print("Extracting chunk...")
	vol = CloudVolume(input_dir,mip=mip_level);
	min_bound = vol.bounds.minpt
	max_bound = vol.bounds.maxpt

	for i in range(3):
		if ind[2*i] < min_bound[i]:
			ind[2*i] = min_bound[i]
		if ind[2*i+1] > max_bound[i]:
			ind[2*i+1] = max_bound[i]

	chunk = vol[ind[0]:ind[1],ind[2]:ind[3],ind[4]:ind[5]];
	chunk = chunk[:,:,:,0];

	print("Extracting point cloud...")
	object_loc = np.where(chunk==object_id);

	points = np.zeros([object_loc[0].size,3]);
	for i in range(3):
		points[:,i] = object_loc[i] + ind[2*i];

	return points


def collect_points(points_list):
	for i in range(len(points_list)):
		if i == 0:
			points_merged = points_list[i];

		else:
			points_merged = np.concatenate((points_merged,points_list[i]),axis=0);

	return points_merged


def extract_points_object(input_dir, object_id, mip_level, chunk_list):
	t0 = time();
	points_list = []

	for i in range(len(chunk_list)):
		points = extract_points_chunk(input_dir, object_id, mip_level, chunk_list[i]);
		points_list.append(points);

	object_points = merge_points(points_list);
	t1 = time();
	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))

	return object_points


def extract_points_dist(input_dir, object_id, mip_level, chunk_list):
	t0 = time()
	pool = Pool(n_core)

	n = len(chunk_list)
	points_list = pool.map(partial(extract_points_chunk, input_dir=input_dir, object_id=object_id, mip_level=mip_level), chunk_list)

	pool.close()
	pool.join()
	pool.terminate()

	object_points = collect_points(points_list)

	t1 = time()
	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))
	
	return object_points


def save_points(points_merged, output_file):
	# Save mat file
	print ("Saving...")
	spio.savemat(output_file,{'p':points_merged});
	print("Complete!")
