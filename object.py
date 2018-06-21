# Import necessary packages
import numpy as np

import cloudvolume

from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import json

from time import time
from os import listdir



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


def extract_points_chunk(chunk_range, input_dir, object_id, mip_level, info):
	
	# Extract chunk from google cloud
	range3 = chunk_range.split('_');
	ind = [];
	for i in range(3):
		range_i = range3[i];
		bound = range_i.split('-');
		ind.append(int(bound[0]));
		ind.append(int(bound[1]));
		
	print("Extracting chunk...")
	print(chunk_range)
	vol = cloudvolume.CloudVolume(input_dir, mip=mip_level, progress=True)
	min_bound = vol.bounds.minpt
	max_bound = vol.bounds.maxpt

	for i in range(3):
		if ind[2*i] < min_bound[i]:
			ind[2*i] = min_bound[i]
		if ind[2*i+1] > max_bound[i]:
			ind[2*i+1] = max_bound[i]

	chunk = vol[ind[0]:ind[1],ind[2]:ind[3],ind[4]:ind[5]]
	chunk = chunk[:,:,:,0]

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


def extract_points(input_dir, object_id, mip_level, chunk_list):
	
	t0 = time();
	points_list = []

	for i in range(len(chunk_list)):
		points = extract_points_chunk(input_dir, object_id, mip_level, chunk_list[i]);
		points_list.append(points);

	object_points = merge_points(points_list);
	t1 = time();
	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))

	return object_points


def extract_points_dist(input_dir, object_id, mip_level, chunk_list, n_core=None):
	
	t0 = time()
	vol = cloudvolume.CloudVolume(input_dir)
	with Pool(n_core) as pool:

		points_list = pool.map(partial(extract_points_chunk, input_dir=input_dir, object_id=object_id, mip_level=mip_level, info=vol.info), chunk_list)
		


	object_points = collect_points(points_list)

	t1 = time()
	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))
	
	return object_points


def save_points(points_merged, output_file):
	
	# Save npy file
	print ("Saving...")
	np.save(output_file, points_merged)
	print("Complete!")


def extract_points_object(input_dir, object_id, mip_level, json_file, output_file=''):
	
	chunk_list = list_chunks(json_file)
	print(chunk_list)

	object_points = extract_points_dist(input_dir, object_id, mip_level, chunk_list)

	if output_file != '':
		save_points(object_points, output_file)


	return object_points



# Extracting points script
from sys import argv

input_dir = argv[1]
object_id = int(argv[2])
mip_level = int(argv[3])
json_dir = argv[4]
output_file = argv[5]


if __name__ == '__main__':
	
	gs = cloudvolume.storage.Storage(json_dir)

	# Load json file with chunk list contatining object
	json_file = gs.get_file(str(object_id) + ':0')
	json_file = str(json_file)

	# Extract points
	extract_points_object(input_dir, object_id, mip_level, json_file, output_file)




