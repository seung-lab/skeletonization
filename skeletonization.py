## skeletonization.py
# Skeletonization function using TEASAR algorithm to build the skeleton of the cell

# Import necessary packages
import numpy as np

import scipy.io as spio
from scipy import ndimage
from scipy.sparse.csgraph import *
from scipy.sparse import csr_matrix

import networkx as nx

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


# def extract_points_dist(input_dir, object_id, mip_level, chunk_list):
# 	t0 = time()
# 	client = Client()

# 	n = len(chunk_list)
# 	extract = client.map(extract_points_chunk, [input_dir]*n, [object_id]*n, [mip_level]*n, chunk_list)

# 	points_list = client.gather(extract)

# 	object_points = collect_points(points_list)

# 	t1 = time()
# 	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))
# 	return object_points


def save_points(points_merged, output_file):
	# Save mat file
	print ("Saving...")
	spio.savemat(output_file,{'p':points_merged});
	print("Complete!")



##### SKELETONIZATION #####
# Skeleton format
class Skeleton:

	def __init__(self, nodes=np.array([]), edges=np.array([]), radii=np.array([]), root=np.array([])):

		self.nodes = nodes
		self.edges = edges
		self.radii = radii
		self.root = root

# Convert array to point cloud
def array2point(array, object_id=-1):
	## INPUT ##
	# array : array with labels
	# object_id : object label to extract point cloud

	## OUTPUT ##
	# points : n x 3 point coordinates 

	if object_id == -1:
		object_coord = np.where(array > 0)
	else:
		object_coord = np.where(array == object_id)

	object_x = object_coord[0]
	object_y = object_coord[1]
	object_z = object_coord[2]

	points = np.zeros([len(object_x),3], dtype='uint32')
	points[:,0] = object_x
	points[:,1] = object_y
	points[:,2] = object_z

	return points


# Downsample points
def downsample_points(points, dsmp_resolution):
	## INPUT ##
	# points : n x 3 point coordinates
	# dsmp_resolution : [x, y, z] downsample resolution

	## OUTPUT ##
	# point_downsample : n x 3 downsampled point coordinates

	if len(points.shape) == 1:
			points = np.reshape(points,(1,3))

	dsmp_resolution = np.array(dsmp_resolution, dtype='float')

	point_downsample = np.zeros(points.shape)
	point_downsample = points/dsmp_resolution

	point_downsample = np.round(point_downsample)
	point_downsample = np.unique(point_downsample, axis=0)
	point_downsample = point_downsample.astype('uint16')

	return point_downsample


# Upsample points
def upsample_points(points, dsmp_resolution):
	## INPUT ##
	# points : n x 3 point coordinates
	# dsmp_resolution : [x, y, z] downsample resolution

	## OUTPUT ##
	# point_upsample : n x 3 upsampled point coordinates

	dsmp_resolution = np.array(dsmp_resolution)

	point_upsample = np.zeros(points.shape)
	point_upsample = points*dsmp_resolution
	
	point_upsample = point_upsample.astype('uint16')

	return point_upsample


# Find corresponding row 
def find_row(array, row):
	## INPUT ##
	# array : array to search for
	# row : row to find

	## OUTPUT ##
	# idx : row indices

	row = np.array(row)

	if array.shape[1] != row.size:
		print("Dimension do not match!")
	
	else:
		NDIM = array.shape[1]

		valid = np.zeros(array.shape, dtype='bool')

		for i in range(NDIM):
			valid[:,i] = array[:,i] == row[i]

		row_loc = np.zeros([array.shape[0],1])

		if NDIM == 2:
			row_loc = valid[:,0]*valid[:,1]

		elif NDIM == 3:
			row_loc = valid[:,0]*valid[:,1]*valid[:,2]

		idx = np.where(row_loc==1)[0]

		if len(idx) == 0:
			idx = -1

		return idx


class Nodes:

	def __init__(self, coord, max_bound):
		
		n = coord.shape[0]
		coord = coord.astype('uint32')
		self.coord = dict(zip(map(tuple,coord),np.arange(coord.shape[0])))
		self.max_bound = max_bound.astype('uint32')

		idx = np.zeros(n, dtype='uint64')
		idx = coord[:,0] + max_bound[0]*coord[:,1] + max_bound[0]*max_bound[1]*coord[:,2]
		self.idx = idx

		idx2node = np.ones(np.prod(max_bound))*-1
		idx2node[idx] = np.arange(coord.shape[0], dtype='int64')
		self.node = idx2node


	def sub2idx(self, sub_array):

		if len(sub_array.shape) == 1:
			sub_array = np.reshape(sub_array,(1,3))

		sub_array = sub_array.astype('uint32')

		max_bound = self.max_bound
		idx = np.zeros(sub_array.shape[0])
		idx = sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]

		return idx


	def sub2node(self, sub_array):
		
		if len(sub_array.shape) == 1:
			sub_array = np.reshape(sub_array,(1,3))

		sub_array = sub_array.astype('uint32')

		max_bound = self.max_bound

		idx_array = sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]


		node = self.node[idx_array]
		node = node.astype('int64')

		return node

 
def find_path(predecessor, end, start = []):
	## INPUT ##
	# predecessor : n x n array of predecessors of shortest path from i to j
	# end : destination node
	# start : start node (Not necessary if the predecessor array is 1D array)

	## OUTPUT ##
	# path : n x 1 array consisting nodes in path

	path_list = [end]

	pred = end
	while True:
		if len(predecessor.shape) > 1:
			pred = predecessor[start,pred]

			if pred == -9999:
				break
			else:
				path_list.append(pred)

		else:
			pred = predecessor[pred]

			if pred == -9999:
				break
			else:
				path_list.append(pred)


	path_list.reverse()
	path = np.array(path_list)

	return path


def thr_linear(x, linear_parameters, threshold):
	## INPUT ##
	# x : function input
	# parameters : [slope, constant]
	# threshold : threshold of cutoff

	## OUTPUT ##
	# y : function output

	slope = linear_parameters[0]
	const = linear_parameters[1]

	y = x * slope + const

	if y >= threshold:
		y = threshold

	return y


def path2edge(path):
	## INPUT ##
	# path : sequence of nodes

	## OUTPUT ##
	# edges : sequence separated into edges

	edges = np.zeros([len(path)-1,2], dtype='uint32')
	for i in range(len(path)-1):
		edges[i,0] = path[i]
		edges[i,1] = path[i+1]

	return edges


def reorder_nodes(nodes,edges):
	## INPUT ##
	# nodes : list of node numbers
	# edges : list of edges

	## OUTPUT ##
	# edges_reorder : edges with reordered node numbers

	edges_reorder = np.zeros(edges.shape)
	for i in range(edges.shape[0]):
		edges_reorder[i,0] = np.where(nodes==edges[i,0])[0]
		edges_reorder[i,1] = np.where(nodes==edges[i,1])[0]

	return edges_reorder


def TEASAR(object_points, parameters, init_root=np.array([]), init_dest=np.array([]), soma=0):
	## INPUT ##
	# object_points : n x 3 point cloud format of the object
	# parameters : list of "scale" and "constant" parameters (first: "scale", second: "constant")
	#              larger values mean less senstive to detecting small branches

	## OUTPUT ##
	# skeleton : skeleton object

	NDIM = object_points.shape[1]
	object_points = object_points.astype('uint32')

	max_bound = np.max(object_points, axis=0) + 2

	object_nodes = Nodes(object_points, max_bound)

	bin_im = np.zeros(max_bound, dtype='bool')
	bin_im[object_points[:,0],object_points[:,1],object_points[:,2]] = True


	n = object_points.shape[0]
	print('Number of points ::::: ' + str(n))


	# Distance to the boundary map
	print("Creating DBF...")
	print bin_im.shape
	DBF = ndimage.distance_transform_edt(bin_im)


	# Penalty weight for the edges
	M = np.max(DBF)**1.01
	p_v = 100000*(1-DBF/M)**16


	# 26-connectivity
	nhood_26 = np.zeros([3,3,3], dtype='bool')
	nhood_26 = np.where(nhood_26==0)

	nhood = np.zeros([nhood_26[0].size,3])
	for i in range(NDIM):
		nhood[:,i] = nhood_26[i]
	nhood = nhood - 1
	nhood = np.delete(nhood,find_row(nhood,[0,0,0]),axis=0)

	n_nhood = nhood.shape[0]	
	nhood_weight = np.sum(nhood**2,axis=1)**16

	nhood_points = np.zeros([n,3])
	nhood_nodes = np.ones([n,n_nhood])*-1 
	obj_node = np.zeros([n,n_nhood])
	edge_dist = np.zeros([n,n_nhood])
	edge_weight = np.zeros([n,n_nhood])
	
	print("Setting edge weight...")
	for i in range(n_nhood):

		obj_node[:,i] = np.arange(n)

		nhood_points = object_points + nhood[i,:]

		valid = np.all(nhood_points>=0, axis=1)*np.all(nhood_points<max_bound, axis=1)

		nhood_nodes[valid,i] = object_nodes.sub2node(nhood_points[valid,:])

		valid = nhood_nodes[:,i] != -1
		edge_dist[valid,i] = nhood_weight[i]

		valid_idx = np.where(valid)[0]

		if soma:
			edge_weight[valid,i] = nhood_weight[i] * p_v[object_points[valid_idx,0],object_points[valid_idx,1],object_points[valid_idx,2]]
		else:
			edge_weight[valid,i] = p_v[object_points[valid_idx,0],object_points[valid_idx,1],object_points[valid_idx,2]]
		

	print("Creating graph...")
	valid_edge = np.where(nhood_nodes != -1)
	G_dist = csr_matrix((edge_dist[valid_edge[0],valid_edge[1]],(obj_node[valid_edge[0],valid_edge[1]],nhood_nodes[valid_edge[0],valid_edge[1]])), shape=(n,n))
	G = csr_matrix((edge_weight[valid_edge[0],valid_edge[1]],(obj_node[valid_edge[0],valid_edge[1]],nhood_nodes[valid_edge[0],valid_edge[1]])), shape=(n,n))


	root_nodes = object_nodes.sub2node(init_root)
	n_root = root_nodes.shape[0]
	
	is_disconnected = np.ones(n, dtype='bool')
	
	r = 0
	c = 0 
	nodes = np.array([])
	edges = np.array([])

	# When destination nodes are not set.
	if init_dest.shape[0] == 0:
		while np.any(is_disconnected):
			print("Processing connected component...")
			# Calculate distance map from the root node
			if r <= n_root - 1: 
				root = root_nodes[r]
				
				D_Gdist = dijkstra(G_dist, directed=True, indices=root)

				cnt_comp = ~np.isinf(D_Gdist)
				is_disconnected = is_disconnected * ~cnt_comp

				cnt_comp = np.where(cnt_comp)[0]
				cnt_comp_im = np.zeros(max_bound, dtype='bool')
				cnt_comp_im[object_points[cnt_comp,0],object_points[cnt_comp,1],object_points[cnt_comp,2]] = 1

			# Set separate root node for broken pieces
			else:
				root = np.where(is_disconnected==1)[0][0]

				D_Gdist = dijkstra(G_dist, directed=True, indices=root)

				cnt_comp = ~np.isinf(D_Gdist)
				is_disconnected = is_disconnected * ~cnt_comp

				cnt_comp = np.where(cnt_comp)[0]
				cnt_comp_im = np.zeros(max_bound, dtype='bool')
				cnt_comp_im[object_points[cnt_comp,0],object_points[cnt_comp,1],object_points[cnt_comp,2]] = 1

				
			# Graph shortest path in the weighted graph
			D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)


			# Build skeleton and remove pieces that are completed.
			# Iterate until entire connected component is completed.

			if cnt_comp.shape[0] < 5000:
				r = r + 1 
				continue

			path_list = [];
			while np.any(cnt_comp):
				print("Finding path...")
				dest_node = cnt_comp[np.where(D_Gdist[cnt_comp]==np.max(D_Gdist[cnt_comp]))[0][0]]

				path = find_path(pred_G, dest_node)
				path_list.append(path)

				for i in range(len(path)):
					path_node = path[i]
					path_point = object_points[path_node,:]

					d = thr_linear(DBF[path_point[0],path_point[1],path_point[2]], parameters, 500)
					
					cube_min = np.zeros(3, dtype='uint32')
					cube_min = path_point - d
					cube_min[cube_min<0] = 0
					cube_min = cube_min.astype('uint32')
					
					cube_max = np.zeros(3, dtype='uint32')
					cube_max = path_point + d
					cube_max[cube_max>max_bound] = max_bound[cube_max>max_bound]
					cube_max = cube_max.astype('uint32')

					cnt_comp_im[cube_min[0]:cube_max[0],cube_min[1]:cube_max[1],cube_min[2]:cube_max[2]] = 0

				cnt_comp_sub = array2point(cnt_comp_im)
				cnt_comp = object_nodes.sub2node(cnt_comp_sub)


			for i in range(len(path_list)):
				path = path_list[i]

				if c + i == 0:
					nodes = path
					edges = path2edge(path)

				else:
					nodes = np.concatenate((nodes,path))
					edges_path = path2edge(path)
					edges = np.concatenate((edges,edges_path))


			r = r + 1 
			c = c + 1

	# When destination nodes are set. 
	else:
		dest_nodes = object_nodes.sub2node(init_dest)

		path_list = []
		for r in range(root_nodes.shape[0]):
			root = root_nodes[r]

			D_G, pred_G = dijkstra(G, directed=True, indices=root, return_predecessors=True)

			for i in range(dest_nodes.shape[0]):
				dest = dest_nodes[i]

				if np.isfinite(D_G[dest]):
					path = find_path(pred_G, dest)
					path_list.append(path)

		for i in range(len(path_list)):
			path = path_list[i]

			if soma:
				path = np.delete(path,np.arange(1,int(path.shape[0]*0.8)))

			if i == 0:
				nodes = path
				edges = path2edge(path)
				
			else:
				nodes = np.concatenate((nodes,path))
				edges_path = path2edge(path)
				edges = np.concatenate((edges,edges_path))
					

	if nodes.shape[0] == 0 or edges.shape[0] == 0:
		skeleton = Skeleton()

	else:
		# Consolidate nodes and edges
		nodes = np.unique(nodes)
		edges = np.unique(edges, axis=0)

		skel_nodes = object_points[nodes,:]
		skel_edges = reorder_nodes(nodes,edges)
		skel_edges = skel_edges.astype('uint32')
		skel_radii = DBF[skel_nodes[:,0],skel_nodes[:,1],skel_nodes[:,2]]

		skeleton = Skeleton(skel_nodes, skel_edges, skel_radii)

		skeleton = consolidate_skeleton(skeleton)


	return skeleton	


# Skeletonization
def skeletonize(object_input, object_id = 1, dsmp_resolution = [1,1,1], parameters = [6,6], init_root = [], init_dest = [], soma = 0):
	## INPUT ##
	# object_input : object to skeletonize (N x 3 point cloud or 3D labeled array)
	# object_id : object ID to skeletonize (Don't need this if object_input is in point cloud format)
	# dsmp_resolution : downsample resolution
	# parameters : list of "scale" and "constant" parameters (first: "scale", second: "constant")
	#              larger values mean less senstive to detecting small branches
	# init_roots ; N x 3 array of initial root coordinates

	## OUTPUT ##
	# skeleton : skeleton object

	# Don't run skeletonization if the input is an empty array
	if object_input.shape[0] == 0:
		skeleton =  Skeleton()

	else:
		# Convert object_input to point cloud format
		if object_input.shape[1] == 3: # object input: point cloud
			obj_points = object_input
		else:                          # object input: 3D array
			obj_points = array2point(object_input) 

		# If initial roots is empty, take the first point
		init_root = np.array(init_root)
		init_dest = np.array(init_dest)

		if len(init_root) == 0:
			init_root = obj_points[0,:]

		else:
			for i in range(init_root.shape[0]):
				root = init_root[i,:]
				root_idx = find_row(obj_points, root)

				if root_idx == -1:
					dist = np.sum((obj_points - root)**2,1)
					root = obj_points[np.argmin(dist),:]
					init_root[i,:] = root 


		for i in range(init_dest.shape[0]):
			dest = init_dest[i,:]
			dest_idx = find_row(obj_points, dest)

			if dest_idx == -1:
				dist = np.sum((obj_points - dest)**2,1)
				dest = obj_points[np.argmin(dist),:]
				init_dest[i,:] = dest 

	
		# Downsample points
		if sum(dsmp_resolution) > 3:
			print(">>>>> Downsample...")
			obj_points = downsample_points(obj_points, dsmp_resolution)
			init_root = downsample_points(init_root, dsmp_resolution)

			if init_dest.shape[0] != 0:
				init_dest = downsample_points(init_dest, dsmp_resolution)

		# Convert coordinates to bounding box
		min_bound = np.min(obj_points, axis=0)
		obj_points = obj_points - min_bound + 1
		init_root = init_root - min_bound + 1

		if init_dest.shape[0] != 0:
			init_dest = init_dest - min_bound + 1


		# Skeletonize chunk surrounding object
		print(">>>>> Building skeleton...")
		t0 = time()
		skeleton = TEASAR(obj_points, parameters, init_root, init_dest, soma)
		t1 = time()
		print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))

		# Convert coordinates back into original coordinates
		if skeleton.nodes.shape[0] != 0:
			skeleton.nodes = upsample_points(skeleton.nodes + min_bound - 1, dsmp_resolution)


	return skeleton


##### Post-processing #####
from scipy import spatial
from scipy.sparse import *
import scipy.sparse.csgraph as csgraph


def get_valid(points, bound):

	valid = (points[:,0]>bound[0,0]) * (points[:,0]<bound[1,0]) * (points[:,1]>bound[0,1]) * (points[:,1]<bound[1,1]) * (points[:,2]>bound[0,2]) * (points[:,2]<bound[1,2])

	return valid


def edges2sparse(nodes, edges):

	s = nodes.shape[0]
	conn_mat = lil_matrix((s, s), dtype=bool)
	conn_mat[edges[:,0],edges[:,1]] = 1
	conn_mat[edges[:,1],edges[:,0]] = 1

	return conn_mat


def find_connected(nodes, edges):

	s = nodes.shape[0] 

	nodes = np.unique(edges)
	nodes = nodes.astype('int')

	conn_mat = lil_matrix((s, s), dtype=bool)
	conn_mat[edges[:,0],edges[:,1]] = 1

	n, l = csgraph.connected_components(conn_mat, directed=False)
	
	l_nodes = l[nodes]
	l_list = np.unique(l_nodes)

	path_list = []
	for i in l_list:
		path = l==i
		
		path_list.append(path)

	return path_list


def in_bound(points, bound):

	min_bound = bound[0,:]

	valid = (points - min_bound) >= 0

	if len(valid.shape) == 1: 
		valid = np.reshape(valid,[1,3])
	
	inbound = valid[:,0] * valid[:,1] * valid[:,2]

	return inbound 


def remove_edges(edges, predecessor, start_idx, end_idx):
	
	edges = np.sort(edges)

	path = [end_idx]
	current_idx = end_idx
	
	while current_idx != start_idx:

		pred_idx = predecessor[start_idx, current_idx]
		
		current_edge = np.sort(np.array([pred_idx, current_idx]))
		current_edge_idx = np.where((edges[:,0]==current_edge[0])*(edges[:,1]==current_edge[1]))[0]
		edges = np.delete(edges,current_edge_idx,0)

		current_idx = pred_idx
		path.append(pred_idx)

	path = np.array(path)

	return edges, path


def remove_overlap_edges(nodes_overlap, edges):
	
	edge_overlap = np.isin(edges, nodes_overlap)

	del_idx = np.where(edge_overlap[:,0]*edge_overlap[:,1])

	edges = np.delete(edges,del_idx,0)

	return edges


def merge_skeletons(skeleton1, skeleton2):
	
	nodes1 = skeleton1.nodes
	nodes2 = skeleton2.nodes

	edges1 = skeleton1.edges
	edges2 = skeleton2.edges


	if True:
	# if edges2.shape[0] >= edges1.shape[0]:
		tree1 = spatial.cKDTree(nodes1)

		nodes2 = nodes2.astype('float32')
		(dist, nodes1_idx) = tree1.query(nodes2)

		graph2 = edges2sparse(nodes2, edges2)
		
		overlap = dist == 0
		nodes2_overlap = np.where(overlap)[0]
	
		edges2 = remove_overlap_edges(nodes2_overlap, edges2) 

		
###
# Method 1
		# connected = find_connected(nodes2, edges2)

		# conn_mat1 = edges2sparse(nodes1, edges1)
		# conn_mat2 = edges2sparse(nodes2, edges2)
		# dist_mat1, pred1 = shortest_path(conn_mat1, directed=False, return_predecessors=True)
		# dist_mat2, pred2 = shortest_path(conn_mat2, directed=False, return_predecessors=True)


		# for i in range(len(connected)):
		# 	path = connected[i]
		# 	path_idx = np.where(path)[0]

			
		# 	nodes_path = nodes2[path,:]
		# 	(dist, nodes1_idx) = tree1.query(nodes_path)

		# 	overlap = dist == 0
		# 	nodes2_overlap = np.where(overlap)[0]

		# 	if np.sum(overlap) < 2:
		# 		continue

		# 	nodes1_overlap = nodes1_idx[overlap]

		# 	pairs = combination_pairs(nodes1_overlap.shape[0])
		# 	for j in range(pairs.shape[0]):
		# 		start_node1 = nodes1_overlap[pairs[j,0]]
		# 		end_node1 = nodes1_overlap[pairs[j,1]]

		# 		if np.isfinite(dist_mat1[start_node1, end_node1]):
		# 			start_node2 = path_idx[nodes2_overlap[pairs[j,0]]]
		# 			end_node2 = path_idx[nodes2_overlap[pairs[j,1]]]

		# 			if np.isfinite(dist_mat2[start_node2, end_node2]):	
		# 				path_piece = find_path(pred2, end_node2, start_node2)

		# 				edges2 = remove_overlap_edges(path_piece, edges2)


# Method 2
		connected = find_connected(nodes2, edges2)

		conn_mat1 = edges2sparse(nodes1, edges1)
		conn_mat2 = edges2sparse(nodes2, edges2)
		dist_mat1, pred1 = shortest_path(conn_mat1, directed=False, return_predecessors=True)

		for i in range(len(connected)):
			path = connected[i]
			path_idx = np.where(path)[0]

			conn_mat_path = conn_mat2[path,:][:,path]
			
			end_nodes2 = np.where(np.sum(conn_mat_path, 0)==1)[1]
			
			end_idx = path_idx[end_nodes2]

			end_points = nodes2[end_idx,:]
			
			end_nodes1 = np.zeros(end_points.shape[0], dtype='int')
			for j in range(end_points.shape[0]):
				p_end = end_points[j,:]

				node_end = find_row(nodes1, p_end)

				end_nodes1[j] = node_end

			if np.sum(end_nodes1<0) > 0:
				continue

			c = 1
			pairs = combination_pairs(end_nodes1.shape[0])
			for j in range(pairs.shape[0]):
				c = c * np.isfinite(dist_mat1[end_nodes1[pairs[j,0]],end_nodes1[pairs[j,1]]])

			if c==1:
				edges2 = remove_overlap_edges(path_idx, edges2)
##

		skeleton2.edges = edges2
		skeleton2 = consolidate_skeleton(skeleton2)

	# else:
	# 	tree2 = spatial.cKDTree(nodes2)

	# 	nodes1 = nodes1.astype('float32')
	# 	(dist, nodes2_idx) = tree2.query(nodes1)

	# 	graph1 = edges2sparse(nodes1, edges1)
		
	# 	overlap = dist == 0
	# 	nodes1_overlap = np.where(overlap)[0]
	
	# 	edges1 = remove_overlap_edges(nodes1_overlap, edges1) 

	# 	skeleton1.edges = edges1
	# 	skeleton1 = remove_dust(skeleton1)

	# skeleton2 = remove_dust(skeleton2)


	return skeleton1, skeleton2


def remove_dust(skeleton, dust_threshold):

	nodes = skeleton.nodes
	edges = skeleton.edges 

	connected = find_connected(nodes, edges)

	for i in range(len(connected)):
		path = connected[i]

		if np.sum(path) < dust_threshold:
			path_nodes = np.where(path)[0]

			for j in range(len(path_nodes)):
				del_row_idx, del_col_idx = np.where(edges==path_nodes[j])
				edges = np.delete(edges, del_row_idx, 0)

	skeleton.edges = edges
	skeleton = consolidate_skeleton(skeleton)


	return skeleton


def merge_cell(skeletons, points_list):
	
	adjacent_chunks = points_list[1].astype('uint32')
	
	for i in range(adjacent_chunks.shape[0]):

		print(adjacent_chunks[i,:])
		skeleton1 = skeletons[adjacent_chunks[i,0]]
		skeleton2 = skeletons[adjacent_chunks[i,1]]

		if skeleton1.nodes.shape[0] == 0 or skeleton2.nodes.shape[0] == 0:
			continue

		skeleton1 = consolidate_skeleton(skeleton1)
		skeleton2 = consolidate_skeleton(skeleton2)
		skeleton1, skeleton2 = merge_skeletons(skeleton1, skeleton2)

		skeletons[adjacent_chunks[i,0]] = skeleton1
		skeletons[adjacent_chunks[i,1]] = skeleton2
		
		print("%f merged." %(float(i)/adjacent_chunks.shape[0]))

	offset = 0
	for i in range(len(points_list)-2):
		
		skeleton = skeletons[i]

		if skeleton.edges.shape[0] == 0:																																																																																																																																																																														
			continue

		skeleton.edges = skeleton.edges.astype('uint32')
		skeleton.edges = skeleton.edges + offset
		
		if offset == 0:
			nodes = skeleton.nodes
			edges = skeleton.edges
			radii = skeleton.radii

		else:
			nodes = np.concatenate((nodes,skeleton.nodes), axis=0)
			edges = np.concatenate((edges,skeleton.edges), axis=0)
			radii = np.concatenate((radii,skeleton.radii), axis=0)

		offset = offset + skeleton.nodes.shape[0]

	skeleton_merged = Skeleton(nodes,edges,radii)
	skeleton_merged = consolidate_skeleton(skeleton_merged)


	return skeleton_merged


def remove_row(array, rows2remove):

	for i in range(rows2remove.shape[0]):
		idx = find_row(array,rows2remove[i,:])

		array = np.delete(array, idx, axis=0)


	return array


def interpolate_line(point1, point2):

	n_step = 10

	NDIM = point1.size
	int_points = np.zeros((n_step,NDIM))
	
	point1 = point1.astype('float32')
	point2 = point2.astype('float32')
	for i in range(n_step):
		a = i+1
		b = n_step-i
		int_points[i,:] = (a*point2+b*point1)/(a+b)

	int_points = np.round(int_points)
	int_points = np.unique(int_points, axis=0)

	return int_points


def connect_pieces(skeleton, p):
	# Connect broken pieces
	nodes = skeleton.nodes
	edges = skeleton.edges

	all_connected = 1
	while all_connected == 1:
		connected = find_connected(nodes, edges)

		n_connected = len(connected)

		pairs = combination_pairs(n_connected)

		all_connected = 0
		for i in range(pairs.shape[0]):
			path_piece = connected[pairs[i,0]]
			nodes_piece = nodes[path_piece]
			nodes_piece = nodes_piece.astype('float32')
			nodes_piece_idx = np.where(path_piece)[0]

			path_tree = connected[pairs[i,1]]
			nodes_tree = nodes[path_tree]
			nodes_tree_idx = np.where(path_tree)[0]
			tree = spatial.cKDTree(nodes_tree)

			(dist, idx) = tree.query(nodes_piece)

			min_dist = np.min(dist)

			if min_dist < 50:
		
				min_dist_idx = int(np.where(dist==min_dist)[0][0])
				start_idx = nodes_piece_idx[min_dist_idx]
				end_idx = nodes_tree_idx[idx[min_dist_idx]]

				int_points = interpolate_line(nodes[start_idx,:],nodes[end_idx,:])
	
				for k in range(int_points.shape[0]):
					in_seg = np.sum(~np.any(p - int_points[k,:], axis=1))

					if in_seg == 0:
						break

				if in_seg:
					new_edge = np.array([start_idx,end_idx])
					
					new_edge = np.reshape(new_edge,[1,2])
					edges = np.concatenate((edges,new_edge),0)
					print('Connected.')

					all_connected += 1

					break


	skeleton.edges = edges

	skeleton = consolidate_skeleton(skeleton)

	return skeleton


def remove_ticks(skeleton, threshold=150):
	# Remove ticks
	edges = skeleton.edges

	path_all = np.ones(1)
	while path_all.shape[0] != 0:
		
		unique_nodes, unique_counts = np.unique(edges, return_counts=True)

		end_idx = np.where(unique_counts==1)[0]

		path_all = np.array([])
		for i in range(end_idx.shape[0]):
			idx = end_idx[i]
			current_node = unique_nodes[idx]

			edge_row_idx, edge_col_idx = np.where(edges==current_node)

			path = np.array([])
			single_piece = 0
			while edge_row_idx.shape[0] == 1 and path.shape[0] < threshold:
				
				next_node = edges[edge_row_idx,1-edge_col_idx]
				path = np.concatenate((path,edge_row_idx))

				prev_row_idx = edge_row_idx
				prev_col_idx = 1-edge_col_idx
				current_node = next_node
				
				edge_row_idx, edge_col_idx = np.where(edges==current_node)

				if edge_row_idx.shape[0] == 1:
					single_piece = 1
					break

				next_row_idx = np.setdiff1d(edge_row_idx,prev_row_idx)
				next_col_idx = edge_col_idx[np.where(edge_row_idx==next_row_idx[0])[0]]

				edge_row_idx = next_row_idx 
				edge_col_idx = next_col_idx

			# print path.shape
			if path.shape[0] < threshold and single_piece == 0:
				path_all = np.concatenate((path_all,path))
				print("Tick removed.")
			
		edges = np.delete(edges,path_all,axis=0)

	skeleton.edges = edges

	skeleton = consolidate_skeleton(skeleton)
	
	return skeleton


def remove_loops(skeleton):
	# Remove loops
	nodes = skeleton.nodes
	edges = skeleton.edges
	edges = np.sort(edges, axis=1)

	
	cycle_exists = 1

	while cycle_exists == 1:
		G = nx.Graph()

		for i in range(edges.shape[0]):
			G.add_edge(edges[i,0], edges[i,1])

		try: 
			edges_cycle = nx.find_cycle(G, orientation='ignore')
			print('Loop removed.')

		except:
			cycle_exists = 0
			print('All loops removed.')
			continue

		edges_cycle = np.array(edges_cycle)
		edges_cycle = np.sort(edges_cycle, axis=1)

		nodes_cycle = np.unique(edges_cycle)
		nodes_cycle = nodes_cycle.astype('int')
		
		unique_nodes, unique_counts = np.unique(edges, return_counts=True)

		branch_nodes = unique_nodes[unique_counts>=3]

		branch_cycle = nodes_cycle[np.isin(nodes_cycle,branch_nodes)]
		branch_cycle = branch_cycle.astype('int')

		if branch_cycle.shape[0] == 1:
			branch_cycle_point = nodes[branch_cycle,:]

			cycle_points = nodes[nodes_cycle,:]

			dist = np.sum((cycle_points - branch_cycle_point)**2, 1)
			end_node = nodes_cycle[np.argmax(dist)]

			edges = remove_row(edges, edges_cycle)

			new_edge = np.array([branch_cycle[0],end_node])
			new_edge = np.reshape(new_edge,[1,2])
			edges = np.concatenate((edges,new_edge), 0)

		elif branch_cycle.shape[0] == 2:
			path = nx.shortest_path(G,branch_cycle[0],branch_cycle[1])

			edge_path = path2edge(path)
			edge_path = np.sort(edge_path, axis=1)

			row_valid = np.ones(edges_cycle.shape[0])
			for i in range(edge_path.shape[0]):
				row_valid = row_valid - (edges_cycle[:,0]==edge_path[i,0])*(edges_cycle[:,1]==edge_path[i,1])

			row_valid = row_valid.astype('bool')
			edge_path = edges_cycle[row_valid,:]

			edges = remove_row(edges, edge_path)

		elif branch_cycle.shape[0] == 0:
			edges = remove_row(edges, edges_cycle)

		else:
			branch_cycle_points = nodes[branch_cycle,:]

			centroid = np.mean(branch_cycle_points, axis=0)
			dist = np.sum((nodes - centroid)**2, 1)
			intersect_node = np.argmin(dist)
			intersect_point = nodes[intersect_node,:]

			edges = remove_row(edges, edges_cycle)

			new_edges = np.zeros((branch_cycle.shape[0],2))
			new_edges[:,0] = branch_cycle
			new_edges[:,1] = intersect_node

			if np.isin(intersect_node, branch_cycle):
				idx = np.where(branch_cycle==intersect_node)
				new_edges = np.delete(new_edges, idx, 0)

			edges = np.concatenate((edges,new_edges), 0)


	skeleton.nodes = nodes
	skeleton.edges = edges
	skeleton = consolidate_skeleton(skeleton)

	return skeleton


def trim_skeleton(skeleton, p):

	skeleton = remove_dust(skeleton, 100)

	skeleton = remove_loops(skeleton)

	skeleton = connect_pieces(skeleton, p)

	skeleton = remove_loops(skeleton)

	skeleton = remove_ticks(skeleton)

	return skeleton


def consolidate_skeleton(skeleton):

	nodes = skeleton.nodes 
	edges = skeleton.edges
	radii = skeleton.radii

	if nodes.shape[0] == 0 or edges.shape[0] == 0:
		skeleton = Skeleton()

	else:
		# Remove duplicate nodes
		unique_nodes, unique_idx, unique_counts = np.unique(nodes, axis=0, return_index=True, return_counts=True)
		unique_edges = np.copy(edges)

		dup_idx = np.where(unique_counts>1)[0]
		for i in range(dup_idx.shape[0]):
			dup_node = unique_nodes[dup_idx[i],:]
			dup_node_idx = find_row(nodes, dup_node)

			for j in range(dup_node_idx.shape[0]-1):
				start_idx, end_idx = np.where(edges==dup_node_idx[j+1])
				unique_edges[start_idx, end_idx] = unique_idx[dup_idx[i]]


		# Remove unnecessary nodes
		eff_node_list = np.unique(unique_edges)
		eff_node_list = eff_node_list.astype('int')
		
		eff_nodes = nodes[eff_node_list]
		eff_radii = radii[eff_node_list]

		eff_edges = np.copy(unique_edges)
		for i, node in enumerate(eff_node_list, 0):
			row_idx, col_idx = np.where(unique_edges==node)

			eff_edges[row_idx,col_idx] = i

		eff_edges = np.unique(eff_edges, axis=0)

		skeleton.nodes = eff_nodes
		skeleton.edges = eff_edges
		skeleton.radii = eff_radii


	return skeleton


def combination_pairs(n):
	pairs = np.array([])

	for i in range(n):
		for j in range(n-i-1):
			pairs = np.concatenate((pairs,np.array([i,i+j+1])))

	pairs = np.reshape(pairs,[pairs.shape[0]/2,2])
	pairs = pairs.astype('uint8')

	return pairs


def chunk_points(p, chunk_size=512, overlap=100):

	min_p = np.min(p, 0)
	max_p = np.max(p, 0)

	min_bound = np.floor(min_p/256)*256
	max_bound = np.ceil((max_p-min_bound)/chunk_size)*chunk_size + min_bound

	n_chunk = (max_bound - min_bound)/chunk_size
	n_chunk = n_chunk.astype('uint16')

	p_list = []
	filled = np.array([])
	bound_list = np.zeros([n_chunk[0]*n_chunk[1]*n_chunk[2],2,3])
	c = 0
	for i in range(n_chunk[0]):
		for j in range(n_chunk[1]):
			for k in range(n_chunk[2]):
				chunk_range = np.array([[i*chunk_size+min_bound[0],j*chunk_size+min_bound[1],k*chunk_size+min_bound[2]],
					[(i+1)*chunk_size+min_bound[0],(j+1)*chunk_size+min_bound[1],(k+1)*chunk_size+min_bound[2]]])

				chunk_range[0,:] = chunk_range[0,:] - overlap
				chunk_range[1,:] = chunk_range[1,:] + overlap

				bound_list[c,:,:] = chunk_range

				valid = get_valid(p,chunk_range)

				if np.sum(valid) < 50:
					p_list.append(np.array([]))
					c = c + 1
					continue
					
				p_valid = p[valid,:]

				p_list.append(p_valid)
				filled = np.concatenate((filled,np.array([i,j,k])))

				c = c + 1
	
	filled = np.reshape(filled,[filled.shape[0]/3,3])
	n_filled = filled.shape[0]

	chunk_pairs = combination_pairs(n_filled)

	adjacent_chunks = np.array([])
	for i in range(chunk_pairs.shape[0]):
		chunk1_coord = filled[chunk_pairs[i,0],:]
		chunk2_coord = filled[chunk_pairs[i,1],:]	
		loc_diff = np.abs(chunk1_coord - chunk2_coord)

		if np.max(loc_diff) == 1:
			chunk1_idx = chunk1_coord[2] + n_chunk[2]*chunk1_coord[1] + n_chunk[2]*n_chunk[1]*chunk1_coord[0]
			chunk2_idx = chunk2_coord[2] + n_chunk[2]*chunk2_coord[1] + n_chunk[2]*n_chunk[1]*chunk2_coord[0]
		
			adjacent_chunks = np.concatenate((adjacent_chunks,np.array([chunk1_idx,chunk2_idx])))

	adjacent_chunks = np.reshape(adjacent_chunks,[adjacent_chunks.shape[0]/2,2])

	p_list.insert(0, adjacent_chunks)
	p_list.insert(0, bound_list)


	return p_list


def crop_skeleton(skeleton, bound):

	nodes = skeleton.nodes
	edges = skeleton.edges
	radii = skeleton.radii

	nodes_valid_mask = get_valid(nodes, bound)
	nodes_valid_idx = np.where(nodes_valid_mask)[0]

	edges_valid_mask = np.isin(edges, nodes_valid_idx)
	edges_valid_idx = edges_valid_mask[:,0]*edges_valid_mask[:,1]
	edges_valid = edges[edges_valid_idx,:]

	skeleton.edges = edges_valid

	skeleton = consolidate_skeleton(skeleton)


	return skeleton


def skeletonize_cell(points_list, parameters):

	bound_list = points_list[0]

	skeletons = []
	for i in range(len(points_list)-2):

		p = points_list[i+2]
		
		if p.shape[0] != 0:
			skeleton = skeletonize(p,1,[1,1,1],parameters)
			
			if skeleton.nodes.shape[0] != 0:
				bound = np.zeros([2,3])
				bound[0,:] = bound_list[i,0,:] + 50
				bound[1,:] = bound_list[i,1,:] - 50
				skeleton = crop_skeleton(skeleton, bound)

				skeletons.append(skeleton)

			else:
				skeletons.append(skeleton)

		else:
			skeleton = Skeleton()
			skeletons.append(skeleton)


	return skeletons


def skeletonize_cell_dist(points_list, parameters, n_core=None):
	t0 = time()
	pool = Pool(n_core)
	
	# Skeletonize 
	points_chunk = points_list[2:]

	try:
		skeletons = pool.map(partial(skeletonize, object_id=1, dsmp_resolution=[1,1,1], parameters=parameters), points_chunk)

		pool.close()
		pool.join()
		pool.terminate()

	except:
		pool.terminate()
		
	t1 = time()
	print(">>>>> Elapsed time : " + str(np.round(t1-t0, decimals=3)))
	

	return skeletons


def crop_cell(skeletons, points_list):
	# Crop skeletons
	bound_list = points_list[0]
	print(bound_list.shape)
	print(len(skeletons))
	for i in range(len(skeletons)):
		skeleton = skeletons[i]

		if skeleton.nodes.shape[0] != 0:
			bound = np.zeros([2,3])
			bound[0,:] = bound_list[i,0,:] + 50
			bound[1,:] = bound_list[i,1,:] - 50
			
			skeleton = crop_skeleton(skeleton, bound)

			skeletons[i] = skeleton


	return skeletons


def remove_soma(p, soma_coord, threshold=-1):

	if threshold == -1:
		dist = np.sum((p - soma_coord)**2,1)**0.5

		hist, bin_edges = np.histogram(dist, np.arange(1,np.max(dist),50))

		hist_diff = np.diff(hist)
		hist_diff = hist_diff.astype('float32')
		diff_ratio = np.abs(hist_diff/hist[:-1])

		thr_idx = np.where(diff_ratio<0.15)[0][0]
		threshold = np.sum(bin_edges[thr_idx:thr_idx+2])/2
		print(threshold)

		# hist_ratio = hist/float(np.max(hist))
		# thr_idx = np.where(hist_ratio<0.01)[0][0]
		# threshold = np.sum(bin_edges[thr_idx:thr_idx+2])/2
		# print threshold

	p_nosoma = p[dist>threshold,:]


	return p_nosoma


def connect_soma(skeleton, soma_coord, p):

	nodes = skeleton.nodes
	edges = skeleton.edges
	radii = skeleton.radii

	connected = find_connected(nodes, edges)
	print len(connected)

	neighbors = np.array([])
	for i in range(len(connected)):
		piece_mask = connected[i]

		nodes_piece = nodes[piece_mask,:]

		dist = np.sum((nodes_piece - soma_coord)**2,1)**0.5
		
		if min(dist) > 700:
			continue

		piece_idx = np.where(piece_mask)[0]

		node_neighbor = piece_idx[np.argmin(dist)]

		edge_row_idx_init, edge_col_idx_init = np.where(edges==node_neighbor)
		
		next_node = np.copy(node_neighbor)

		n = 15
		del_idx_list = []
		neighbor_candidates = np.zeros(edge_row_idx_init.shape[0])
		for j in range(edge_row_idx_init.shape[0]):		
		
			edge_row_idx = edge_row_idx_init[j]
			edge_col_idx = edge_col_idx_init[j]
			edge_row_idx = np.array([edge_row_idx])
			edge_col_idx = np.array([edge_col_idx])

			c = 0
			del_idx = [] 
			while edge_row_idx.shape[0] != 0 and c != n:

				next_node = edges[edge_row_idx[0], 1-edge_col_idx[0]]
				del_idx.append(edge_row_idx[0])
				neighbor_candidates[j] = next_node

				prev_idx = edge_row_idx
				next_row_idx, next_col_idx = np.where(edges==next_node)
				edge_row_idx = np.setdiff1d(next_row_idx, prev_idx)
				if edge_row_idx.shape[0] != 0:
					edge_col_idx = next_col_idx[np.where(next_row_idx==edge_row_idx[0])[0]]

				c = c + 1

			del_idx_list.append(del_idx)

		
		l_del_idx = np.zeros(edge_row_idx_init.shape[0])
		del_idx = []
		for j in range(edge_row_idx_init.shape[0]):
			l_del_idx[j] = len(del_idx_list[j])

			del_idx = del_idx + del_idx_list[j]

		if np.sum(l_del_idx==n) < 2:
			node_neighbor = neighbor_candidates[np.argmax(l_del_idx)]

			edges = np.delete(edges, del_idx, 0)
		
		node_neighbor = int(node_neighbor)
		p_neighbor = nodes[node_neighbor,:]
		neighbors = np.concatenate((neighbors, p_neighbor))

	neighbors = np.reshape(neighbors,[neighbors.shape[0]/3,3])
	print(neighbors)

	soma_coord = np.reshape(soma_coord,[1,3])
	point_group = np.concatenate((neighbors, soma_coord))
	bound = np.zeros([2,3])
	bound[0,:] = np.min(point_group, 0) - [20,20,20]
	bound[1,:] = np.max(point_group, 0) + [20,20,20]

	valid = get_valid(p, bound)
	p_valid = p[valid,:]

	# Skeletonize soma
	dsmp_resolution = 2

	for i in range(neighbors.shape[0]):
		coord = neighbors[i,:]

		coord_approx = np.round(coord/dsmp_resolution)*dsmp_resolution

		node_idx = find_row(nodes, coord)
		nodes[node_idx,:] = coord_approx
		neighbors[i,:] = coord_approx

	# Parameters are meaningless
	soma_skeleton = skeletonize(p_valid, 1, [dsmp_resolution]*3, [10,10], soma_coord, neighbors, 1)

	# Consolidate soma skeleton and the rest
	soma_skeleton.edges = soma_skeleton.edges + nodes.shape[0]

	nodes = np.concatenate((nodes, soma_skeleton.nodes))
	edges = np.concatenate((edges, soma_skeleton.edges))
	radii = np.concatenate((radii, soma_skeleton.radii))
	root = upsample_points(downsample_points(soma_coord, [dsmp_resolution]*3), [dsmp_resolution]*3) 

	skeleton = Skeleton(nodes, edges, radii, root)

	skeleton = consolidate_skeleton(skeleton)


	return skeleton


def smooth_skeleton(skeleton, ratio=2):									

	nodes = skeleton.nodes
	edges = skeleton.edges
	radii = skeleton.radii
	
	root_node = np.setdiff1d(edges[:,0], edges[:,1])[0]
	next_node = root_node

	edges_smooth = np.array([])
	while edges.shape[0] >= ratio:	

		for i in range(ratio):

			edge_row_idx, edge_col_idx = np.where(edges==next_node)

			if edge_row_idx.shape[0] > 1 and i != 0:
				dest_node = next_node
				break

			elif edge_row_idx.shape[0] == 0:
				dest_node = next_node
				break

			else:
				next_node = edges[edge_row_idx[0],1-edge_col_idx[0]]
				dest_node = next_node
			
				edges = np.delete(edges,edge_row_idx[0],0)				

		if root_node == dest_node:

			if np.unique(edges[:,0]).shape[0] >= np.unique(edges[:,1]).shape[0]: 
				next_node = np.setdiff1d(edges[:,0], edges[:,1])[0]
			else:
				next_node = np.setdiff1d(edges[:,1], edges[:,0])[0]

		else:

			new_edge = np.array([root_node,dest_node])
			edges_smooth = np.concatenate((edges_smooth,new_edge),0)

			edge_row_idx, edge_col_idx = np.where(edges==dest_node)
			if  edge_row_idx.shape[0] != 0:
				root_node = dest_node
				next_node = root_node
				
			else:
				np.save('edges.npy',edges)
				if np.unique(edges[:,0]).shape[0] >= np.unique(edges[:,1]).shape[0]: 
					root_node = np.setdiff1d(edges[:,0], edges[:,1])[0]
				else:
					root_node = np.setdiff1d(edges[:,1], edges[:,0])[0]

				next_node = root_node


	edges_smooth = np.reshape(edges_smooth,[edges_smooth.shape[0]/2,2])
	edges_smooth = np.concatenate((edges_smooth,edges),0)

	skeleton.nodes = nodes
	skeleton.edges = edges_smooth.astype('uint16')
	skeleton.radii = radii


	return skeleton


def skeletonize_file(points_file, output_file, n_core=2, soma=0, soma_coord=np.array([])):

	
	print('Loading points...')
	p = np.load(points_file)

	if soma:
		p_wsoma = np.copy(p)

		print('Removing soma...')
		p = remove_soma(p_wsoma,soma_coord)

	points_list = chunk_points(p,512,256)

	skeletons = skeletonize_cell_dist(points_list, [10,10], n_core)
	skeletons = crop_cell(skeletons, points_list)

	print('Merging chunks...')
	skeleton = merge_cell(skeletons, points_list)
	skeleton = trim_skeleton(skeleton, p)


	if soma:
		skeleton = connect_soma(skeleton, soma_coord, p)


	save_skeleton(skeleton, output_file)



##### Skeletons in neuroglancer #####
# from __future__ import absolute_import

import collections
import io
import six
import struct

class SkeletonPrecomputed(object):

    def __init__(self, vertex_positions, edges, vertex_attributes=None):
        self.vertex_positions = np.array(vertex_positions, dtype='<f4')
        self.edges = np.array(edges, dtype='<u4')
        self.vertex_attributes = vertex_attributes

    def encode(self, source=None):
        result = io.BytesIO()
        edges = self.edges
        vertex_positions = self.vertex_positions
        vertex_attributes = self.vertex_attributes
        result.write(struct.pack('<II', vertex_positions.shape[0], edges.shape[0] // 2))
        result.write(vertex_positions.tobytes())
        if source and len(source.vertex_attributes) > 0:
            for name, info in six.iteritems(source.vertex_attributes):

                attribute = np.array(vertex_attributes[name],
                                     np.dtype(info.data_type).newbyteorder('<'))
                expected_shape = (vertex_positions.shape[0], info.num_components)
                if (attribute.shape[0] != expected_shape[0] or
                        attribute.size != np.prod(expected_shape)):
                    raise ValueError('Expected attribute %r to have shape %r, but was: %r' %
                                     (name, expected_shape, attribute.shape))
                result.write(attribute.tobytes())
        result.write(edges.tobytes())

        return result.getvalue()


def make_precomputed_skeleton(filename, output_filename='./skeleton'):
    if filename[-3:] == 'mat':
        mat = spio.loadmat(filename)
        skeleton = mat['skeleton']
        n = skeleton['nodes'][0][0]
        e = skeleton['edges'][0][0]

    else:
        skeleton = load_skeleton(filename)
        n = skeleton['nodes']
        e = skeleton['edges']

    print n.shape, e.shape
    n = n.astype('uint32')
    n[:,:2] = n[:,:2]*32
    n[:,2] = n[:,2]*40

    e = np.reshape(e,[1, e.size])
    e = e[0]
    
    pyskel = Skeleton(n,e).encode()

    f = open(output_filename,'w')
    f.write(pyskel)
    f.close()


##### Data I/O #####
import cPickle as pickle

def save_skeleton(skeleton, filename='./skeleton.pkl'):
	
	SkeletonDict = {}
	SkeletonDict["nodes"] = skeleton.nodes.astype('uint16')
	SkeletonDict["edges"] = skeleton.edges.astype('uint32')
	SkeletonDict["radii"] = skeleton.radii.astype('float32')
	SkeletonDict["root"] = skeleton.root.astype('uint16')

	with open(filename, 'wb') as output:
		pickle.dump(SkeletonDict, output, pickle.HIGHEST_PROTOCOL)


def load_skeleton(filename):

	with open(filename, 'rb') as input:
		skeleton = pickle.load(input)

	return skeleton


##### Python/Matlab #####
def load_points_mat(mat_file):

	mat = spio.loadmat(mat_file)
	p = mat['p']

	return p


def save_skeleton_mat(skeleton,out_filename='./skeleton.mat'):

	spio.savemat(out_filename, {'skeleton':skeleton})
