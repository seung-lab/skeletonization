# Import necessary packages
import numpy as np
import scipy.io as spio
from scipy import ndimage
from scipy.sparse.csgraph import *
from scipy.sparse import csr_matrix

import networkx as nx

from multiprocessing import Pool
from functools import partial

from utils import *

from time import time
from os import listdir

 

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

			# if soma:
			# 	path = np.delete(path,np.arange(1,int(path.shape[0]*0.4)))

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
	# init_roots : N x 3 array of initial root coordinates

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
