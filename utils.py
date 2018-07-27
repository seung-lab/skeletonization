# Import necessary packages
import numpy as np


# Skeleton 
class Skeleton:

	def __init__(self, nodes=np.array([]), edges=np.array([]), radii=np.array([]), root=np.array([])):

		if nodes.size == 0:
			nodes = nodes.reshape((0,3))

		if edges.size == 0:
			edges = edges.reshape((0,3))

		self.nodes = nodes
		self.edges = edges
		self.radii = radii
		self.root = root

	def empty(self):

		return self.nodes.size == 0 or self.edges.size == 0


# Nodes
class Nodes:

	def __init__(self, coord, max_bound):
		
		n = coord.shape[0]

		coord = coord.astype(np.uint32)
		self.max_bound = max_bound.astype(np.uint32)

		idx = coord[:,0] + max_bound[0]*coord[:,1] + max_bound[0]*max_bound[1]*coord[:,2]
		
		idx2node = np.ones(np.prod(max_bound), dtype=np.int32)*-1
		idx2node[idx] = np.arange(coord.shape[0], dtype=np.int32)
		self.node = idx2node

	def sub2idx(self, sub_array):

		if len(sub_array.shape) == 1:
			sub_array = np.reshape(sub_array,(1,3))

		sub_array = sub_array.astype(np.int32)

		max_bound = self.max_bound
		
		return sub_array[:,0] + max_bound[0]*sub_array[:,1] + max_bound[0]*max_bound[1]*sub_array[:,2]

	def sub2node(self, sub_array):
		
		idx_array = self.sub2idx(sub_array)

		return self.node[idx_array].astype(np.int32)


# Convert array to point cloud
def array2point(array, object_id=-1):
	"""
	[INPUT]
	array : array with labels
	object_id : object label to extract point cloud

	[OUTPUT]
	points : n x 3 point coordinates 
	"""

	if object_id is None:
		object_coord = np.where(array > 0)
	else:
		object_coord = np.where(array == object_id)

	object_x = object_coord[0]
	object_y = object_coord[1]
	object_z = object_coord[2]

	points = np.zeros([len(object_x),3], dtype=np.int32)
	points[:,0] = object_x
	points[:,1] = object_y
	points[:,2] = object_z

	return points


# Downsample points
def downsample_points(points, dsmp_resolution):
	"""
	[INPUT]
	points : n x 3 point coordinates
	dsmp_resolution : [x, y, z] downsample resolution

	[OUTPUT]
	point_downsample : n x 3 downsampled point coordinates
	"""

	if len(points.shape) == 1:
			points = np.reshape(points,(1,3))

	dsmp_resolution = np.array(dsmp_resolution, dtype=np.float)

	point_downsample = points/dsmp_resolution

	point_downsample = np.round(point_downsample)
	point_downsample = np.unique(point_downsample, axis=0)

	return point_downsample.astype(np.int32)


# Upsample points
def upsample_points(points, dsmp_resolution):
	"""
	[INPUT]
	points : n x 3 point coordinates
	dsmp_resolution : [x, y, z] downsampled resolution

	[OUTPUT]
	point_upsample : n x 3 upsampled point coordinates
	"""

	dsmp_resolution = np.array(dsmp_resolution)
	
	point_upsample = points*dsmp_resolution
	
	return point_upsample.astype(np.int32)

	
# Find corresponding row 
def find_row(array, row):
	"""
	[INPUT]
	array : array to search for
	row : row to find

	[OUTPUT]
	idx : row indices
	"""

	row = np.array(row)

	if array.shape[1] != row.size:
		raise ValueError("Dimension do not match!")
	
	NDIM = array.shape[1]

	row_loc = np.ones((array.shape[0],), dtype='bool')
	
	for i in range(NDIM):
		valid = array[:,i] == row[i]
		row_loc = row_loc * valid

	idx = np.where(row_loc==1)[0]

	if len(idx) == 0:
		idx = -1

	return idx


# Find path from predecessor matrix
def find_path(predecessor, end, start = []):
	"""
	[INPUT]
	predecessor : n x n array of predecessors of shortest path from i to j
	end : destination node
	start : start node (Not necessary if the predecessor array is 1D array)

	[OUTPUT]
	path : n x 1 array consisting nodes in path
	"""

	path_list = [end]
	pred = end

	while True:
		if len(predecessor.shape) > 1:
			pred = predecessor[start,pred]

		else:
			pred = predecessor[pred]

		if pred == -9999:
			break
		else:
			path_list.append(pred)

	path_list.reverse()

	return np.array(path_list)


# Thresholded linear function (saturated linear)
def thr_linear(x, linear_parameters, threshold):
	"""
	[INPUT]
	x : function input
	parameters : [slope, constant] (y = slope*x + constant)
	threshold : threshold or cutoff

	[OUTPUT]
	y : function output
	"""

	slope, const = linear_parameters
	
	return min(x * slope + const, threshold)


# Reorder nodes so there is no unused node ids
def reorder_nodes(nodes, edges):
	"""
	[INPUT]
	nodes : list of node numbers
	edges : list of edges

	[OUTPUT]
	edges_reorder : edges with reordered node numbers
	"""

	edges_reorder = np.zeros(edges.shape)
	for i in range(edges.shape[0]):
		edges_reorder[i,0] = np.where(nodes==edges[i,0])[0]
		edges_reorder[i,1] = np.where(nodes==edges[i,1])[0]

	return edges_reorder


# Convert n x 1 path array to list of edges
def path2edge(path):
	"""
	[INPUT]
	path : sequence of nodes (n x 1 array)

	[OUTPUT]
	edges : list of edges that form path (n x 2 array)
	"""

	edges = np.zeros([len(path)-1,2], dtype=np.uint32)
	for i in range(len(path)-1):
		edges[i,0] = path[i]
		edges[i,1] = path[i+1]

	return edges
