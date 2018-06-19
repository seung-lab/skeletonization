# Import necessary packages
from skeletonization import *
from postprocess import *


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
