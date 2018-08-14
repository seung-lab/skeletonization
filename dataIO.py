import scipy.io as spio
import _pickle as pickle

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


def save_skeleton_mat(skeleton, out_filename='./skeleton.mat'):

	spio.savemat(out_filename, {'skeleton':skeleton})