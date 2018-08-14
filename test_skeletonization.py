from skeletonization import *
from dataIO import *
from sys import argv


input_file = argv[1]
output_file = argv[2]


p = np.load(input_file)

skeleton = skeletonize(p, 1, [1,1,1], [10,10])

# np.save('./test/skeleton.npy', skeleton)
save_skeleton_mat(skeleton, output_file)
