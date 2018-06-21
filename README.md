# skeletonization
Reconstructed object skeletonization

## Skeletonization pipeline
Point cloud generation (object.py) >>> Skeletonization (skeletonize.py) >>> Post-process (postprocess.py)

## Point cloud generation
### object.py
Usage:

```bash
python3 object.py /segmentation/dir/ object_id mip_level /cell_info/json_file/dir/ /output_filename

python3 object.py gs://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap/ 20763362 3 g
s://neuroglancer/pinky40_v11/watershed_mst_trimmed_sem_remap/mesh_mip_3_err_40/ ./test.npy`
```
```bash
from object import *

extract_points_object('/segmentation/dir/', object_id, mip_level, '/cell_info/json_file', '/output_filename')
```
