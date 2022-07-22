CN, MCI, AD - Groups 
ADNI_demos.csv - subjects
NM_icns_info.csv - Components


1. 4D image -> adni_aa__sub01_component_ica_s1_.nii
2. timeCourse Data -> adni_aa__postprocess_results.mat
3. NM_icns_info.csv -> Neuro marks indexs for single brain.


ADNI_demos.csv (CN, MCI, AD)

reading subjects w.r.t groups -> ADNI_demos.csv 
  |
  |
4D - image (read) // adni_aa__sub01_component_ica_s1_.nii or spacialMap
  |  array type (float) of size 100
  |
100 3D images or Components
  |
  |  (53 images -> NM_icns_info.csv)
  |
Each Image
  |
  |
  | ------ > Voxel Count - size of the node or brain nodal size

Structure:
    VoxelCount : {
        CN: {
            paths: [sub1, sub2, sub3, sub4, ....................],
            indexes:
              69:    [vo1, vo2, vo3, vo4, vo5,.....................]
              53:    [vo1, vo2, vo3, vo4, vo5,.....................]
              98:    [vo1, vo2, vo3, vo4, vo5,.....................]
              99:    [vo1, vo2, vo3, vo4, vo5,.....................]
              45:
              ...
        },
        MCI: {
            paths:  [sub1, sub2, sub3, sub4, ....................],
            indexes:
              69:     [vo1, vo2, vo3, vo4, vo5,.....................],
              53:     [vo1, vo2, vo3, vo4, vo5,.....................],
              98:     [vo1, vo2, vo3, vo4, vo5,.....................],
              99:     [vo1, vo2, vo3, vo4, vo5,.....................],
              45:
              ...
        },
        AD: {
            paths:  [sub1, sub2, sub3, sub4, ....................],
            indexes:
              69:     [vo1, vo2, vo3, vo4, vo5,.....................],
              53:     [vo1, vo2, vo3, vo4, vo5,.....................],
              98:     [vo1, vo2, vo3, vo4, vo5,.....................],
              99:     [vo1, vo2, vo3, vo4, vo5,.....................],
              45:
              ...
        },
    }








