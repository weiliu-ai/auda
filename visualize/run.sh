# night
python xmuda/test_visualize_night.py --cfg=configs/nuscenes/day_night/xmuda_test.yaml @/model_2d_075000.pth @/model_3d_090000.pth
python xmuda/test_visualize_night.py --cfg=configs/nuscenes/day_night/auda_best.yaml @/model_2d_145000.pth @/model_3d_145000.pth
 
# SG
python xmuda/test_visualize_SG.py --cfg=configs/nuscenes/usa_singapore/auda_test.yaml @/model_2d_070000.pth @/model_3d_100000.pth
python xmuda/test_visualize_SG.py --cfg=configs/nuscenes/usa_singapore/xmuda_test.yaml @/model_2d_065000.pth @/model_3d_095000.pth

#KITTI
python xmuda/test_visualize_semantic_kitti.py --cfg=configs/a2d2_semantic_kitti/auda_test.yaml  @/model_2d_110000.pth @/model_3d_095000.pth
python xmuda/test_visualize_semantic_kitti.py --cfg=configs/a2d2_semantic_kitti/xmuda_test.yaml  @/model_2d_060000.pth @/model_3d_095000.pth
