cd ~/object_skeleton
source skeleton-venv3_6/bin/activate
#
# OIL_SEPARATOR_CAST_IRON
#python lightweight-human-pose-estimation.pytorch/demo_iros.py --checkpoint-path Model_oil_castIron/checkpoint_iter_12500.pth --images data/rgb_01.png
#
# AIR_PIPE 
#python demo.py --checkpoint-path Model_airPipe/checkpoint_iter_13000.pth --images data/rgb_01.png
#
# OIL_SEPARATOR_PLASTIC
python demo.py --checkpoint-path Model_oil_plastic/checkpoint_iter_30000.pth --images data/rgb_01.png
