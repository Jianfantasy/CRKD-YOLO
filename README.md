# 进行测试
python val_test.py --weights 'weights/DOTA_512.pt' --data data/DOTA.yaml --batch-size 8 --img 512 --task test --save-conf --save-txt

# 合并结果
python data/DOTA_devkit_YOLO/ResultMerge.py --scratch runs/test/exp/labels/

# 压缩合并后的结果
cd runs/test/exp/labels_merge/
zip test.zip *.txt

# 提交 test.zip 到 DOTA
# 提交地址: http://bed4rs.net:8001/evaluation2/
