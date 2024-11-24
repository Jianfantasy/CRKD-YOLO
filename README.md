

### test
```bash
python val_test.py --weights 'weights/DOTA_512.pt' --data data/DOTA.yaml --batch-size 8 --img 512 --task test --save-conf --save-txt
```

### merge results
```bash
python data/DOTA_devkit_YOLO/ResultMerge.py --scratch runs/test/exp/labels/
```

### 
```bash
cd runs/test/exp/labels_merge/
zip test.zip *.txt
```

### submit test.zip to DOTA
[DOTA](http://bed4rs.net:8001/evaluation2/)
