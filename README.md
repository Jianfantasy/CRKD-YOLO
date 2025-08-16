

### Test
```bash
python val_test.py --weights 'weights/DOTA_512.pt' --data data/DOTA.yaml --batch-size 8 --img 512 --task test --save-conf --save-txt
```

### Merge results
```bash
python data/DOTA_devkit_YOLO/ResultMerge.py --scratch runs/test/exp/labels/
```


```bash
cd runs/test/exp/labels_merge/
zip test.zip *.txt
```

### Submit test.zip to DOTA
[DOTA](http://bed4rs.net:8001/evaluation2/)



### Train

```bash
python train_CRKD.py --weights "" --data data/DOTA.yaml --cfg_low path/to/cfg_low.yaml --cfg_high path/to/cfg_high.ytaml --batch-size 8 --device 0
```









<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src=".\figures\target-size-vary.png"
         alt="False"
         style="zoom:80%"/>
    <br>		<!--换行-->
    Comparison of performance on varying size targets between our CRKD-YOLO and baseline YOLOv5s.	<!--标题-->
    </center>
</div>

<div>			<!--块级封装-->
    <center>	<!--将图片和文字居中-->
    <img src=".\figures\Visualization of the detection results in high- and low-resolution images.png"
         alt="False"
         style="zoom:80%"/>
    <br>		<!--换行-->
    Visualization of the detection results in high- and low-resolution images.png	<!--标题-->
    </center>
</div>

