# Some example programs

## Check model segmentation time on one picture
```
python3 check_time.py --model_path ../segmentation/models/edanetlr\=0.001optim\=Adamaxepoch\=38.pth.tar 
```

## Check model segmentation time on directory with pictures

```
python3 check_time_loop.py --model_path ../segmentation/models/edanetlr\=0.001optim\=Adamaxepoch\=38.pth.tar --test_dir ./test-pictures
```
