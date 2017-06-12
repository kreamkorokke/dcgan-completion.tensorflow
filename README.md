## DCGAN Image Completion Algorithm
Our implementaion is a modificaiton based on Brandon Amos's "Image Completion with Deep Learning in Tensorflow"

To train:
```
$ ./train_dcgan.py --dataset /path/to/preprocessed/dataset --epoch <num epoch>
```

To perform image completion on images:
```
$ ./complete_images.py /path/to/preprocessed/dataset/waiting/to/be/masked/and/completed --output-dir <output dir> --num-iters <num iters>
```
Run the above command with  ```--log-l1-loss``` to log L1 loss in log.txt.
