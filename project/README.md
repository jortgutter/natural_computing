# README

To run the code, simply use main.py.
The CIFAR-10 dataset is automatically downloaded through Keras.
```
usage: main.py [-h] [--n_decision ND] [--activation A] [--n_nets N] [--ensemble_method M] [--drop_classes DC] [--val_split VS] [--test_split TS] [--dropout] [--p_dropout PD] [--early_stopping] [--silent] [--seed SE] [--use_adam]
               model epochs n_conv start_channels output_filename

Train a network (ensemble) on CIFAR10 data and predict

positional arguments:
  model                Either 'base' or 'ensemble'
  epochs               Number of epochs for training
  n_conv               Number of convolution blocks in the network
  start_channels       Number of output channels of first convolution block (after that gets doubled every block)
  output_filename      Name of the output file to write the training logs and results to

optional arguments:
  -h, --help           show this help message and exit
  --n_decision ND      Number of dense decision layers of the network
  --activation A       Activation function for convolution blocks
  --n_nets N           Number of networks in ensemble
  --ensemble_method M  Method of distributing data to ensemble
  --drop_classes DC    Number of classes to drop for ensemble data distribution (dropout)
  --val_split VS       Fraction of data used for validation
  --test_split TS      Fraction of data used for testing
  --dropout            Activate dropout during training
  --p_dropout PD       Dropout probability
  --early_stopping     Use early stopping during training
  --silent             Deactivate verbosity
  --seed SE            Seed used for randomness
  --use_adam           Optimizer to use. Default: SGD

```

The outputs will automatically be generated in the `out` folder. Many experiment outputs can already be found here.
The output for one of `python3 main.py base 10 6 32 base_10ep_6conv_32chan.txt` looks like this:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 32, 32, 32)        896       
                                                                 
 conv2d_1 (Conv2D)           (None, 32, 32, 64)        18496     
                                                                 
 conv2d_2 (Conv2D)           (None, 32, 32, 128)       73856     
                                                                 
 max_pooling2d (MaxPooling2D  (None, 16, 16, 128)      0         
 )                                                               
                                                                 
 conv2d_3 (Conv2D)           (None, 16, 16, 256)       295168    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 8, 8, 256)        0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 8, 8, 512)         1180160   
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 4, 4, 512)        0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 4, 4, 1024)        4719616   
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 2, 2, 1024)       0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 4096)              0         
                                                                 
 dense (Dense)               (None, 10)                40970     
                                                                 
=================================================================
Total params: 6,329,162
Trainable params: 6,329,162
Non-trainable params: 0
_________________________________________________________________
loss: [1.7741189002990723, 1.2054717540740967, 0.8766117095947266, 0.6530761122703552, 0.4702526330947876, 0.31345242261886597, 0.19863328337669373, 0.14023198187351227, 0.09786204248666763, 0.07911155372858047]
accuracy: [0.35280001163482666, 0.5736444592475891, 0.694088876247406, 0.7743777632713318, 0.8371999859809875, 0.8894888758659363, 0.9305333495140076, 0.9511111378669739, 0.9660221934318542, 0.9727333188056946]
val_loss: [1.5258727073669434, 1.026525855064392, 0.8277543187141418, 0.80281662940979, 0.8203626871109009, 0.7537364959716797, 0.9040626883506775, 1.0693464279174805, 1.0425690412521362, 1.1762558221817017]
val_accuracy: [0.45559999346733093, 0.6376000046730042, 0.7142000198364258, 0.7271999716758728, 0.7271999716758728, 0.7577999830245972, 0.7491999864578247, 0.7486000061035156, 0.7562000155448914, 0.7447999715805054]

Training time: 4249.44 seconds
accuracy: 0.7447999715805054
```