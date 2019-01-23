# Requiremets
To run this coe, a Linux machine with python3 is needed. There is a high chance it will also work on Mac but it wasn't tested. 

Additionally python3 dependencies from **requirements.txt** (sudo pip3 install -r requirements.txt).

Internet connection is also needed (to download data).

## Docker    
If you wish so, you can run the scripts in docker, a Dockerfile and launch script is prepared (see: scripts/docker_build_n_run.sh). In this case no dependencies except for Docker are required

# How to run

Just run:
```bash
./scripts.icml_reproduce.sh
```

You should see this output on the console:

```
# Optimizers: 26
# Models: 1
Tests in total: 26
Downloading bank-additional.zip 101.3%
Successfully downloaded bank-additional.zip 444572 bytes.
Running optimizers for dataset: 'UCI_Bank', model: 'LR{'init0': True}'
adagrad_l1.0:   0%|                                      | 0/10 [00:00<?, ?it/s]
```
Do not be bothered by the fact that more than 100% is downloaded. Our scripts work hard!

>>> Unfortunately reproducing the whole experiment will take much time on a single machine (More than a day most likely) because the code was created with more focus on deep models and batchsize>1.

# Output
The scripts create **tb_logs_linear** directory with summaries from tensorflow, however do not try to run it via tensorboard because so much data will clog your ram. This directory weighs ~2.7 GB because tensordflow apparently can't write data efficiently.


Additionally **graphs_linear** directory will be created with graphs just like those used in the paper and more (separate graphs for each algorithm and runs for learning rates not shown in the paper).

## Artificial experiment
**tb_logs_art** are also created which contain logs from artificial experiment. To see them run:

```bash
tensorboard --logdir=tb_logs_art
```





