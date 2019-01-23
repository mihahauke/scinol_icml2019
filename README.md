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

You should see this output on the console.

```bash
todo
```

# Output
The scripts create **tb_logs_linear** directory with summaries from tensorflow, however do not try to run it via tensorboard because so much data will clog your ram. This directory weighs ~2.7 GB because tensordflow apparently can't write data efficiently.

Additionally **graphs_linear** directory will be created with graphs just like those used in the paper and more (separate graphs for each algorithm and runs for learning rates not shown in the paper).




