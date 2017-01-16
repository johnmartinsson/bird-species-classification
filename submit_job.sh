#!/bin/bash

#create log dir (if it doesn't exists)
[ -d log ] || mkdir log 

#submit job, asking for one GPU, and redirecting output to log files
qsub -cwd \
  -l gpu=1\
  -e ./log/run_job.sh.error \
  -o ./log/run_job.sh.log \
  ./run_job_evaluate.sh
