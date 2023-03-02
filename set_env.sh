export EGEN_HOME=/data/i3home/${USER}/egen/repositories/event-generator
export CONFIG_DIR=$EGEN_HOME/configs/training 

export I3_BUILD=/data/condor_builds/users/sgray/icetray/build
export I3_SRC=/data/condor_builds/users/sgray/icetray/src

source ${EGEN_HOME}/py3-v4.1.1_tensorflow2.3/bin/activate

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$I3_BUILD/lib
export PYTHONPATH=$PYTHONPATH:$I3_BUILD/lib

