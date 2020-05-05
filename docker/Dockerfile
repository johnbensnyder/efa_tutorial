FROM nvidia/cuda:10.1-base-ubuntu18.04

LABEL maintainer="Amazon AI"

# prevent stopping by user interaction
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV SAGEMAKER_TRAINING_MODULE sagemaker_tensorflow_container.training:main

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Set environment variables for MKL
# For more about MKL with TensorFlow see:
# https://www.tensorflow.org/performance/performance_guide#tensorflow_with_intel%C2%AE_mkl_dnn
ENV KMP_AFFINITY=granularity=fine,compact,1,0
ENV KMP_BLOCKTIME=1
ENV KMP_SETTINGS=0

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

ARG TF_URL=https://tensorflow-aws.s3-us-west-2.amazonaws.com/2.1/AmazonLinux/gpu/final/tensorflow_gpu-2.1.0-cp36-cp36m-manylinux2010_x86_64.whl

ARG NCCL_VERSION=2.5.6-2
ARG OPENMPI_VERSION_MINOR=4.0.3
ARG OPENMPI_VERSION_MAJOR=4.0

RUN apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated \
    python3-dev \
    python3-pip \
    python3-setuptools \
    ca-certificates \
    cuda-command-line-tools-10-1 \
    cuda-cudart-dev-10-1 \
    cuda-cufft-dev-10-1 \
    cuda-curand-dev-10-1 \
    cuda-cusolver-dev-10-1 \
    cuda-cusparse-dev-10-1 \
    curl \
    libcudnn7=7.6.2.24-1+cuda10.1 \
    libgomp1 \
    libfreetype6-dev \
    libhdf5-serial-dev \
    libpng-dev \
    libzmq3-dev \
    git \
    wget \
    vim \
    build-essential \
    openssh-client \
    openssh-server \
    zlib1g-dev \
    # Install dependent library for OpenCV
    libgtk2.0-dev \
    #cuda-cublas-dev not available with 10-1, install libcublas instead
    #it will downgrade the cublas from 10-2 to 10-1
    #adding an extra flag --allow-downgrades for it
    && apt-get update \
    && apt-get install -y --no-install-recommends --allow-unauthenticated --allow-downgrades \
    libcublas10=10.1.0.105-1 \
    libcublas-dev=10.1.0.105-1 \
 && apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated  \
    nvinfer-runtime-trt-repo-ubuntu1804-5.0.2-ga-cuda10.0 \
 && apt-get update && apt-get install -y --no-install-recommends --allow-unauthenticated  \
    libnvinfer6=6.0.1-1+cuda10.1 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /var/run/sshd

##########################################################################
# Install NCCL, MPI, and EFA adapter
# Note that EFA driver does not install properly in Dockerfile
# But matching driver is needed to compile nccl adapter used
# within the docker container. The solution is to build
# the Docker image on the DL AMI by copying EFA drivers
# into Docker build environment, then temporarily copy
# drivers into image
# cp -R /opt/amazon/efa .
#########################################################################

RUN cd / \
    && wget https://github.com/NVIDIA/nccl/archive/v${NCCL_VERSION}.tar.gz \
    && tar -xf  v${NCCL_VERSION}.tar.gz \
    && mv nccl-${NCCL_VERSION} nccl \
    && cd nccl \
    && make -j src.build

# copy local efa drivers into image
COPY efa /efa

# Install Open MPI
RUN mkdir /tmp/openmpi \
 && cd /tmp/openmpi \
 && curl -fSsL -O https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_MAJOR}/openmpi-${OPENMPI_VERSION_MINOR}.tar.gz \
 && tar zxf openmpi-${OPENMPI_VERSION_MINOR}.tar.gz \
 && cd openmpi-${OPENMPI_VERSION_MINOR} \
 && ./configure --enable-orterun-prefix-by-default \
 && make -j $(nproc) all \
 && make install \
 && ldconfig \
 && rm -rf /tmp/openmpi

RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/openmpi/bin/:$PATH
ENV PATH=/usr/local/nvidia/bin:$PATH

# Download EFA NCCL adapter
RUN cd / \
    && git clone https://github.com/aws/aws-ofi-nccl.git -b aws \
    && cd aws-ofi-nccl \
    && apt-get update && apt-get install -y libudev-dev dh-autoreconf \
    && ./autogen.sh \
    && ./configure \
       --with-libfabric=/efa \
       --with-cuda=/usr/local/cuda \
       --with-nccl=/nccl/build \
       --with-mpi=/usr/local \
       --prefix=/aws-ofi-nccl/install \
    && make && make install

# Delete efa driver from image
RUN rm -rf /efa

#########################################################
# Fix ssh for container communication
#########################################################

# SSH login fix. Otherwise user is kicked off after login
RUN mkdir -p /var/run/sshd \
 && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# Create SSH key.
RUN mkdir -p /root/.ssh/ \
 && ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa \
 && cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys \
 && printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new \
 && echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new \
 && mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

WORKDIR /

######################################################
# Update pip
######################################################

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

# Some TF tools expect a "python" binary
RUN ln -s $(which ${PYTHON}) /usr/local/bin/python \
 && ln -s $(which ${PIP}) /usr/bin/pip

###########################################################
# Install Tensorflow and utilities
###########################################################

# install PyYAML==5.1.2 to avoid conflict with latest awscli
# # python-dateutil==2.8.0 to satisfy botocore associated with latest awscli
RUN ${PIP} install --no-cache-dir -U \
    numpy==1.17.5 \
    scipy==1.2.2 \
    scikit-learn==0.22 \
    pandas==1.0.1 \
    Pillow==7.0.0 \
    h5py==2.10.0 \
    keras_applications==1.0.8 \
    keras_preprocessing==1.1.0 \
    keras==2.3.1 \
    smdebug==0.7.1 \
    python-dateutil==2.8.1 \
    pyYAML==5.3.1 \
    requests==2.22.0 \
    awscli \
    mpi4py==3.0.3 \
    opencv-python==4.2.0.32 \
    sagemaker==1.50.17 \
    sagemaker-experiments==0.1.7 \
    "sagemaker-tensorflow>=2.1,<2.2" \
    "sagemaker-tensorflow-training>2,<4" \
    # Let's install TensorFlow separately in the end to avoid
    # the library version to be overwritten
 && ${PIP} install --no-cache-dir -U \
    ${TF_URL}

#########################################################
# Install Horovod
# point toward NCCL header files used for EFA
#########################################################

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs \
 && HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_NCCL_HOME=/nccl/build ${PIP} install --no-cache-dir horovod==0.18.2 \
 && ldconfig


ADD https://raw.githubusercontent.com/aws/aws-deep-learning-containers-utils/master/deep_learning_container.py /usr/local/bin/deep_learning_container.py

RUN chmod +x /usr/local/bin/deep_learning_container.py

RUN curl https://aws-dlc-licenses.s3.amazonaws.com/tensorflow-2.1/license.txt -o /license.txt

##########################################
# Additional custom utilities for training
##########################################

# Nvidia Dali

ARG DALI_VERSION=0.20.0
RUN ${PIP} install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 \
            nvidia-dali==$DALI_VERSION && \
    ${PIP} install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 \
            nvidia-dali-tf-plugin==$DALI_VERSION

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt install -y python3-opencv libopenblas-base \
	libomp-dev build-essential

RUN ${PIP} install tensorflow_addons \
                s3fs \
                imgaug \
                tqdm \
                tensorflow_datasets \
                scikit-image \
                cython \
                addict \
                terminaltables \
                numba && \
    pip install pycocotools

##########################################
# Additional components for notebook use
##########################################

# Nodejs for jupyter lab extensions

RUN curl -sL https://deb.nodesource.com/setup_13.x  | bash -
RUN apt-get -y install nodejs
RUN npm install

RUN ${PIP} install tornado

RUN ${PIP} install jupyter \
                jupyterlab==1.2.0 \
                tensorflow_addons \
                ipywidgets \
                matplotlib \
                seaborn

RUN ${PIP} install jupyterlab-nvdashboard \
                jupyter-tensorboard && \
    jupyter labextension install @jupyter-widgets/jupyterlab-manager \
                                 jupyterlab-nvdashboard jupyterlab_tensorboard

RUN mkdir /workspace
WORKDIR /workspace

CMD nohup jupyter lab --allow-root --ip=0.0.0.0 --no-browser > notebook.log

########################################################################
# Build docker image
########################################################################
#
# DOCKERHUB_USER=johnbensnyder
# IMAGE_NAME=sagemaker_tf:2.1.0_efa
# docker build -t ${DOCKERHUB_USER}/${IMAGE_NAME} -f Dockerfile.efa .

##################################################################
# Push to Dockerhub
##################################################################

# docker login
# docker push ${DOCKERHUB_USER}/${IMAGE_NAME}

########################################################################
# login to ECS
########################################################################
#
# ECR_USER=jbsnyder
# ALGO_NAME=sagemaker-tf-2.1.0-efa
# region=$(aws configure get region)
# account=$(aws sts get-caller-identity --query Account --output text)
# fullname="${account}.dkr.ecr.${region}.amazonaws.com/${ECR_USER}:${ALGO_NAME}"
# $(aws ecr get-login --region ${region} --no-include-email)
# docker tag ${DOCKERHUB_USER}/${IMAGE_NAME} ${fullname}
# docker push ${fullname}
#
########################################################################
# 578276202366.dkr.ecr.us-east-1.amazonaws.com/jbsnyder:sagemaker-tf-2.1.0
########################################################################
#
#
#
# docker run --rm --gpus 8 \
# --name mpicont \
# --net=host --uts=host --ipc=host \
# --ulimit stack=67108864 --ulimit memlock=-1 \
# --security-opt seccomp=unconfined \
# -v /opt/amazon/efa:/efa \
# --device=/dev/infiniband/uverbs0 -it \
# johnbensnyder/sagemaker_tf:2.1.0_efa \
# bash
#
#