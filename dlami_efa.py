################################################################
# Launch DLAMI with EFA
################################################################

import boto3
import yaml

with open("efa_tutorial/dlami.yaml") as in_config:
    config = yaml.safe_load(in_config)

ec2_session = boto3.Session(region_name="us-east-1")
ec2_client = ec2_session.client("ec2")
ec2_resource = ec2_session.resource("ec2")

response = ec2_client.run_instances(**config)

################################################################
# Create SSH interface to all instances
################################################################
import sys
sys.path.append('efa_tutorial')
import ssh
from time import sleep

while True:
    try:
        instances = [instance['InstanceId'] for instance in response['Instances']]
        status = ec2_resource.meta.client.describe_instances(InstanceIds=instances)
        public_ips = [instance['PublicIpAddress'] for instance in status['Reservations'][0]['Instances']]
        ssh_client = ssh.SSH(public_ips, '/Users/jbsnyder/.aws/jbsnyder.pem')
        # wait a few seconds and run a simple command to make sure instances are up
        pci = ssh_client.run_on_all('lspci')
        break
    except:
        sleep(10)
        continue
print(pci[0]['stdout'])
################################################################
# Use local AWS credentials for EC2
################################################################

import getpass
import configparser
credentials = configparser.ConfigParser()
credentials.read('/Users/{0}/.aws/credentials'.format(getpass.getuser()))
config = configparser.ConfigParser()
config.read('/Users/{0}/.aws/config'.format(getpass.getuser()))

ssh_client.run_on_all('aws configure set aws_access_key_id {}'.format(credentials['default']['aws_access_key_id']))
ssh_client.run_on_all('aws configure set aws_secret_access_key {}'.format(credentials['default']['aws_secret_access_key']))
ssh_client.run_on_all('aws configure set default.region {}'.format(config['default']['region']))

del credentials
del config

################################################################
# Update EFA Driver to 1.8.4 Takes about 2 minutes
################################################################

ssh_client.scp_local_to_all('efa_tutorial/setup_scripts/efa_setup.sh', 'efa_setup.sh')

ssh_client.run_on_all('./efa_setup.sh')

version_check = ssh_client.run_on_all('/opt/amazon/openmpi/bin/mpirun --version')

while not all(['4.0.3' in i['stdout'] for i in  version_check]):
    sleep(10)
    ssh_client.run_on_all('./efa_setup.sh')
    version_check = ssh_client.run_on_all('/opt/amazon/openmpi/bin/mpirun --version')
print(version_check[0]['stdout'])
################################################################
# Check to make sure driver is updated
# should be 4.0.3
################################################################


################################################################
# mount nvme drive
################################################################

ssh_client.run_on_all('mkdir -p ~/shared_workspace')
ssh_client.run_on_all('sudo mkfs -t xfs /dev/nvme0n1')
ssh_client.run_on_all('sudo mount /dev/nvme0n1 ~/shared_workspace')
ssh_client.run_on_all('mkdir -p ~/shared_workspace/data')
ssh_client.run_on_all('sudo chmod -R 777 ~/shared_workspace')

# download coco data

download_coco = "aws s3 cp --recursive s3://jbsnyder-sagemaker/faster-rcnn/data/ ~/shared_workspace/data"

coco_thread = ssh_client.run_on_all(download_coco, wait=False)

################################################################
# Build Docker image Takes about 10 minutes
# Only run first time
################################################################

first_run=False

if first_run:
    dockerhub_user = 'johnbensnyder'
    dockerhub_repo = 'efa'
    dockerhub_tag = 'dlami_28'

    ssh_client.scp_local_to_master('efa_tutorial/docker', 'docker', recursive=True)
    ssh_client.run_on_master('cp -R /opt/amazon/efa docker/')
    ssh_client.run_on_master('cd docker && docker build -t {}/{}:{} .'.format(dockerhub_user,
                                                                              dockerhub_repo,
                                                                              dockerhub_tag))

################################################################
# Deploy Docker image to all nodes
# Only run first time
################################################################
if first_run:
    # Warning: bug in ipykernel can sometimes cause password to echo. recommend run in standard python
    import getpass
    dh_password = getpass.getpass('enter dockerhub password')
    ssh_client.run_on_master('docker login --username {} --password {}'.format(dockerhub_user, dh_password))
    del dh_password

    ssh_client.run_on_master('docker push {}/{}:{}'.format(dockerhub_user,
                                                           dockerhub_repo,
                                                           dockerhub_tag))

    ssh_client.run_on_workers('docker pull {}/{}:{}'.format(dockerhub_user,
                                                            dockerhub_repo,
                                                            dockerhub_tag))

################################################################
# After first run, just pull image to nodes
################################################################
dockerhub_user = 'johnbensnyder'
dockerhub_repo = 'efa'
dockerhub_tag = 'dlami_28'
ssh_client.run_on_all('docker pull {}/{}:{}'.format(dockerhub_user,
                                                        dockerhub_repo,
                                                        dockerhub_tag))

################################################################
# Setup internode communication
################################################################

private_ips = [instance['PrivateIpAddress'] for instance in status['Reservations'][0]['Instances']]
ssh.create_hostfile(ssh_client, private_ips)
ssh.create_ssh_comm(ssh_client)
ssh.setup_container_communication(ssh_client)

################################################################
# Setup Containers
################################################################
dockerhub_user = 'johnbensnyder'
dockerhub_repo = 'efa'
dockerhub_tag = 'dlami_28'

launch_cont = """docker run --rm -it -d --gpus all \
                    --name mpicont \
                    --net=host --uts=host --ipc=host \
                    --ulimit stack=67108864 --ulimit memlock=-1 \
                    --security-opt seccomp=unconfined \
                    -v /opt/amazon/efa:/efa \
                    -v /home/ubuntu/ssh_container:/root/.ssh \
                    -v ~/shared_workspace:/workspace/shared_workspace \
                    --device=/dev/infiniband/uverbs0 \
                    {0}/{1}:{2}
                    """.format(dockerhub_user, dockerhub_repo, dockerhub_tag)

ssh_client.run_on_all(launch_cont)

# setup tests and unarchive data

ssh_client.run_on_all('docker exec mpicont /bin/bash -c "cd /workspace/shared_workspace && git clone https://github.com/NVIDIA/nccl-tests.git"')

ssh_client.run_on_all('docker exec mpicont /bin/bash -c "cd /workspace/shared_workspace/nccl-tests && make MPI=1 MPI_HOME=/usr/local/ NCCL_HOME=/nccl/build"')

# make sure coco download is complete before unarchiving
while not all([i.done() for i in coco_thread]):
    sleep(1)
    continue
ssh_client.run_on_all('cd ~/shared_workspace/data/coco && tar -xf coco.tar')

ssh_client.run_on_all("cd shared_workspace && git clone -b staging https://github.com/aws-samples/deep-learning-models")

notebook = ssh.Notebook(ssh_client)

print(notebook.get_token())

################################################################
# Run NCCL Tests
################################################################
import re
nccl_efa_command = """
mpirun -x FI_PROVIDER="efa" \
            --allow-run-as-root \
            -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efa/lib:/usr/local/lib:/nccl/build/lib:/aws-ofi-nccl/install/lib \
            -x NCCL_DEBUG=INFO \
             -x NCCL_TREE_THRESHOLD=0 \
             -x NCCL_SOCKET_IFNAME=ens5 \
             --hostfile /root/.ssh/hosts \
             --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
             --mca btl_vader_single_copy_mechanism none \
             --mca oob_tcp_if_include ens5 \
             --mca btl_tcp_if_include ens5 \
             --oversubscribe \
             /workspace/shared_workspace/nccl-tests/build/all_reduce_perf \
                 -b 8 -e 4G -f 2 -g 1 -c0
"""

efa_result = ssh_client.run_on_master('docker exec mpicont bash -c \"{}\"'.format(nccl_efa_command))

efa_bandwidth = float(re.findall("\d+\.\d+", efa_result['stdout'].split(':')[-1])[0])
print("EFA bandwidth: {}".format(efa_bandwidth))

################################################################
# Launch Training
################################################################


training_launch = """ 
mpirun --allow-run-as-root \
            -x FI_PROVIDER=\\\"efa\\\" \
            -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/efa/lib:/usr/local/lib:/nccl/build/lib:/aws-ofi-nccl/install/lib \
            -x NCCL_DEBUG=INFO \
             -x NCCL_TREE_THRESHOLD=0 \
             --hostfile /root/.ssh/hosts \
             --mca plm_rsh_no_tree_spawn 1 -bind-to none -map-by slot -mca pml ob1 \
             --mca btl_vader_single_copy_mechanism none \
             --mca oob_tcp_if_include ens5 \
             --mca btl_tcp_if_include ens5 \
             python /workspace/shared_workspace/deep-learning-models/models/vision/detection/tools/train_docker.py \
             --configuration /workspace/shared_workspace/deep-learning-models/models/vision/detection/configs/docker_default_config.py \
             --base_learning_rate 15e-3 \
             --batch_size_per_device 4 \
             --fp16 True \
             --schedule 1x \
             --warmup_init_lr_scale 3.0 \
             --warmup_steps 1000 \
             --use_rcnn_bn False \
             --use_conv True \
             --ls 0.0 \
             --name demo

"""

ssh_client.run_on_master('mkdir -p ~/shared_workspace/logs')
training_thread = ssh_client.run_on_master("""nohup docker exec mpicont bash -c \"{}\" &> ~/shared_workspace/logs/out.log &""".format(training_launch))

notebook.disconnect()

ssh_client.run_on_all("docker stop mpicont")


ec2_client.terminate_instances(InstanceIds=instances)