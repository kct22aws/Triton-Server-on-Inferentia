# Triton-Server-on-Inferentia

This repo documents how to set up and run Nvidia Triton Inferencing Server on AWS Inferentia.

Nvidia Triton Server is built to be a versatile machine learning serving platform which supports various model frameworks and the serving hardware. It has several backends to support major machine learning model frameworks such as TensorFlow, PyTorch, TensorRT, OpenVINO. The implementation is a client-server architecture, where it supports both http and gRPC protocols. 

## Set up server

We will use a Nvidia Triton server docker container as the environment to launch the server. The Triton Server will be invoked within the docker container. In Inf1 instance, we need to clone [Triton server with Python backend](https://github.com/triton-inference-server/python_backend.git) repo for this example. The branch used for this example is r22.04. In the Inf1 instance, open a terminal and run the following command:

```
git clone https://github.com/triton-inference-server/python_backend.git -b r22.04
```

This will create a python_backend directory in the Inf1 instance. For example, the absolute path to this directory may be: 

```
/home/ubuntu/python_backend
```

In this repo, we need to make a few edits for file [python_backend/inferentia/scripts/setup.sh](https://github.com/triton-inference-server/python_backend/blob/main/inferentia/scripts/setup.sh). The purpose of edits are to ensure all necessary dependencies as well make program are properly installed. The edited script is provided and may be accessed [here](./scripts/setup.sh). Replace the original setup.sh with this one and name it as setup.sh. The changes are as follow in the original script: 

1. update libraries in [setup.sh](https://github.com/triton-inference-server/python_backend/blob/main/inferentia/scripts/setup.sh):

```
apt update
apt install rapidjson-dev zlib1g-dev libarchive-dev
pip install cmake --upgrade 
```

2. specify repo tag explicitly when compiling a new virtual environment in [setup.sh](https://github.com/triton-inference-server/python_backend/blob/main/inferentia/scripts/setup.sh):

```
cmake -DTRITON_ENABLE_GPU=ON -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install \
  -D TRITON_BACKEND_REPO_TAG=r22.04 \
  -D TRITON_COMMON_REPO_TAG=r22.04 \
  -D TRITON_CORE_REPO_TAG=r22.04  \
```

Now run the following docker command from your Inf1 instance CLI to launch a Triton inference container:

```
docker run \
--device /dev/neuron0 \
--device /dev/neuron1 \
--device /dev/neuron2 \
--device /dev/neuron3 \
-v /home/ubuntu/python_backend:/home/ubuntu/python_backend \
-v /lib/udev:/mylib/udev \
--shm-size=1g —ulimit memlock=-1 \
-p 8000:8000 -p 8001:8001 -p 8002:8002 \
--ulimit stack=67108864 -ti nvcr.io/nvidia/tritonserver:22.04-py3 \
```

This command will download a Nvidia image and run a docker container, from which you may launch a Triton server for inference. There are four NeuronDevices in Inf1.6xl, we specify --device according to number of NeuronDevices available in the instance (If you are using Inf1.xl or 2xl, then there is only one NeuronDevice and --device /dev/neuron0 would suffice). Also, we specify /mylib/udev which is used for passing Neuron parameters. 


## Launch Triton Inference Container

If you have not downloaded the Triton image previously, it will first start downloading the image. Once completed, the image will be launched and now you are now in the container’s default directory:

```
root@7c9cbfab8dae:/opt/tritonserver#
```

From the container, you will proceed to build a virtual environment with required libraries for Triton server. Run the following command:

```
source /home/ubuntu/python_backend/inferentia/scripts/setup.sh -t
```

Once it is complete, this command will take you to the virtual environment (`test_conda_env`) it created: 

```
(test_conda_env) root@40de218cf275:/home/ubuntu#
```

To launch Triton server from this container, use the following command:

```
tritonserver --model-repository /home/ubuntu/python_backend/models
```

You should expect to see all the models in  models directory. For example:

```
I0605 20:53:54.781187 6941 server.cc:619]
+-----------------------------+---------+--------+
| Model | Version | Status |
+-----------------------------+---------+--------+
| rn50-16neuroncores-default | 1 | READY |
+-----------------------------+---------+--------+
```

and the server is listening for incoming requests in designated ports:

```
I0605 20:53:54.782316 6941 grpc_server.cc:4544] Started GRPCInferenceService at 0.0.0.0:8001
I0605 20:53:54.782554 6941 http_server.cc:3242] Started HTTPService at 0.0.0.0:8000
I0605 20:53:54.823433 6941 http_server.cc:180] Started Metrics Service at 0.0.0.0:8002
```

Now the Triton server is ready for inference request sent via either http or gRPC protocol. This server is now running from Triton container. You may verify the version of some of relevant packages that is now installed:

```
(test_conda_env) root@4ba033832cdf:/home/ubuntu# pip list | grep neuron
neuron-cc 1.11.4.0+97f99abe4
tensorboard-plugin-neuron 2.4.0.0
tensorflow-neuron 1.15.5.2.3.0.0
```

## Set up client

In the same Inf1 instance, open another terminal for setting up the client. Before launching the Triton inference container, copy and place a client script in python_backend directory. Overall, for client setup, it is similar to how we set up the server, i.e., we first download a Triton client docker image, and then run it as a container. The command below downloads Triton client image and runs it:

```
docker run -v /home/ubuntu/python_backend:/home/ubuntu/python_backend \ 
-ti —net host nvcr.io/nvidia/tritonserver:22.04-py3-sdk /bin/bash \
```

Once this command is complete, you will be in the Triton client container:

```
root@ip-xxx-xx-xx-xxx:/workspace#
```

From this directory, you will launch a Python script to run on Triton client container and make http request to the Triton server container. You are now inside Triton client container.

## Launch HTTP client

Once you are in the client container, you may run the following command to execute a benchmark routine for Resnet50 with randomized tensors. [The client script](./scripts/triton-http-client.py) may be launched with the following input parameters:

```
python3 /home/ubuntu/python_backend/triton-http-client.py \
--model_name rn50-16neuroncores-default \
--batch_size 16 \
--benchmark \
--limit 30 \
--num_threads 2 \
```

Since this example uses Inf1.6xl instance, which contains four NeuronDevices with a total of 16 NeuronCores, Triton server will split batch of input data such that every NeuronCore gets at least a sample of 1. This means that batch size must be equal to or more than number of NeuronCores. Therefore the example above set batch size of the request to 16. `—-benchmark` is used so that we may run this script to measure throughput and latency at the client side. `--limit` is set to 30 seconds for the benchmark routine. And we specified `—-num_threads`, which is the number of threads; each thread will keep generating http requests and send the request to server until the time set by `--limit` expires.

Below is an example output from the command above:

```
PASS: rn50-16neuroncores-default
##### Model inference time in ms: 45.36604881286621
##### WARMUP COMPLETE, NOW DO BENCHMARK #####
##### Model name: rn50-16neuroncores-default, Batch size: 16
RUN Thread 0 for 0:00:30 seconds
RUN Thread 1 for 0:00:30 seconds
>>> Running 2 threads
Thread 0: Completed 562 http requests in 30.019228 seconds.
Thread 0 Throughput per second: 299.54094816702394
Thread 0 Latency p50 in ms: 33.65468978881836
Thread 0 Latency p75 in ms: 35.676002502441406
Thread 0 Latency p95 in ms: 37.078678607940674
Thread 0 Latency p99 in ms: 38.220951557159424
Thread 1: Completed 562 http requests in 30.045218 seconds.
Thread 1 Throughput per second: 299.28186663751063
Thread 1 Latency p50 in ms: 34.45303440093994
Thread 1 Latency p75 in ms: 35.26878356933594
Thread 1 Latency p95 in ms: 36.66126728057861
Thread 1 Latency p99 in ms: 38.33159923553466

```

If you simply want to test the connection and execution of inference, you may use the following command:

```
python3 /home/ubuntu/python_backend/triton-http-client.py \
--model_name rn50-16neuroncores-default \
--batch_size 16 \
--limit 30 \
--num_threads 1 \
```

Expected output should be similar to this:

```
Model name = rn50-16neuroncores-default, batch size = 16
PASS: rn50-16neuroncores-default
##### Model inference time in ms: 45.3953742980957
```
In the command above, it only needed one thread to send a request with a specified batch size once. Therefore the end-to-end latency is measured only once.   

