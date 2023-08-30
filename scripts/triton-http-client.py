from tritonclient.utils import *
import tritonclient.http as httpclient
import sys
import argparse
import numpy as np
import time
import threading
from datetime import datetime, timedelta
 
def run_http_benchmark(threadID = False, model_name = False, batch_size = False, limit = False, benchmark = True, num_threads = False):
    class producer_thread(threading.Thread):
        def __init__(self, threadID, model_name, batch_size, limit):
            threading.Thread.__init__(self)
            self.threadID = threadID
            self.model_name = model_name
            self.batch_size = batch_size
            self.limit = limit
            self.counter = 0
     
        def run(self):
            print(f'RUN Thread {self.threadID} for {self.limit} seconds')
            exec_times = []
            latency_list = []
            # model_name = "rn50-4neuroncores-bs1x1"
            BATCH_SIZE = batch_size
            shape = [BATCH_SIZE ,224,224,3]
            start_time = datetime.now()
            while True:
                with httpclient.InferenceServerClient("localhost:8000") as client:
                    input0_data = np.random.rand(*shape).astype(np.float32)
                    # print(input0_data)
                    inputs = [
                        httpclient.InferInput("input", input0_data.shape,
                                np_to_triton_dtype(input0_data.dtype)),
                    ]
 
                    inputs[0].set_data_from_numpy(input0_data)
 
                    outputs = [
                        httpclient.InferRequestedOutput("output"),
                    ]
                    t_inf_start = time.time()
                    response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)
                    t_end = time.time()
 
                    result = response.get_response()
                    delta_t = t_end - t_inf_start
                    latency_list.append(delta_t)
 
                    self.counter +=1
 
                    if((datetime.now() - start_time) > self.limit):
                        F = 1 # fraction to use for calculation. 1 = 100%.
                        selected = int(len(latency_list)*F) 
                        p50 = np.quantile(latency_list[-selected:],0.50) * 1000
                        p75 = np.quantile(latency_list[-selected:],0.75) * 1000
                        p95 = np.quantile(latency_list[-selected:],0.95) * 1000
                        p99 = np.quantile(latency_list[-selected:],0.99) * 1000
            
                        print(f"Thread {self.threadID}: Completed {self.counter} http requests in {(datetime.now() - start_time).total_seconds()} seconds.")
                        throughput = self.counter / (datetime.now() - start_time).total_seconds()
                        print(f"Thread {self.threadID} Throughput per second: ", throughput * BATCH_SIZE)
                        print(f"Thread {self.threadID} Latency p50 in ms: ", p50)
                        print(f"Thread {self.threadID} Latency p75 in ms: ", p75)
                        print(f"Thread {self.threadID} Latency p95 in ms: ", p95)
                        print(f"Thread {self.threadID} Latency p99 in ms: ", p99)
                        break
                        print(f"Sender thread {thread_id} terminating")
 
    time_limit = timedelta(seconds=limit)
 
    for i in range(num_threads):
        a_thread = producer_thread(i, model_name, batch_size, time_limit)
        a_thread.start()
    print(f">>> Running {num_threads} threads")
 
    return 
 
def run_http_client(model_name = False, batch_size = False, benchmark = False, limit = False, num_threads = False):
    print(f'Model name = {model_name}, batch size = {batch_size}')
 
    shape = [batch_size ,224,224,3]
    with httpclient.InferenceServerClient("localhost:8000") as client:
        input0_data = np.random.rand(*shape).astype(np.float32)
 
        inputs = [
            httpclient.InferInput("input", input0_data.shape,
                              np_to_triton_dtype(input0_data.dtype)),
                              #np.float32),
        ]
 
        inputs[0].set_data_from_numpy(input0_data)
 
        outputs = [
        httpclient.InferRequestedOutput("output"),
        ]
 
        t_inf_start = time.time()
        response = client.infer(model_name,
                            inputs,
                            request_id=str(1),
                            outputs=outputs)
        t_end = time.time()
 
        result = response.get_response()
        output0_data = response.as_numpy("output")
        print("=====")
        print(output0_data)
        print("=====")
        print(f'PASS: {model_name}')
        print('##### Model inference time in ms: ', (t_end - t_inf_start) * 1000)
 
 
 
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='provide a model name')
    parser.add_argument('-b', '--batch_size', type=int, required=True, help='batch_size must be at least number of active cores')
    parser.add_argument('-k',  '--benchmark', action='store_true',  help='use this option if doing benchmark')
    parser.add_argument('-l',  '--limit', type=int, required = False,  default=60, help='number of second to run benchmark')
    parser.add_argument('-t',  '--num_threads', type=int, required = True, help='number of threads to run benchmark')
 
 
    opt = parser.parse_args()
    return opt
 
def main(opt):
    
    run_http_client(**vars(opt))
    if opt.benchmark == True: 
        print('##### WARMUP COMPLETE, NOW DO BENCHMARK #####')
        print(f'##### Model name: {opt.model_name}, Batch size: {opt.batch_size}')
        run_http_benchmark(**vars(opt))
 
 
 
if __name__ == "__main__":
    opt = parse_opt()
    main(opt)