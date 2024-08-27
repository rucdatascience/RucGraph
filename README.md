# RucGraph - a fast graph database system on CPU/GPU platforms

RucGraph is a lightweight graph database system that uses both CPUs and GPUs to efficiently perform graph analyses, such as Shortest Path, PageRank, Community Detection etc.


- "Ruc" is the abbreviation of "[Renmin University of China](https://www.ruc.edu.cn/)".


- RucGraph works efficiently on large graphs with billions of vertices and edges. In particular, on [LDBC Graphalytics Benchmarks](https://ldbcouncil.org/benchmarks/graphalytics/), RucGraph is <b>10 times faster than [neo4j](https://neo4j.com) on CPUs</b>, and <b>50 times faster than  [neo4j](https://neo4j.com) on GPUs</b>.




## Graph data structures

RucGraph is now using [Adjacency Lists](https://www.geeksforgeeks.org/adjacency-list-meaning-definition-in-dsa/) to store graphs in CPU memory, and using [Sparse Matrix Representations](https://www.geeksforgeeks.org/sparse-matrix-representations-set-3-csr/) (CSRs) to store graphs in GPU memory. 

More diversified functions, such as using Adjacency Lists in GPU memory, is now under development.


## Code File structures

- `include/`: header files

- `include/CPU_adj_list/`: header files for operating Adjacency Lists on CPUs

- `include/CPU_adj_list/CPU_adj_list.hpp`: an Adjacency List on CPUs

- `include/CPU_adj_list/algorithm/`: header files for graph analysis operators on CPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have passed the LDBC Graphalytics Benchmark test



- `include/GPU_csr/`: header files for operating CSRs on GPUs

- `include/GPU_csr/GPU_csr.hpp`: a CSR on GPUs

- `include/GPU_csr/algorithm/`: header files for graph analysis operators on GPUs, such as Shortest Path, PageRank, Community Detection operators; these operators have also passed the LDBC Graphalytics Benchmark test


- `include/LDBC/`: header files for performing the LDBC Graphalytics Benchmark test



 <br /> 
 

- `src/`: source files
- `src/CPU_adj_list/CPU_example.cpp`: an example of performing graph analysis operators on CPUs
- `src/GPU_csr/GPU_example.cu`: an example of performing graph analysis operators on GPUs
- `src/LDBC/LDBC_CPU_adj_list.cpp`: the source codes of performing the LDBC Graphalytics Benchmark test on CPUs
- `src/LDBC/LDBC_GPU_csr.cu`: the source codes of performing the LDBC Graphalytics Benchmark test on GPUs



## Copy & Run

Here, we show how to build & run RucGraph on a Linux server with the Ubuntu 20.04 system, 2 Intel(R) Xeon(R) Gold 5218 CPUs, and an NVIDIA GeForce RTX 3090 GPU. The environment is as follows.

- `cmake --version`: cmake version 3.27.9
- `g++ --version`: g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
- `nvidia-smi`: NVIDIA-SMI 550.54.14         /      Driver Version: 550.54.14   /   CUDA Version: 12.4


First, download the files onto the server, e.g., onto the following path: `/home/username/RucGraph`. Second, enter the following commands on a terminal at this path:

```shell
username@server:~/RucGraph$ mkdir build
username@server:~/RucGraph$ cd build
username@server:~/RucGraph/build$ cmake .. -DBUILD_CPU=ON -DBUILD_GPU=ON
username@server:~/RucGraph/build$ make
username@server:~/RucGraph/build$ ./bin_cpu/CPU_example
username@server:~/RucGraph/build$ ./bin_gpu/GPU_example
username@server:~/RucGraph/build$ ./bin_cpu/Test_CPU
username@server:~/RucGraph/build$ ./bin_gpu/Test_GPU
```

There are some explanations for the above commands:

- `-DBUILD_CPU=ON -DBUILD_GPU=ON` is to compile both CPU and GPU codes. If GPUs are not available, then we can change `-DBUILD_GPU=ON` to `-DBUILD_GPU=OFF`.


- `./bin_cpu/CPU_example` is to run the source codes at `src/CPU_adj_list/CPU_example.cpp`

- `./bin_gpu/GPU_example` is to run the source codes at `src/GPU_csr/GPU_example.cu`

- `./bin_cpu/Test_CPU` is to run the source codes at `src/LDBC/LDBC_CPU_adj_list.cpp`

- `./bin_gpu/Test_GPU` is to run the source codes at `src/LDBC/LDBC_GPU_csr.cu`

We can run "CPU_example" and "GPU_example" without any graph dataset. The outputs of graph analysis operators will be printed on the terminal. 

Nevertheless, before running "Test_CPU" and "Test_GPU", we need to download the [LDBC Graphalytics datasets](https://repository.surfsara.nl/datasets/cwi/graphalytics) at first. Then, when running "Test_CPU" and "Test_GPU", the program will ask us to input the data path and name sequentially. 
```shell
Please input the data directory: # The program asks
/home/username/data # Input the data path
Please input the graph name: # The program asks
datagen-7_5-fb # Input a data name
```

After inputting the data path and name, the program will perform the LDBC Graphalytics Benchmark test for this dataset. Specifically, the program will print some parameters of this test, as well as the consumed times of different graph analysis operators on this dataset.


## License

RucGraph is released under the [Apache 2.0 license](LICENSE.txt).
