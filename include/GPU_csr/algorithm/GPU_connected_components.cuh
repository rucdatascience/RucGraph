#ifndef UF_CUH
#define UF_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <GPU_csr/GPU_csr.hpp>

std::vector<int> gpu_connected_components(CSR_graph<double>& input_graph, int threads = 1024000);

__device__ int findRoot(int* parent, int i);
__global__ void Hook(int* parent, int* Start_v, int* End_v, int E);

std::vector<std::pair<std::string, std::string>> Cuda_WCC(graph_structure<double>& graph, CSR_graph<double>& csr_graph);

__device__ int findRoot(int* parent, int i) {
    //Recursively searching for the ancestor of node i
    int par = parent[i];
    if (par != i) {
        int next, prev = i;
        while (par > (next = parent[par])) {
            parent[prev] = next;
            prev = par;
            par = next;
        }
    }
    return par;
}

__global__ void Hook(int* parent, int* Start_v, int* End_v, int E, int threads, int work_size) {
    //Merge operations on each edge
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    //Calculate thread ID
    if (id < threads) {
        int start = id * work_size;
        int end = min(start + work_size, E);

        for (int i = start; i < end; i++) {
            //A thread may handle multiple edges
            int u = Start_v[i];
            int v = End_v[i];
            //u,v are the starting and ending vertex of the edge
            int rootU = findRoot(parent, u);
            int rootV = findRoot(parent, v);
            //Obtain Root Node
            while (rootU != rootV) {
                int expected = rootU > rootV ? rootU : rootV;
                int desired = rootU < rootV ? rootU : rootV;
                //During multi-core operations, the root node may be manipulated by other threads, so locking is necessary for the operation
                int observed = atomicCAS(&parent[expected], expected, desired);
                /*
                compare and swap
                int atomicCAS(int* address, int compare, int val);
                Check if the address and compare are the same. If they are the same, enter address as desired. Otherwise, no action will be taken
                observed = parent[expected]

                */
                
            
                if (observed == expected)//If the observed values are correct and the merge operation is successful, exit the loop
                    break;
                //If the observed value has been modified, the modified new root node needs to be obtained
                rootU = findRoot(parent, u);
                rootV = findRoot(parent, v);
            }
        }
    }
}

//template <typename T>
std::vector<int> gpu_connected_components(CSR_graph<double>& input_graph, int threads) {
    //Using BFS method to find connectivity vectors starting from each node
    int N = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();
    //Number of nodes and edges
    
    int* Start_v;
    int* End_v;
    int* Parent;
    // Allocate GPU memory
    cudaMallocManaged((void**)&Start_v, E * sizeof(int));
    cudaMallocManaged((void**)&End_v, E * sizeof(int));
    cudaMallocManaged((void**)&Parent, N * sizeof(int));
    cudaDeviceSynchronize();
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return std::vector<int>();
    }
    //Forming an edge list
    // Copy data to GPU
    for (int i = 0; i < N; i++) {
        for (int j = input_graph.OUTs_Neighbor_start_pointers[i]; j < input_graph.OUTs_Neighbor_start_pointers[i + 1]; j++) {
			Start_v[j] = i;
			End_v[j] = input_graph.OUTs_Edges[j];
		}
        Parent[i] = i;//initialization
    }

    int threadsPerBlock = 1024;
    int blocksPerGrid = 0;
    //Disperse E operations on threads
    if (E < threads)
        threads = E;

    blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;
    int work_size = (E + threads - 1) / threads;

    Hook<<<blocksPerGrid, threadsPerBlock>>>(Parent, Start_v, End_v, E, threads, work_size);
    cudaDeviceSynchronize();
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
		return std::vector<int>();
	}
    // Process components on CPU
    std::vector<int> result;
    //Using a linked list to record connected components
    for (int i = 0; i < N; i++) {
        if (Parent[i] != i) {
            //If it is not the root node, add the node to the linked list of the root node it belongs to
            int j = i;
            while (Parent[j] != j)
                j = Parent[j];
            Parent[i] = j;
            result.push_back(j);
        }
        else  //The root node is directly added to the root node linked list
            result.push_back(i);
    }

    // Free GPU memory
    cudaFree(Start_v);
    cudaFree(End_v);
    cudaFree(Parent);

    return result;
}


std::vector<std::pair<std::string, std::string>> Cuda_WCC(graph_structure<double>& graph, CSR_graph<double>& csr_graph) {
    std::vector<int> wccVecGPU = gpu_connected_components(csr_graph);
    return graph.res_trans_id_id(wccVecGPU);
}

#endif