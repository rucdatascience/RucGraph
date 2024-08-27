#ifndef SSSP_PRE_H
#define SSSP_PRE_H

#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GPU_csr/GPU_csr.hpp>
#include <GPU_csr/algorithm/GPU_shortest_paths.cuh>

__global__ void Relax_pre(int* out_pointer, int* out_edge, double* out_edge_weight, double* dis, int* queue, int* queue_size, int* visited, int* pre, int* mutex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < *queue_size) {
        int v = queue[idx];

        for (int i = out_pointer[v]; i < out_pointer[v + 1]; i++) {
            int new_v = out_edge[i];
            double weight = out_edge_weight[i];

            double new_w = dis[v] + weight;

            // try doing relaxation
            // mutex is used
            while (atomicCAS(&mutex[new_v], 0, 1) != 0);

            if (new_w < dis[new_v]) {
                dis[new_v] = new_w;
                // update the previous vertex
                pre[new_v] = v;
                atomicExch(&visited[new_v], 1);
                // share the updated distance with other threads in different blocks
                __threadfence();
            }

            atomicExch(&mutex[new_v], 0);
        }
    }
}

void gpu_sssp_pre(CSR_graph<double>& input_graph, int source, std::vector<double>& distance, std::vector<int>& pre_v, double max_dis) {
    // the only difference between this function and the previous one is that we need to record the previous vertex
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();

    double* dis;
    int* out_edge = input_graph.out_edge;
    double* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;
    int* visited;
    
    int* queue, * next_queue;
    int* queue_size, * next_queue_size;
    int* mutex;
    int* pre;

    cudaMallocManaged((void**)&dis, V * sizeof(double));
    cudaMallocManaged((void**)&visited, V * sizeof(int));
    cudaMallocManaged((void**)&queue, V * sizeof(int));
    cudaMallocManaged((void**)&next_queue, V * sizeof(int));
    cudaMallocManaged((void**)&queue_size, sizeof(int));
    cudaMallocManaged((void**)&next_queue_size, sizeof(int));

    cudaMallocManaged((void**)&mutex, V * sizeof(int));
    cudaMallocManaged((void**)&pre, V * sizeof(int));

    cudaDeviceSynchronize();
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    for (int i = 0; i < V; i++) {
		dis[i] = max_dis;
		visited[i] = 0;
        mutex[i] = 0;
        pre[i] = -1;
	}
    dis[source] = 0;


    *queue_size = 1;
    queue[0] = source;
    *next_queue_size = 0;

    int threadsPerBlock = 1024;
    int numBlocks = 0;

    while (*queue_size > 0) {
		numBlocks = (*queue_size + threadsPerBlock - 1) / threadsPerBlock;
		Relax_pre <<< numBlocks, threadsPerBlock >>> (out_pointer, out_edge, out_edge_weight, dis, queue, queue_size, visited, pre, mutex);
		cudaDeviceSynchronize();

        cudaError_t cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Relax kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

		numBlocks = (V + threadsPerBlock - 1) / threadsPerBlock;
		CompactQueue <<< numBlocks, threadsPerBlock >>> (V, next_queue, next_queue_size, visited);
		cudaDeviceSynchronize();

        cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess) {
			fprintf(stderr, "CompactQueue kernel launch failed: %s\n", cudaGetErrorString(cuda_status));
			return;
		}
		
        std::swap(queue, next_queue);

		*queue_size = *next_queue_size;
        *next_queue_size = 0;
	}

    cudaMemcpy(distance.data(), dis, V * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(pre_v.data(), pre, V * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dis);
    cudaFree(visited);
    cudaFree(queue);
    cudaFree(next_queue);
    cudaFree(queue_size);
    cudaFree(next_queue_size);
    cudaFree(mutex);
    cudaFree(pre);

    return;
}

std::vector<std::pair<std::string, double>> Cuda_SSSP_pre(graph_structure<double>& graph, CSR_graph<double>& csr_graph, std::string src_v, std::vector<int>& pre_v, double max_dis) {
    int src_v_id = graph.vertex_str_to_id[src_v];
    std::vector<double> gpuSSSPvec(graph.V, 0);
    pre_v.resize(graph.V);
    gpu_sssp_pre(csr_graph, src_v_id, gpuSSSPvec, pre_v, max_dis);

    // check the correctness of the previous vertex
    /*for (int i = 0; i < graph.V; i++) {
		double dis = gpuSSSPvec[i];
		int pre = pre_v[i];
        int now = i;
		double sum = 0;
		while (pre != -1) {
			bool ff = false;
			for (auto edge : graph.OUTs[pre]) {
				if (edge.first == now) {
					sum += edge.second;
                    now = pre;
                    pre = pre_v[pre];
					ff = true;
					break;
				}
			}
			if (!ff) {
				std::cout << "Not found!" << std::endl;
				break;
			}
		}
		if (fabs(sum - dis) > 1e-4) {
			std::cout << "Error: pre_v is wrong!" << std::endl;
            std::cout << "sum: " << sum << " dis: " << dis << std::endl;
		}
	}*/

    return graph.res_trans_id_val(gpuSSSPvec);
}

#endif