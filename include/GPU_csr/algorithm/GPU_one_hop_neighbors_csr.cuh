#ifndef GPU_one_hop_neighbors
#define GPU_one_hop_neighbors

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <GPU_csr/GPU_csr.hpp>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <vector>

// Define block calculation macro
#define MAX_BLOCKS_NUM 1024
#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) min(MAX_BLOCKS_NUM, (CALC_SIZE - 1) / ITEMS_PER_BLOCK + 1)

__global__ void compute_degrees_kernel (const int *row_offsets, const int *query_v, int *deg, int num_sources)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_sources) {
        int v = query_v[tid];
        deg[tid] = row_offsets[v + 1] - row_offsets[v];
    }
}

__device__ int find_owner (const int* offsets, int edge_idx, int num_queries)
{
    int low = 0, high = num_queries - 1;
    int ans = 0;
    while (low <= high) {
        int mid = (low + high) >> 1;
        if (offsets[mid] <= edge_idx) {
            ans = mid;
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return ans;
}

__global__ void get_neighbors_warp_kernel (const int *row_offsets, const int *all_edges, const int *query_v, int num_sources, const int *deg_offsets, int total_edges, int *neighbors) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    int q_idx = find_owner (deg_offsets, tid, num_sources);

    int u = query_v[q_idx];
    int offset_u = tid - deg_offsets[q_idx];
    
    neighbors[tid] = all_edges[row_offsets[u] + offset_u];
}

std::vector<std::vector<int>> cuda_one_hop_neighbors (CSR_graph<double> &input_graph, std::vector<int> &sources)
{
    // init
    int num_sources = sources.size();
    std::vector<std::vector<int>> result(num_sources);

    if (num_sources == 0) {
        return result;
    }

    int *row_offsets = input_graph.out_pointer, *all_edges = input_graph.out_edge;
    int *query_v, *deg, *deg_offsets, *neighbors;
    
    // Allocate GPU memory
    cudaMalloc(&deg, num_sources * sizeof(int));
    cudaMalloc(&deg_offsets, (num_sources + 1) * sizeof(int));
    cudaMalloc(&query_v, num_sources * sizeof(int));
    
    // Copy source vertices to GPU
    cudaMemcpy(query_v, sources.data(), num_sources * sizeof(int), cudaMemcpyHostToDevice);

    int THREADS_NUM = 256;
    int BLOCKS_NUM = (num_sources + THREADS_NUM - 1) / THREADS_NUM;

    // Compute degrees of source vertices
    compute_degrees_kernel <<<BLOCKS_NUM, THREADS_NUM>>> (row_offsets, query_v, deg, num_sources);
    cudaDeviceSynchronize();

    // Compute exclusive scan to get offsets
    thrust::device_ptr<int> dev_degrees(deg);
    thrust::device_ptr<int> dev_offsets(deg_offsets);
    thrust::exclusive_scan(dev_degrees, dev_degrees + num_sources, dev_offsets);
    cudaDeviceSynchronize();
    
    // Calculate total edges to fetch
    int h_last_degree = 0;
    cudaMemcpy(&h_last_degree, deg + (num_sources - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int h_last_offset = 0;
    cudaMemcpy(&h_last_offset, deg_offsets + (num_sources - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int total_edges_to_fetch = h_last_offset + h_last_degree;

    if (total_edges_to_fetch > 0) {
        // Allocate memory for neighbors
        cudaMalloc(&neighbors, total_edges_to_fetch * sizeof(int));
        
        // Get one-hop neighbors
        BLOCKS_NUM = (total_edges_to_fetch + THREADS_NUM - 1) / THREADS_NUM;
        get_neighbors_warp_kernel <<<BLOCKS_NUM, THREADS_NUM>>> (row_offsets, all_edges, query_v, num_sources, deg_offsets, total_edges_to_fetch, neighbors);
        cudaDeviceSynchronize();
        
        // Copy degrees back to host
        std::vector<int> degrees(num_sources);
        cudaMemcpy(degrees.data(), deg, num_sources * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy offsets back to host
        std::vector<int> offsets(num_sources + 1);
        cudaMemcpy(offsets.data(), deg_offsets, (num_sources + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Copy neighbors back to host
        std::vector<int> all_neighbors(total_edges_to_fetch);
        cudaMemcpy(all_neighbors.data(), neighbors, total_edges_to_fetch * sizeof(int), cudaMemcpyDeviceToHost);
        
        // Reconstruct the result
        for (int i = 0; i < num_sources; i++) {
            int start = offsets[i];
            int end = offsets[i] + degrees[i];
            result[i] = std::vector<int>(all_neighbors.begin() + start, all_neighbors.begin() + end);
        }
        
        // Free neighbors memory
        cudaFree(neighbors);
    }

    // Free GPU memory
    cudaFree(deg);
    cudaFree(deg_offsets);
    cudaFree(query_v);
    
    return result;
}

std::vector<std::vector<std::string>> Cuda_one_hop_neighbors (graph_structure<double> &graph, CSR_graph<double> &csr_graph, std::vector<std::string> &src_v)
{
    std::vector<int> src_v_id(src_v.size());
    for (int i = 0; i < src_v.size(); i++) {
        src_v_id[i] = graph.vertex_str_to_id[src_v[i]];
    }
    
    std::vector<std::vector<int>> gpu_one_hopNeighbors = cuda_one_hop_neighbors(csr_graph, src_v_id);
    std::vector<std::vector<std::string>> result(src_v.size());
    
    for (int i = 0; i < src_v.size(); i++) {
        // Convert each neighbor ID to string
        std::vector<std::string> neighbors;
        for (int neighbor_id : gpu_one_hopNeighbors[i]) {
            if (neighbor_id < graph.vertex_id_to_str.size() && graph.vertex_id_to_str[neighbor_id].second) {
                neighbors.push_back(graph.vertex_id_to_str[neighbor_id].first);
            }
        }
        result[i] = neighbors;
    }
    
    return result;
}

#endif