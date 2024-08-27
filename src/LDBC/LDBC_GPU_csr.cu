#include <time.h>
#include <chrono>

#include <GPU_csr/algorithm/GPU_BFS.cuh>
#include <GPU_csr/algorithm/GPU_connected_components.cuh>
#include <GPU_csr/algorithm/WCC.cuh>
#include <CPU_adj_list/algorithm/CPU_sssp_pre.hpp>
#include <GPU_csr/algorithm/GPU_shortest_paths.cuh>
#include <GPU_csr/algorithm/GPU_sssp_pre.cuh>
#include <GPU_csr/algorithm/GPU_PageRank.cuh>
#include <GPU_csr/algorithm/GPU_Community_Detection.cuh>

#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);    

    //freopen("../input.txt", "r", stdin);

    std::vector<std::pair<std::string, std::string>> result_all;

    std::string directory;
    std::cout << "Please input the data directory: " << std::endl;
    std::cin >> directory;

    if (directory.back() != '/')
        directory += "/";

    std::string graph_name;
    std::cout << "Please input the graph name: " << std::endl;
    std::cin >> graph_name;

    std::string config_file_path = directory + graph_name + ".properties";

    LDBC<double> graph(directory, graph_name);
    graph.read_config(config_file_path); //Read the ldbc configuration file to obtain key parameter information in the file

    auto begin = std::chrono::high_resolution_clock::now();
    graph.load_graph(); //Read the vertex and edge files corresponding to the configuration file, // The vertex information in graph is converted to csr format for storage   
    auto end = std::chrono::high_resolution_clock::now();
    double load_ldbc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    printf("load_ldbc_time cost time: %f s\n", load_ldbc_time);

    begin = std::chrono::high_resolution_clock::now();
    CSR_graph<double> csr_graph = toCSR(graph);
    end = std::chrono::high_resolution_clock::now();
    double graph_to_csr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
    std::cout << "Number of vertices: " << csr_graph.OUTs_Neighbor_start_pointers.size()-1 << std::endl;
    std::cout << "Number of edges: " << csr_graph.OUTs_Edges.size() << std::endl;
    printf("graph_to_csr_time cost time: %f s\n", graph_to_csr_time);

    if (1) {
        if (graph.sup_bfs) {
            double gpu_bfs_time = 0;

            try {
                std::vector<std::pair<std::string, int>> bfs_result;
                begin = std::chrono::high_resolution_clock::now();
                bfs_result = Cuda_Bfs(graph, csr_graph, graph.bfs_src_name);
                end = std::chrono::high_resolution_clock::now();
                gpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU BFS cost time: %f s\n", gpu_bfs_time);
                
                if (Bfs_checker(graph, bfs_result, graph.base_path + "-BFS"))
                    result_all.push_back(std::make_pair("BFS", std::to_string(gpu_bfs_time)));
                else
                    result_all.push_back(std::make_pair("BFS", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("BFS", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("BFS", "N/A"));

    }

    if (1) {
        if (graph.sup_sssp) {
            double gpu_sssp_time = 0;

            try {
                std::vector<std::pair<std::string, double>> sssp_result;
                //std::vector<int> pre_v;
                begin = std::chrono::high_resolution_clock::now();
                sssp_result = Cuda_SSSP(graph, csr_graph, graph.sssp_src_name, std::numeric_limits<double>::max());
                //sssp_result = Cuda_SSSP_pre(graph, csr_graph, graph.sssp_src_name, pre_v, std::numeric_limits<double>::max());
                end = std::chrono::high_resolution_clock::now();
                gpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU SSSP cost time: %f s\n", gpu_sssp_time);
                if (SSSP_checker(graph, sssp_result, graph.base_path + "-SSSP"))
                    result_all.push_back(std::make_pair("SSSP", std::to_string(gpu_sssp_time)));
                else
                    result_all.push_back(std::make_pair("SSSP", "wrong"));
                
                /*std::vector<std::pair<std::string, std::string>> path = path_query(graph, graph.sssp_src_name, "338", pre_v);
                for (auto p : path) {
                    std::cout << p.first << "->" << p.second << std::endl;
                }*/
            }
            catch (...) {
                result_all.push_back(std::make_pair("SSSP", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("SSSP", "N/A"));
    }

    if (1) {
        if (graph.sup_wcc) {
            double gpu_wcc_time = 0;

            try {
                std::vector<std::pair<std::string, std::string>> wcc_result;
                begin = std::chrono::high_resolution_clock::now();
                wcc_result = Cuda_WCC_two(graph, csr_graph);
                end = std::chrono::high_resolution_clock::now();
                gpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU WCC cost time: %f s\n", gpu_wcc_time);
                if (WCC_checker(graph, wcc_result, graph.base_path + "-WCC"))
                    result_all.push_back(std::make_pair("WCC", std::to_string(gpu_wcc_time)));
                else
                    result_all.push_back(std::make_pair("WCC", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("WCC", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("WCC", "N/A"));
    }

    if (1) {
        if (graph.sup_pr) {
            double gpu_pr_time = 0;

            try {
                std::vector<std::pair<std::string, double>> pr_result;
                begin = std::chrono::high_resolution_clock::now();
                pr_result = Cuda_PR(graph, csr_graph, graph.pr_its, graph.pr_damping);
                end = std::chrono::high_resolution_clock::now();
                gpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU PageRank cost time: %f s\n", gpu_pr_time);
                if (PR_checker(graph, pr_result, graph.base_path + "-PR"))
                    result_all.push_back(std::make_pair("PR", std::to_string(gpu_pr_time)));
                else
                    result_all.push_back(std::make_pair("PR", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("PR", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("PR", "N/A"));
    }

    if (1) {
        if (graph.sup_cdlp) {
            double gpu_cdlp_time = 0;

            try {
                std::vector<std::pair<std::string, std::string>> cdlp_result;
                begin = std::chrono::high_resolution_clock::now();
                cdlp_result = Cuda_CDLP(graph, csr_graph, graph.cdlp_max_its);
                end = std::chrono::high_resolution_clock::now();
                gpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("GPU Community Detection cost time: %f s\n", gpu_cdlp_time);
                if (CDLP_checker(graph, cdlp_result, graph.base_path + "-CDLP"))
                    result_all.push_back(std::make_pair("CDLP", std::to_string(gpu_cdlp_time)));
                else
                    result_all.push_back(std::make_pair("CDLP", "wrong"));
            }
            catch (...) {
                result_all.push_back(std::make_pair("CDLP", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("CDLP", "N/A"));
    }

    std::cout << "Result: " << std::endl;
    int res_size = result_all.size();
    for (int i = 0; i < res_size; i++) {
        std::cout << result_all[i].second;
        if (i != res_size - 1)
            std::cout << ",";
    }
    std::cout << std::endl;

    graph.save_to_CSV(result_all, "./result-gpu.csv");

    //freopen("/dev/tty", "r", stdin);

    return 0;
}
