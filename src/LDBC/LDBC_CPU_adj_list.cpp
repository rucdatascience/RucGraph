#include <chrono>
#include <time.h>

#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_sssp_pre.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>
#include <LDBC/checker.hpp>
#include <LDBC/ldbc.hpp>

int main()
{
    ios::sync_with_stdio(false);
    std::cin.tie(0);
    std::cout.tie(0);

    //freopen("../input.txt", "r", stdin);

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

    std::vector<std::pair<std::string, std::string>> result_all;

    if (1) {
        if (graph.sup_bfs) {
            double cpu_bfs_time = 0;

            try{
                std::vector<std::pair<std::string, int>> cpu_bfs_result;
                begin = std::chrono::high_resolution_clock::now();
                cpu_bfs_result = CPU_Bfs(graph, graph.bfs_src_name);
                end = std::chrono::high_resolution_clock::now();
                cpu_bfs_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("CPU BFS cost time: %f s\n", cpu_bfs_time);

                if(Bfs_checker(graph, cpu_bfs_result, graph.base_path + "-BFS"))
                    result_all.push_back(std::make_pair("BFS", std::to_string(cpu_bfs_time)));
                else
                    result_all.push_back(std::make_pair("BFS", "wrong"));
            }
            catch(...) {
                result_all.push_back(std::make_pair("BFS", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("BFS", "N/A"));
    }

    if (1) {
        if (graph.sup_sssp) {
            double cpu_sssp_time = 0;

            try {
                //std::vector<int> pre_v;
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP(graph, graph.sssp_src_name);
                //std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP_pre(graph, graph.sssp_src_name, pre_v);
                end = std::chrono::high_resolution_clock::now();
                /*std::vector<std::pair<std::string, std::string>> path = path_query(graph, graph.sssp_src_name, "338", pre_v);
                for (auto p : path)
                    std::cout << p.first << " -> " << p.second << std::endl;*/
                cpu_sssp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("CPU SSSP cost time: %f s\n", cpu_sssp_time);

                if (SSSP_checker(graph, cpu_sssp_result, graph.base_path + "-SSSP"))
                    result_all.push_back(std::make_pair("SSSP", std::to_string(cpu_sssp_time)));
                else
                    result_all.push_back(std::make_pair("SSSP", "wrong"));
            }
            catch(...) {
                result_all.push_back(std::make_pair("SSSP", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("SSSP", "N/A"));
    }

    if (1) {
        if (graph.sup_wcc) {
            double cpu_wcc_time = 0;

            try {
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::pair<std::string, std::string>> cpu_wcc_result = CPU_WCC(graph);
                end = std::chrono::high_resolution_clock::now();
                cpu_wcc_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("CPU WCC cost time: %f s\n", cpu_wcc_time);

                if (WCC_checker(graph, cpu_wcc_result, graph.base_path + "-WCC"))
                    result_all.push_back(std::make_pair("WCC", std::to_string(cpu_wcc_time)));
                else
                    result_all.push_back(std::make_pair("WCC", "wrong"));
            }
            catch(...) {
                result_all.push_back(std::make_pair("WCC", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("WCC", "N/A"));
    }

    if (1) {
        if (graph.sup_pr) {
            double cpu_pr_time = 0;

            try {
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::pair<std::string, double>> cpu_pr_result = CPU_PR(graph, graph.pr_its, graph.pr_damping);
                end = std::chrono::high_resolution_clock::now();
                cpu_pr_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("CPU PageRank cost time: %f s\n", cpu_pr_time);

                if (PR_checker(graph, cpu_pr_result, graph.base_path + "-PR"))
                    result_all.push_back(std::make_pair("PageRank", std::to_string(cpu_pr_time)));
                else
                    result_all.push_back(std::make_pair("PageRank", "wrong"));
            }
            catch(...) {
                result_all.push_back(std::make_pair("PageRank", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("PageRank", "N/A"));
    }

    if (1) {
        if (graph.sup_cdlp) {
            double cpu_cdlp_time = 0;

            try {
                begin = std::chrono::high_resolution_clock::now();
                std::vector<std::pair<std::string, std::string>> cpu_cdlp_result = CPU_CDLP(graph, graph.cdlp_max_its);
                end = std::chrono::high_resolution_clock::now();
                cpu_cdlp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                printf("CPU Community Detection cost time: %f s\n", cpu_cdlp_time);

                if (CDLP_checker(graph, cpu_cdlp_result, graph.base_path + "-CDLP"))
                    result_all.push_back(std::make_pair("CommunityDetection", std::to_string(cpu_cdlp_time)));
                else
                    result_all.push_back(std::make_pair("CommunityDetection", "wrong"));
            }
            catch(...) {
                result_all.push_back(std::make_pair("CommunityDetection", "failed!"));
            }
        }
        else
            result_all.push_back(std::make_pair("CommunityDetection", "N/A"));
    }

    std::cout << "Result: " << std::endl;
    int res_size = result_all.size();
    for (int i = 0; i < res_size; i++) {
        std::cout << result_all[i].second;
        if (i != res_size - 1)
            std::cout << ",";
    }
    std::cout << std::endl;

    graph.save_to_CSV(result_all, "./result-cpu.csv");

    //freopen("/dev/tty", "r", stdin);

    return 0;
}
