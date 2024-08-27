#include <string>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/algorithm/CPU_BFS.hpp>
#include <CPU_adj_list/algorithm/CPU_connected_components.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>
#include <CPU_adj_list/algorithm/CPU_PageRank.hpp>
#include <CPU_adj_list/algorithm/CPU_Community_Detection.hpp>

int main()
{
    ios::sync_with_stdio(false);
    std::cin.tie(0), std::cout.tie(0);

    graph_structure<double> graph; // directed graph

    // Add vertices and edges
    graph.add_vertice("one");
    graph.add_vertice("two");
    graph.add_vertice("three");
    graph.add_vertice("four");
    graph.add_vertice("five");
    graph.add_vertice("R");

    graph.add_edge("one", "two", 0.8);
    graph.add_edge("two", "three", 1);
    graph.add_edge("two", "R", 1);
    graph.add_edge("two", "four", 0.1);
    graph.add_edge("R", "three", 1);
    graph.add_edge("one", "three", 1);
    graph.add_edge("one", "four", 1);
    graph.add_edge("four", "three", 1);
    graph.add_edge("four", "five", 1);

    // Remove a vertex
    graph.remove_vertice("R");

    // Add a vertex
    graph.add_vertice("six");

    // Remove an edge
    graph.remove_edge("two", "four");

    // Add an edge
    graph.add_edge("one", "six", 1);

    // BFS
    std::cout << "Running BFS..." << std::endl;
    std::vector<std::pair<std::string, int>> cpu_bfs_result = CPU_Bfs(graph, "one");
    std::cout << "BFS result: " << std::endl;
    for (auto& res : cpu_bfs_result)
        std::cout << res.first << " " << res.second << std::endl;

    // Connected Components
    std::cout << "Running Connected Components..." << std::endl;
    std::vector<std::pair<std::string, std::string>> cpu_connected_components_result = CPU_WCC(graph);
    std::cout << "Connected Components result: " << std::endl;
    for (auto& res : cpu_connected_components_result)
        std::cout << res.first << " " << res.second << std::endl;

    // SSSP
    std::cout << "Running SSSP..." << std::endl;
    std::vector<std::pair<std::string, double>> cpu_sssp_result = CPU_SSSP(graph, "one");
    std::cout << "SSSP result: " << std::endl;
    for (auto& res : cpu_sssp_result)
        std::cout << res.first << " " << res.second << std::endl;

    // PageRank
    std::cout << "Running PageRank..." << std::endl;
    std::vector<std::pair<std::string, double>> cpu_pagerank_result = CPU_PR(graph, 10, 0.85);
    std::cout << "PageRank result: " << std::endl;
    for (auto& res : cpu_pagerank_result)
        std::cout << res.first << " " << res.second << std::endl;

    // Community Detection
    std::cout << "Running Community Detection..." << std::endl;
    std::vector<std::pair<std::string, std::string>> cpu_community_detection_result = CPU_CDLP(graph, 10);
    std::cout << "Community Detection result: " << std::endl;
    for (auto& res : cpu_community_detection_result)
        std::cout << res.first << " " << res.second << std::endl;

    return 0;
}