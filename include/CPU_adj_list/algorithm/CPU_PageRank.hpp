#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>
#include <vector>
#include <iostream>
#include <CPU_adj_list/ThreadPool.h>
#include <algorithm>

// PageRank Algorithm
// call this function like: ans_cpu = CDLP(graph.INs, graph.OUTs, damp, graph.cdlp_max_its);
// used to show the relevance and importance of vertices in the graph
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
std::vector<double> PageRank (std::vector<std::vector<std::pair<int, double>>>& in_edge,
    std::vector<std::vector<std::pair<int, double>>>& out_edge, double damp, int iters) {

    int N = in_edge.size(); // number of vertices in the graph

    std::vector<double> rank(N, 1 / N); // The initial pagerank of each vertex is 1/|V|
    std::vector<double> new_rank(N); // temporarily stores the updated pagerank

    double d = damp; // damping factor
    double teleport = (1 - damp) / N; // teleport mechanism

    std::vector<int> sink; // the set of sink vertices
    for (int i = 0; i < N; i++)
    {
        if (out_edge[i].size() == 0)
            sink.push_back(i); // record the sink vertices
    }

    for (int i = 0; i < iters; i++) { // continue for a fixed number of iterations
        double sink_sum = 0;
        for (int i = 0; i < sink.size(); i++) // If the out-degree of the vertex is zero, it is a sink node
        {
            sink_sum += rank[sink[i]]; // calculate the sinksum, which is the sum of the pagerank value of all sink vertices
        }

        double x = sink_sum * d / N + teleport; // sum of sinksum and teleport

        ThreadPool pool_dynamic(100);
        std::vector<std::future<int>> results_dynamic;
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &rank, &out_edge, &new_rank, &x]
                {
                    int start = (long long)q * N / 100, end = std::min((long long)N - 1, (long long)(q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {
                        rank[i] /= out_edge[i].size();
                        new_rank[i] = x; // record redistributed from sinks and teleport value
                    }

                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);

        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edge, &rank, &new_rank, &d]
                {
                    int start = (long long)q * N / 100, end = std::min((long long)N - 1, (long long)(q + 1) * N / 100);
                    for (int v = start; v <= end; v++) {
                        double tmp = 0; // sum the rank and then multiply damping to improve running efficiency
                        for (auto& y : in_edge[v]) {
                            tmp = tmp + rank[y.first]; // calculate the importance value for each vertex
                        }
                        new_rank[v] += d * tmp;
                    }
                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);


        rank.swap(new_rank); // store the updated pagerank in the rank
    }
    return rank; // return the pagerank of each vertex
}

// PageRank Algorithm
// return the pagerank of each vertex based on the graph, damping factor and number of iterations.
// the type of the vertex and pagerank are string
std::vector<std::pair<std::string, double>> CPU_PR (graph_structure<double>& graph, int iterations, double damping) {
    std::vector<double> prValueVec = PageRank(graph.INs, graph.OUTs, damping, iterations); // get the pagerank in double type
    return graph.res_trans_id_val(prValueVec);
}