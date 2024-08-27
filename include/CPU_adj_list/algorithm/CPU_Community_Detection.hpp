#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>
#include <vector>
#include <iostream>
#include <unordered_map>
#include <CPU_adj_list/ThreadPool.h>
#include <numeric>

// Community Detection Using Label Propagation
// call this function like:ans_cpu = CDLP(graph.INs, graph.OUTs, graph.vertex_id_to_str, graph.cdlp_max_its);
// Returns label of the graph based on the graph and number of iterations.
std::vector<std::string> CDLP(graph_structure<double>& graph, int iters)
{
    auto& in_edges = graph.INs; // incoming edges of each vertex in the graph
    auto& out_edges = graph.OUTs; // outgoing edges of each vertex in the graph

    int N = in_edges.size(); // number of vertices in the graph
    std::vector<int> label(N); // record the label of the vertex
    std::iota(std::begin(label), std::end(label), 0);
    std::vector<int> new_label(N); // temporarily stores the updated label
    
    ThreadPool pool_dynamic(100); 
    std::vector<std::future<int>> results_dynamic;

    for (int k = 0; k < iters; k++) // continue for a fixed number of iterations
    {
        for (int q = 0; q < 100; q++)
        {
            results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &in_edges, &out_edges, &label, &new_label]
                {
                    int start = (long long)q * N / 100, end = std::min((long long)N - 1, (long long)(q + 1) * N / 100);
                    for (int i = start; i <= end; i++) {

                        std::unordered_map<int, int> count; // record the label information of the neighbor vertex. the first keyword is the label and the second keyword is the number of occurrences
                        for (auto& x : in_edges[i]) // traverse the incoming edges of vertex i
                        {
                            count[label[x.first]]++; // count the number of label occurrences of the neighbor vertices
                        }
                        for (auto& x : out_edges[i]) // traverse the outcoming edges of vertex i
                        {
                            count[label[x.first]]++; // count the number of label occurrences of the neighbor vertices
                        }
                        int maxcount = 0; // the maximum number of maxlabel occurrences, the initial value is set to 0, which means that all existing labels can be recorded
                        int maxlabel = label[i]; // consider the possibility of isolated points, the initial label is label[i] instead of 0
                        for (std::pair<int, int> p : count) // traversal the label statistics protector of the neighbor node
                        {
                            if (p.second > maxcount) // the number of label occurrences currently traversed is greater than the recorded value
                            {
                                maxcount = p.second; // update the label
                                maxlabel = p.first;
                            }
                            else if (p.second == maxcount) // take a smaller value when the number of label occurrences is the same
                            {
                                maxlabel = std::min(p.first, maxlabel);
                            }
                        }
                        
                        new_label[i] = maxlabel; // record the maxlabel

                    }
                    return 1; }));
        }
        for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic); // clear results dynamic

        std::swap(new_label, label); // store labels of type string
    }

    std::vector<std::string>res(N);
    for (int i = 0; i < N; i++)
    {
        res[i] = graph.vertex_id_to_str[label[i]].first; // convert the label to string and store it in res
    }

    return res;
}

// Community Detection Using Label Propagation
// Returns label of the graph based on the graph and number of iterations.
// the type of the vertex and label are string
std::vector<std::pair<std::string, std::string>> CPU_CDLP(graph_structure<double>& graph, int iterations)
{
    std::vector<std::string> cdlpVec = CDLP(graph, iterations); // get the labels of each vertex. vector index is the id of vertex

    std::vector<std::pair<std::string, std::string>> res; // store results, the first value in pair records the vertex id, and the second value records the label
    int size = cdlpVec.size();
    for (int i = 0; i < size; i++)
        res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, cdlpVec[i])); // for each vertex, get its string number and store it in res

    return res; // return the results
}