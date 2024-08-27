#pragma once

#include <vector>
#include <numeric>
#include <iostream>
#include <queue>
#include <algorithm>

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/algorithm/CPU_shortest_paths.hpp>

std::vector<double> CPU_shortest_paths_pre(std::vector<std::vector<std::pair<int, double>>>& input_graph, int source, std::vector<int>& pre_v) {
	//dijkstras-shortest-path-algorithm
	
	double inf = std::numeric_limits<double>::max();

	int N = input_graph.size();

	std::vector<double> distances;
	distances.resize(N, inf); // initial distance from source is inf

	if (source < 0 || source >= N) {
		std::cout << "Invalid source vertex" << std::endl;//Abnormal input judgment
		return distances;
	}

	distances[source] = 0;//Starting distance is 0
	std::vector<int> vis(N, 0);

	std::priority_queue<node, vector<node>, greater<node> > Q;//Using Heap Optimization Algorithm
	Q.push({0, source});

	while (Q.size() > 0) {

		int u = Q.top().u;

		Q.pop();//remove vertex visited this round

		if (vis[u]) continue;//if vertex has already been visited,it shouldn't be pushed to queue again.
		vis[u] = 1;//mark

		for (auto edge : input_graph[u]) {
			//Traverse all adjacent vertexs of a vertex
			int v = edge.first;//vertex pointed by edge
			double w = edge.second;//weight of edge
			//use v to update path cost
			if (distances[v] > distances[u] + w) {
				//If the path cost is smaller, update the new path cost
				distances[v] = distances[u] + w;
				pre_v[v] = u;
				Q.push({distances[v], v});//add new vertex to queue
			}
		}

	}

	return distances;
}

std::vector<std::pair<std::string, double>> CPU_SSSP_pre(graph_structure<double>& graph, std::string src_v, std::vector<int>& pre_v) {
	pre_v.resize(graph.V, -1);
	std::vector<double> ssspVec = CPU_shortest_paths_pre(graph.OUTs, graph.vertex_str_to_id[src_v], pre_v);

	// check the pre_v
	/*for (int i = 0; i < graph.V; i++) {
		double dis = ssspVec[i];
		int pre = pre_v[i];
        int now = i;
		double sum = 0;
		while (now != graph.vertex_str_to_id[src_v]) {
			bool ff = false;
			//std::cout << "pre: " << pre << " now: " << now << std::endl;
			for (auto edge : graph.OUTs[pre]) {
				//std::cout << "there is an edge from " << pre << " to " << edge.first << " with weight " << edge.second << std::endl;
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
		if (fabs(dis - sum) > 1e-4) {
			std::cout << "Error: pre_v is wrong!" << std::endl;
			std::cout << "dis: " << dis << " sum: " << sum << std::endl;
		}
	}*/

	return graph.res_trans_id_val(ssspVec);
}

std::vector<std::pair<std::string, std::string>> path_query(graph_structure<double>& graph, std::string src_v, std::string dst_v, std::vector<int>& pre_v) {
    int dst_id = graph.vertex_str_to_id[dst_v];
    int src_id = graph.vertex_str_to_id[src_v];

    std::vector<std::pair<std::string, std::string>> path;

    if (src_id < 0 || src_id >= graph.V || dst_id < 0 || dst_id >= graph.V) {
        std::cout << "Invalid source or destination vertex" << std::endl;
        return path;
    }

    if (pre_v[dst_id] == -1) {
        std::cout << "No path from " << src_v << " to " << dst_v << std::endl;
        return path;
    }

    int now = dst_id;
    while (now != src_id) {
        path.push_back(std::make_pair(graph.vertex_id_to_str[pre_v[now]].first, graph.vertex_id_to_str[now].first));
        if (pre_v[now] == -1 && now != src_id) {
            std::cout << "Error: pre_v is wrong!" << std::endl;
            break;
        }
        now = pre_v[now];
    }

    std::reverse(path.begin(), path.end());

    return path;
}