// #pragma once

// #include <CPU_adj_list/CPU_adj_list.hpp>
// #include <queue>
// #include <vector>

// template<typename T> // T is float or double
// std::vector<int> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph, std::vector<std::vector<std::pair<int, T>>>& output_graph) {
// 	//Using BFS method to find connectivity vectors starting from each node
// 	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
// 	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
// 	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

// 	std::vector<int> parent;

// 	/*time complexity: O(V)*/
// 	int N = input_graph.size();
// 	std::vector<bool> discovered(N, false);
// 	parent.resize(N);
// 	//Vector initialization
// 	for (int i = 0; i < N; i++) {

// 		if (discovered[i] == false) {
// 			//If the node has not yet been added to the connected component, search for the connected component starting from the node
// 			/*below is a depth first search; time complexity O(|V|+|E|)*/
// 			std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
// 			Q.push(i);
// 			parent[i] = i;
// 			discovered[i] = true;
// 			while (Q.size() > 0) {
// 				int v = Q.front();
// 				Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

// 				for (auto& x : input_graph[v]) {
// 					int adj_v = x.first;
// 					if (discovered[adj_v] == false) {
// 						Q.push(adj_v);
// 						parent[adj_v] = parent[v];
// 						discovered[adj_v] = true;
// 					}
// 				}
// 				for (auto& x : output_graph[v]) {
// 					int adj_v = x.first;
// 					if (discovered[adj_v] == false) {
// 						Q.push(adj_v);
// 						parent[adj_v] = parent[v];
// 						discovered[adj_v] = true;
// 					}
// 				}
// 			}
// 		}
// 	}
// 	return parent;
// }

// std::vector<std::pair<std::string, std::string>> CPU_WCC(graph_structure<double> & graph){
// 	std::vector<int> wccVec = CPU_connected_components(graph.OUTs, graph.INs);
// 	return graph.res_trans_id_id(wccVec);
// }

#pragma once

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <CPU_adj_list/ThreadPool.h>
#include <queue>
#include <vector>

template<typename T> // T is float or double
std::vector<int> CPU_connected_components(std::vector<std::vector<std::pair<int, T>>>& input_graph, std::vector<std::vector<std::pair<int, T>>>& output_graph) {
	//Using BFS method to find connectivity vectors starting from each node
	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

	/*time complexity: O(V)*/

	int N = input_graph.size();

	std::vector<int> component;
	component.resize(N);
	for (int u = 0; u < N; u++) {
		component[u] = u;
	}

	int change = true;
	while (change) {
		change = false;

		ThreadPool pool_dynamic(100); 
		std::vector<std::future<int>> results_dynamic;
		for (long long q = 0; q < 100; q++) {
			results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &change, &input_graph, &component]
				{
					int start = q * N / 100, end = std::min(N - 1, (int)((q + 1) * N / 100));
					for (int u = start; u <= end; u++) {
						for (auto& x : input_graph[u]) {
							int v = x.first;
							int comp_u = component[u];
							int comp_v = component[v];
							if (comp_u == comp_v) continue;
							int high_comp = comp_u > comp_v ? comp_u : comp_v;
							int low_comp = comp_u + (comp_v - high_comp);
							if (high_comp == component[high_comp]) {
								change = true;
								component[high_comp] = low_comp;
							}
						}
					}
				return 1; }));
		}
		for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);
		
		for (long long q = 0; q < 100; q++) {
			results_dynamic.emplace_back(pool_dynamic.enqueue([q, N, &component]
				{
					int start = q * N / 100, end = std::min(N - 1, (int)((q + 1) * N / 100));
					for (int u = start; u <= end; u++) {
						while (component[u] != component[component[u]]) {
							component[u] = component[component[u]];
						}
					}
				return 1; }));
		}
		for (auto&& result : results_dynamic)
        {
            result.get();
        }
        std::vector<std::future<int>>().swap(results_dynamic);
	}

	return component;
}

std::vector<std::pair<std::string, std::string>> CPU_WCC(graph_structure<double> & graph){
	std::vector<int> wccVec = CPU_connected_components(graph.OUTs, graph.INs);
	return graph.res_trans_id_id(wccVec);
}
