#pragma once

#include <queue>
#include <climits>
#include <algorithm>
#include <stdexcept>

#include <CPU_adj_list/CPU_adj_list.hpp>
template <typename T> // T is float or double
std::vector<int> CPU_BFS(std::vector<std::vector<std::pair<int, T>>> &input_graph, int root = 0, int min_depth = 0, int max_depth = INT_MAX)
{
	/* The CPU code for breadth first search uses queues to traverse the graph and record depth,
	   which is also used to prevent duplicate traversal */
	int N = input_graph.size();

	std::vector<int> depth(N, INT_MAX);
	depth[root] = 0;
	std::vector<int> searched_vertices; // use to prevent duplicate traversal while recording depth

	std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
	Q.push(root);
	while (Q.size() > 0)
	{
		int v = Q.front();
		if (depth[v] >= min_depth && depth[v] <= max_depth)
			searched_vertices.push_back(v);

		Q.pop(); // Removing that vertex from queue,whose neighbour will be visited now

		if (depth[v] + 1 <= max_depth)
		{
			// Traversing node v in the graph yields a pair value, adjfirst being the adjacency point
			for (auto &adj : input_graph[v])
			{ /*processing all the neighbours of v*/
				if (depth[adj.first] > depth[v] + 1)
				{
					// If the depth of adjacent points is greater, add them to the queue. Otherwise, it means that the adjacent points have already been traversed before
					depth[adj.first] = depth[v] + 1;
					Q.push(adj.first);
				}
			}
		}
	}

	return depth;
}

std::vector<std::pair<std::string, int>> CPU_Bfs(graph_structure<double> &graph, std::string src_v, int min_depth = 0, int max_depth = INT_MAX)
{
	std::vector<int> depth = CPU_BFS(graph.OUTs, graph.vertex_str_to_id[src_v], min_depth, max_depth);
	return graph.res_trans_id_val(depth);
}