#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <queue>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <algorithm>

#include <CPU_adj_list/parse_string.hpp>
#include <CPU_adj_list/sorted_vector_binary_operations.hpp>
#include <CPU_adj_list/binary_save_read_vector_of_vectors.hpp>

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure
{
	//The data structure used on the CPU provides common operations such as adding nodes and edges on the graph

/* 	Special note that the actual labels of nodes on the graph may be unordered.
	 In the end, we provide some functions for converting from continuous labels (used in operators) to actual labels (original data) */
public:
	int V = 0;		 // the number of vertices
	long long E = 0; // the number of edges

	// OUTs[u] = v means there is an edge starting from u to v
	std::vector<std::vector<std::pair<int, weight_type>>> OUTs;
	// INs is transpose of OUTs. INs[u] = v means there is an edge starting from v to u
	std::vector<std::vector<std::pair<int, weight_type>>> INs;

	/*constructors*/
	graph_structure() {}
	graph_structure(int n)
	{
		V = n;
		OUTs.resize(n); // initialize n vertices
		INs.resize(n);
	}
	int size()
	{
		return V;
	}

	/*class member functions*/
	inline void add_edge(int, int, weight_type); // this function can change edge weights

	inline void remove_edge(int, int);//Remove any edge that connects two vertices
	inline void remove_edge(std::string, std::string);//Remove any edge that connects two vertices
	inline void remove_all_adjacent_edges(int);//Remove all edges, the input params is vertex numbers

	inline bool contain_edge(int, int); // whether there is an edge
	inline weight_type edge_weight(int, int); //get edge weight
	inline long long int edge_number(); // the total number of edges

	inline void print();//print graph
	inline void clear();// clear graph
	inline int out_degree(int);//get graph out degree
	inline int in_degree(int);//get graph in degree

	std::unordered_map<std::string, int> vertex_str_to_id; // vertex_str_to_id[vertex_name] = vertex_id
	std::vector<std::pair<std::string, bool>> vertex_id_to_str;	// vertex_id_to_str[vertex_id].first = vertex_name, vertex_id_to_str[vertex_id].second = whether the vertex is valid

	std::queue<int> invalid_vertex_id; // store the invalid vertex id

	int add_vertice(std::string); // Read the vertex information as a string
	void remove_vertice(std::string); // Remove the vertex information as a string
	void add_edge(std::string, std::string, weight_type);

	template <typename T>
	std::vector<std::pair<std::string, T>> res_trans_id_val(std::vector<T> &res);
	std::vector<std::pair<std::string, std::string>> res_trans_id_id(std::vector<int> &wcc_res);
};

/*class member functions*/

template <typename weight_type>
int graph_structure<weight_type>::add_vertice(std::string vertex)
{
	if (vertex_str_to_id.find(vertex) == vertex_str_to_id.end())
	{
		if (invalid_vertex_id.empty()) {
			vertex_id_to_str.push_back(std::make_pair(vertex, true));
			vertex_str_to_id[vertex] = V++;
			std::vector<std::pair<int, weight_type>> x;
			OUTs.push_back(x);
			INs.push_back(x);
		}
		else {
			int v = invalid_vertex_id.front();
			invalid_vertex_id.pop();
			vertex_id_to_str[v].first = vertex;
			vertex_id_to_str[v].second = true;
			vertex_str_to_id[vertex] = v;

			std::cout << "Recover vertex id " << v << std::endl;
		}
	}
	return vertex_str_to_id[vertex];
}

template <typename weight_type>
void graph_structure<weight_type>::remove_vertice(std::string vertex) {
	// if the vertex is not exist, return
	if (vertex_str_to_id.find(vertex) == vertex_str_to_id.end()) {
		std::cerr << "vertex " << vertex << " not exist!" << std::endl;
		return;
	}
	int v = vertex_str_to_id[vertex];
	remove_all_adjacent_edges(v);
	vertex_str_to_id.erase(vertex);
	vertex_id_to_str[v].second = false;
	invalid_vertex_id.push(v);
}

template <typename weight_type>
void graph_structure<weight_type>::add_edge(int e1, int e2, weight_type ec)
{
	sorted_vector_binary_operations_insert(OUTs[e1], e2, ec);
	sorted_vector_binary_operations_insert(INs[e2], e1, ec);
}

template <typename weight_type>
void graph_structure<weight_type>::add_edge(std::string e1, std::string e2, weight_type ec)
{
	E++;
	int v1 = add_vertice(e1);
	int v2 = add_vertice(e2);
	add_edge(v1, v2, ec);
}

template <typename weight_type>
void graph_structure<weight_type>::remove_edge(int e1, int e2)
{
	sorted_vector_binary_operations_erase(OUTs[e1], e2);
	sorted_vector_binary_operations_erase(INs[e2], e1);
}

template <typename weight_type>
void graph_structure<weight_type>::remove_edge(std::string e1, std::string e2)
{
	if (vertex_str_to_id.find(e1) == vertex_str_to_id.end()) {
		std::cerr << "vertex " << e1 << " not exist!" << std::endl;
		return;
	}
	if (vertex_str_to_id.find(e2) == vertex_str_to_id.end()) {
		std::cerr << "vertex " << e2 << " not exist!" << std::endl;
		return;
	}
	int v1 = vertex_str_to_id[e1];
	int v2 = vertex_str_to_id[e2];
	remove_edge(v1, v2);
}

template <typename weight_type>
void graph_structure<weight_type>::remove_all_adjacent_edges(int v)
{
	for (auto it = OUTs[v].begin(); it != OUTs[v].end(); it++)
		sorted_vector_binary_operations_erase(INs[it->first], v);

	for (auto it = INs[v].begin(); it != INs[v].end(); it++)
		sorted_vector_binary_operations_erase(OUTs[it->first], v);

	std::vector<std::pair<int, weight_type>>().swap(OUTs[v]);
	std::vector<std::pair<int, weight_type>>().swap(INs[v]);
}

template <typename weight_type>
bool graph_structure<weight_type>::contain_edge(int e1, int e2)
{
	return sorted_vector_binary_operations_search(OUTs[e1], e2);
}
template <typename weight_type>
weight_type graph_structure<weight_type>::edge_weight(int e1, int e2)
{
	return sorted_vector_binary_operations_search_weight(OUTs[e1], e2);
}
template <typename weight_type>
long long int graph_structure<weight_type>::edge_number()
{
	long long int num = 0;
	for (auto it : OUTs)
		num = num + it.size();

	return num;
}
template <typename weight_type>
void graph_structure<weight_type>::clear()
{
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(OUTs);
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(INs);
}
template <typename weight_type>
int graph_structure<weight_type>::out_degree(int v)
{
	return OUTs[v].size();
}
template <typename weight_type>
int graph_structure<weight_type>::in_degree(int v)
{
	return INs[v].size();
}
template <typename weight_type>
void graph_structure<weight_type>::print()
{

	std::cout << "graph_structure_print:" << std::endl;

	for (int i = 0; i < V; i++)
	{
		std::cout << "Vertex " << i << " OUTs List: ";
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++)
		{
			std::cout << "<" << OUTs[i][j].first << "," << OUTs[i][j].second << "> ";
		}
		std::cout << std::endl;
	}
	std::cout << "graph_structure_print END" << std::endl;
}

template <typename weight_type>
template <typename T>
std::vector<std::pair<std::string, T>> graph_structure<weight_type>::res_trans_id_val(std::vector<T> &res)
{
	std::vector<std::pair<std::string, T>> res_str;
	int res_size = res.size();
	for (int i = 0; i < res_size; i++) {
		if (vertex_id_to_str[i].second)
			res_str.push_back(std::make_pair(vertex_id_to_str[i].first, res[i]));
	}

	return res_str;
}

template <typename weight_type>
std::vector<std::pair<std::string, std::string>> graph_structure<weight_type>::res_trans_id_id(std::vector<int> &wcc_res)
{
	std::vector<std::pair<std::string, std::string>> res_str;
	int res_size = wcc_res.size();
	for (int i = 0; i < res_size; i++) {
		if (vertex_id_to_str[i].second) {
			if (vertex_id_to_str[wcc_res[i]].second)
				res_str.push_back(std::make_pair(vertex_id_to_str[i].first, vertex_id_to_str[wcc_res[i]].first));
			else
				std::cerr << "vertex " << vertex_id_to_str[wcc_res[i]].first << " not exist!" << std::endl;
		}	
	}

	return res_str;
}
