#pragma once

#include <CPU_adj_list/CPU_adj_list.hpp>
#include <algorithm>
#include <cmath>
#include <limits.h>


bool compare(std::vector<int>& a, std::vector<int>& b) {
    return a[0] < b[0];
}

// checker for the bfs graph operator
// return check results(true or false) that based on graphs, results, and baseline.
bool Bfs_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, int>>& res, std::string base_line_file) {

    int size = res.size(); // get the result size

    if (size != graph.V) { // the result size does not match the graph size
        std::cout << "Size of BFS results is not equal to the number of vertices!" << std::endl;
        return false;
    }

    std::ifstream base_line(base_line_file); // read the baseline file

    if (!base_line.is_open()) { // failed to open the baseline file
        std::cout << "Baseline file not found!" << std::endl;
        return false;
    }

    std::vector<int> id_res(graph.V, -1);
    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second; // convert vertex id of string type to integer and stores in id-res

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) { // check each item in the baseline file
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) { // Baseline file format error
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return false;
        }
        if (id >= size) { // id >= size means that more files are read from baseline than in results 
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return false;
        }
        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) { // the vertex cannot be found in results
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return false;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        if (id_res[v_id] != std::stol(tokens[1])) { // the results are different from the baseline
            if (!(id_res[v_id] == INT_MAX && std::stol(tokens[1]) == LLONG_MAX)) { // make sure it's not different because of the maximum value
                std::cout << "Baseline file and GPU BFS results are not the same!" << std::endl;
                std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
                std::cout << "BFS result: " << graph.vertex_id_to_str[v_id].first << " " << id_res[v_id] << std::endl;
                base_line.close();
                return false;
            }
        }
        id++;
    }
    if (id != size) { // id != size means that more reults item than baseline 
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return false;
    }

    std::cout << "BFS results are correct!" << std::endl;
    base_line.close();
    return true; // BFS results are correct, return true
}

void set_root(std::vector<int>& parent, int v) {
    if (parent[v] == v)
        return;
    set_root(parent, parent[v]);
    parent[v] = parent[parent[v]];
}

// checker for the WCC graph operator
// return check results(true or false) that based on graphs, results, and baseline.
bool WCC_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, std::string>>& res, std::string base_line_file) {
    std::vector<std::vector<int>> temp; // record the connection components for each vertex in the results
    temp.resize(graph.V);
    for (auto &p : res)
        temp[graph.vertex_str_to_id[p.second]].push_back(graph.vertex_str_to_id[p.first]); // add the vertex to its appropriate Weakly Connected Components
    std::vector<std::vector<int>> components; // vector components[i] indicate that vertices in the i-th connection components in results
    for (int i = 0; i < graph.V; i++) {
        if (temp[i].size() > 0)
            components.push_back(temp[i]); // extract every Weakly Connected Components from temp
    }

    int size = components.size();
    for (auto &v : components) {
        if (!v.size()) {
            std::cout << "One of WCC results is empty!" << std::endl;
            return false;
        }
        std::sort(v.begin(), v.end()); // sort the vertices of the same connected components 
    }

    std::sort(components.begin(), components.end(), compare);

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) { // failed to open the baseline file
        std::cout << "Baseline file not found!" << std::endl;
        return false;
    }

    std::vector<std::vector<int>> base_res; // vector base_res[i] indicate that vertices in the i-th connection components in baseline
    std::vector<int> base_components; // record the connection components for each vertex in the baseline

    base_components.resize(graph.V, 0);

    std::string line;

    while (std::getline(base_line, line)) { // read the baseline line by line
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) { // Baseline file format error
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return false;
        }
        base_components[graph.vertex_str_to_id[tokens[0]]] = graph.vertex_str_to_id[tokens[1]]; // store baseline file per row value to component
    }

    for (int i = 0; i < graph.V; i++)
        set_root(base_components, i);

    std::vector<std::vector<int>> componentLists(graph.V);

    // the following operations are the same as the results operations, but work with baseline data
    for (int i = 0; i < graph.V; i++) {
        componentLists[base_components[i]].push_back(i);
    }

    for (int i = 0; i < graph.V; i++) {
		if (componentLists[i].size() > 0)
			base_res.push_back(componentLists[i]);
	}

    for (auto &v : base_res) {
        if (!v.size()) {
            std::cout << "One of baseline WCC results is empty!" << std::endl;
            base_line.close();
            return false;
        }
        std::sort(v.begin(), v.end());
    }

    std::sort(base_res.begin(), base_res.end(), compare);

    if (size != base_res.size()) {
        std::cout << "Baseline file and WCC results are not the same!" << std::endl;
        std::cout << "Baseline total component is " << base_res.size() << std::endl;
        std::cout << "WCC result total component is " << components.size() << std::endl;
        return false;
    }

    for (int i = 0; i < size; i++) { // compare each Weakly Connected Component
        if (base_res[i].size() != components[i].size()) { // different sizes mean different results and baseline
            std::cout << "Baseline file and WCC results are not the same!" << std::endl;
            std::cout << "Baseline component size is " << base_res[i].size() << std::endl;
            std::cout << "WCC result component size is " << components[i].size() << std::endl;
            return false;
        }
        for (int j = 0; j < base_res[i].size(); j++) {
            if (base_res[i][j] != components[i][j]) { // since both baseline and results are ordered, simply compare the elements in order
                std::cout << "Baseline file and WCC results are not the same!" << std::endl;
                std::cout << "Difference at: " << graph.vertex_id_to_str[base_res[i][j]].first << " " << graph.vertex_id_to_str[components[i][j]].first << std::endl;
                base_line.close();
                return false;
            }
        }
    }

    std::cout << "WCC results are correct!" << std::endl;
    base_line.close();
    return true; // WCC results are correct, return true
}

// checker for the SSSP graph operator
// return check results(true or false) that based on graphs, results, and baseline.
bool SSSP_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, double>>& res, std::string base_line_file) {
    
    int size = res.size(); // get the result size

    if (size != graph.V) { // the result size does not match the graph size
        std::cout << "Size of SSSP results is not equal to the number of vertices!" << std::endl;
        return false;
    }

    std::ifstream base_line(base_line_file); // read the baseline file

    if (!base_line.is_open()) { // failed to open the baseline file
        std::cout << "Baseline file not found!" << std::endl;
        return false;
    }

    std::vector<double> id_res(graph.V, INT_MAX);

    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second; // convert vertex id of string type to integer and stores in id-res

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) { // check each item in the baseline file
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) { // Baseline file format error
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return false;
        }
        if (id >= size) { // id >= size means that more files are read from baseline than in results 
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return false;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) { // the vertex cannot be found in results
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return false;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (tokens[1] == "infinity" || tokens[1] == "inf") { // "infinity" in baseline, so check wether the results is max
            if (id_res[v_id] != std::numeric_limits<double>::max()) {
                std::cout << "Baseline file and SSSP results are not the same!" << std::endl;
                std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
                std::cout << "SSSP result: " << graph.vertex_id_to_str[v_id].first << " " << id_res[v_id] << std::endl;
                base_line.close();
                return false;
            }
        }
        else if (fabs(id_res[v_id] - std::stod(tokens[1])) > 1e-4) { // set the error range to 1e-4, and answers within the range are considered to be correct
            std::cout << "Baseline file and SSSP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "SSSP result: " << graph.vertex_id_to_str[v_id].first << " " << id_res[v_id] << std::endl;
            base_line.close();
            return false;
        }
        id++;
    }
    if (id != size) { // id != size means that more reults item than baseline 
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return false;
    }

    std::cout << "SSSP results are correct!" << std::endl;
    base_line.close();
    return true; // SSSP results are correct, return true
}

// checker for the PageRank graph operator
// return check results(true or false) that based on graphs, results, and baseline.
bool PR_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, double>>& res, std::string base_line_file) {

    int size = res.size(); // get the result size

    std::vector<double> id_res(graph.V, 0);

    for (auto &p : res)
        id_res[graph.vertex_str_to_id[p.first]] = p.second; // convert vertex id of string type to integer and stores in id-res

    if (size != graph.V) { // the result size does not match the graph size
        std::cout << "Size of PageRank results is not equal to the number of vertices!" << std::endl;
        return false;
    }

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) { // failed to open the baseline file
        std::cout << "Baseline file not found!" << std::endl;
        return false;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) { // check each item in the baseline file
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) { // Baseline file format error
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return false;
        }
        if (id >= size) { // id >= size means that more files are read from baseline than in results 
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return false;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) { // the vertex cannot be found in results
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return false;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];

        if (fabs(id_res[v_id] - std::stod(tokens[1])) > 1e-2) { // set the error range to 1e-2, and answers within the range are considered to be correct
            std::cout << "Baseline file and PageRank results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "PageRank result: " << graph.vertex_id_to_str[v_id].first << " " << id_res[v_id] << std::endl;
            base_line.close();
            return false;
        }
        id++;
    }
    if (id != size) { // id != size means that more reults item than baseline 
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return false;
    }

    std::cout << "PageRank results are correct!" << std::endl;
    base_line.close();
    return true; // PageRank results are correct, return true
}

// checker for the PageRank graph operator
// return check results(true or false) that based on graphs, results, and baseline.
bool CDLP_checker(graph_structure<double>& graph, std::vector<std::pair<std::string, std::string>>& res, std::string base_line_file) {
    int size = res.size(); // get the result size

    std::vector<std::string> id_res;

    for (auto &p : res) 
        id_res.push_back(p.second); // store the results into id res in order

    if (size != graph.V) { // the result size does not match the graph size
        std::cout << "Size of CDLP results is not equal to the number of vertices!" << std::endl;
        return false;
    }

    std::ifstream base_line(base_line_file);

    if (!base_line.is_open()) { // failed to open the baseline file
        std::cout << "Baseline file not found!" << std::endl;
        return false;
    }

    int id = 0;
    std::string line;
    while (std::getline(base_line, line)) { // check each item in the baseline file
        std::vector<std::string> tokens;
        tokens = parse_string(line, " ");
        if (tokens.size() != 2) { // Baseline file format error
            std::cout << "Baseline file format error!" << std::endl;
            base_line.close();
            return false;
        }
        if (id >= size) { // id >= size means that more files are read from baseline than in results 
            std::cout << "Size of baseline file is larger than the result!" << std::endl;
            base_line.close();
            return false;
        }

        if (graph.vertex_str_to_id.find(tokens[0]) == graph.vertex_str_to_id.end()) { // the vertex cannot be found in results
            std::cout << "Baseline file contains a vertex that is not in the graph!" << std::endl;
            base_line.close();
            return false;
        }
        int v_id = graph.vertex_str_to_id[tokens[0]];
        
        if (id_res[v_id] != tokens[1]) { // the results are different from the baseline
            std::cout << "Baseline file and CDLP results are not the same!" << std::endl;
            std::cout << "Baseline file: " << tokens[0] << " " << tokens[1] << std::endl;
            std::cout << "CDLP result: " << id_res[v_id] << std::endl;
            base_line.close();
            return false;
        }
        id++;
    }
    if (id != size) { // id != size means that more reults item than baseline 
        std::cout << "Size of baseline file is smaller than the result!" << std::endl;
        base_line.close();
        return false;
    }

    std::cout << "CDLP results are correct!" << std::endl;
    base_line.close();
    return true; // CDLP results are correct, return true
}
