#pragma once
#include <CPU_adj_list/CPU_adj_list.hpp>

// the LDBC class records the graph operation parameters and graph structure
// defines functions for reading configuration files and graph structure files
template <typename weight_type>
class LDBC : public graph_structure<weight_type> {
    public:
	// class initializer
    LDBC() : graph_structure<weight_type>() {}
    LDBC(int n) : graph_structure<weight_type>(n) {}
	LDBC(std::string directory, std::string name) : graph_structure<weight_type>() {
		this->base_path = directory + name;
	}

	bool is_directed = true; // direct graph or undirect graph
	bool is_weight = false; // weight graph or no weight graph
	bool is_sssp_weight = true; // the weight of sssp

	bool sup_bfs = false; // records the graph operator to be computed
	bool sup_cdlp = false;
	bool sup_pr = false;
	bool sup_wcc = false;
	bool sup_sssp = false;
	std::string bfs_src_name; // get bfs vertex source
	std::string sssp_src_name; // get sssp vertex source
	std::string base_path; // graph structure file storage path
	int bfs_src = 0; // define bfs vertex source is 0
	int cdlp_max_its = 10; // cdlp algo max iterator num
	int pr_its = 10; // pr algo iterator num
	int sssp_src = 0; // define sssp vertex source is 0
	double pr_damping = 0.85; // pr algorithm damping coefficient

	void load_graph();
	void read_config(std::string config_path);

	void save_to_CSV(std::vector<std::pair<std::string, std::string>>& res, std::string file_path);
};

// read config file
// the specific information includes the number of edges, the number of vertices, the parameters of graph operators, etc
template <typename weight_type>
void LDBC<weight_type>::read_config(std::string config_path) {
	std::ifstream file(config_path);
    std::string line;

    if (!file.is_open()) { // unable to open file
        std::cerr << "Unable to open file: " << config_path << std::endl;
        return;
    }

	std::cout << "Reading config file..." << std::endl;

    while (getline(file, line)) { // read the config file line-by-line
		if (line.empty() || line[0] == '#') // invalid information, blank lines or comment lines
			continue;

		auto split_str = parse_string(line, " = "); // read configuration entries and their configuration values

		if (split_str.size() != 2) {
			std::cerr << "Invalid line: " << line << std::endl;
			continue;
		}

        auto key = split_str[0]; // configuration entry
		auto value = split_str[1]; // configuration value

        auto parts = parse_string(key, ".");
        if (parts.size() >= 2) {
			if (parts.back() == "vertex-file") // Reading *.properties file to get vertex file. eg. datagen-7_5-fb.v
				std::cout << "vertex_file: " << value << std::endl;
			else if (parts.back() == "edge-file") // Reading *.properties file to get edge file
				std::cout << "edge_file: " << value << std::endl;
			else if (parts.back() == "vertices") // Reading *.properties file to get the number of vertices
				std::cout << "V: " << value << std::endl;
			else if (parts.back() == "edges") // Reading *.properties file to get the number of edges
				std::cout << "E: " << value << std::endl;
			else if (parts.back() == "directed") { // Reading *.properties file to knows whether the graph is directed or undirected
				if (value == "false")
					this->is_directed = false;
				else
					this->is_directed = true;
				std::cout << "is_directed: " << this->is_directed << std::endl;
			}
			else if (parts.back() == "names") {//eg. graph.datagen-7_5-fb.edge-properties.names = weight
				if (value == "weight")
					this->is_weight = true;
				else
					this->is_weight = false;
				std::cout << "is_weight: " << this->is_weight << std::endl;
			}
			else if (parts.back() == "algorithms") { // gets the type of algorithm contained in the configuration file
				auto algorithms = parse_string(value, ", ");
				for (auto& algorithm : algorithms) {
					if (algorithm == "bfs")
						sup_bfs = true;
					else if (algorithm == "cdlp")
						sup_cdlp = true;
					else if (algorithm == "pr")
						sup_pr = true;
					else if (algorithm == "sssp")
						sup_sssp = true;
					else if (algorithm == "wcc")
						sup_wcc = true;
				}
				std::cout << "bfs: " << sup_bfs << std::endl;
				std::cout << "cdlp: " << sup_cdlp << std::endl;
				std::cout << "pr: " << sup_pr << std::endl;
				std::cout << "sssp: " << sup_sssp << std::endl;
				std::cout << "wcc: " << sup_wcc << std::endl;
			}
			else if (parts.back() == "cdlp-max-iterations") { // iteration parameters in Community Detection
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "pr-damping-factor") { // damping factor in PageRank
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "pr-num-iterations") { // number of iterations in PageRank
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "sssp-weight-property") { // weight property in sssp
				if (value == "weight")
					this->is_sssp_weight = true;
				else
					this->is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << this->is_sssp_weight << std::endl;
			}
			else if (parts.back() == "max-iterations") {
				cdlp_max_its = stoi(value);
				std::cout << "cdlp_max_its: " << cdlp_max_its << std::endl;
			}
			else if (parts.back() == "damping-factor") {
				pr_damping = stod(value);
				std::cout << "pr_damping: " << pr_damping << std::endl;
			}
			else if (parts.back() == "num-iterations") {
				pr_its = stoi(value);
				std::cout << "pr_its: " << pr_its << std::endl;
			}
			else if (parts.back() == "weight-property") {
				if (value == "weight")
					this->is_sssp_weight = true;
				else
					this->is_sssp_weight = false;
				std::cout << "is_sssp_weight: " << this->is_sssp_weight << std::endl;
			}
            else if (parts.back() == "source-vertex") {
				if (parts[parts.size() - 2] == "bfs") {
					bfs_src_name = value; // get bfs source vertex; eg. graph.datagen-7_5-fb.bfs.source-vertex = 6
					std::cout << "bfs_source_vertex: " << value << std::endl;
				}
				else {
					sssp_src_name = value; // get sssp source vertex; eg. graph.datagen-7_5-fb.sssp.source-vertex = 6
					std::cout << "sssp_source_vertex: " << value  << std::endl;
				}
            }
        }
    }
	std::cout << "Done." << std::endl; // read complete
    file.close();
}

// read the structure of the graph, including vertices and edges
template <typename weight_type>
void LDBC<weight_type>::load_graph() {

	std::string vertex_file_path;
	vertex_file_path = this->base_path + ".v"; // file with ".v" suffix stores vertices information

	std::cout << "Loading vertices..." << std::endl;
	std::string line_content;
	std::ifstream myfile(vertex_file_path); // open the vertex data file

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) // read data line by line
			this->add_vertice(line_content); // Parsed the read data
		myfile.close();
	}
	else { // Unable to open file
		std::cout << "Unable to open file " << vertex_file_path << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}

	std::cout << "Done." << std::endl;
	if (sup_bfs) {
		if (this->vertex_str_to_id.find(bfs_src_name) == this->vertex_str_to_id.end()) { // bfs_src_name from read_configure
			std::cout << "Invalid source vertex for BFS" << std::endl;
			getchar();
			exit(1);
		}
		else
			bfs_src = this->vertex_str_to_id[bfs_src_name];
	}
		
	if (sup_sssp) {
		if (this->vertex_str_to_id.find(sssp_src_name) == this->vertex_str_to_id.end()) { // sssp_src_name from read_configure
			std::cout << "Invalid source vertex for SSSP" << std::endl;
			getchar();
			exit(1);
		}
		else
			sssp_src = this->vertex_str_to_id[sssp_src_name];
	}

	std::string edge_file_path;
	edge_file_path = this->base_path + ".e"; // file with ".e" suffix stores edges information

	std::cout << "Loading edges..." << std::endl;
	myfile.open(edge_file_path); // open the edge data file

	if (myfile.is_open()) {
		while (getline(myfile, line_content)) { // read data line by line
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");
			if (Parsed_content.size() < 2) {
				std::cerr << "Invalid edge input!" << std::endl;
				continue;
			}
			weight_type ec = Parsed_content.size() > 2 ? std::stod(Parsed_content[2]) : 1; // get weight
			this->add_edge(Parsed_content[0], Parsed_content[1], ec); // add edge
			if (!is_directed) { // undirected graphs require additional opposite edges
				this->add_edge(Parsed_content[1], Parsed_content[0], ec);
			}
		}
		myfile.close();
	}
	else { // Unable to open file
		std::cout << "Unable to open file " << edge_file_path << std::endl
			<< "Please check the file location or file name." << std::endl;
		getchar();
		exit(1);
	}
	std::cout << "Done." << std::endl;
}

// save the results in csv format to the given path
template <typename weight_type>
void LDBC<weight_type>::save_to_CSV(std::vector<std::pair<std::string, std::string>>& res, std::string file_path) {
	std::ofstream out(file_path, std::ios::app);

	int res_size = res.size();
	for (int i = 0; i < res_size; i++) {
        out << res[i].second;
        if (i != res_size - 1)
            out << ",";
    }
    out << std::endl;

	out.close();
}
