#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <thread>
#include <numeric>
#include <algorithm>

struct Edge {
    int32_t from;
    int32_t to;
};

const int32_t NUM_THREADS = 16;
const int32_t MAX_ITER = 50;
const double EPSILON = 1e-5;

int32_t compute_degrees(const std::string& csv_path, std::vector<int32_t>& out_deg, std::vector<int32_t>& in_deg) {
    std::ifstream file(csv_path);
    std::string line;
    std::getline(file, line);

    int32_t max_v = 0;
    std::vector<Edge> temp_edges;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t comma = line.find(',');
        int32_t from = std::stoi(line.substr(0, comma));
        int32_t to = std::stoi(line.substr(comma + 1));

        max_v = std::max({ max_v, from, to });
        temp_edges.push_back({ from, to });
    }

    int32_t num_vertices = max_v + 1;
    out_deg.resize(num_vertices, 0);
    in_deg.resize(num_vertices, 0);

    for (const auto& edge : temp_edges) {
        out_deg[edge.from]++;
        in_deg[edge.to]++;
    }
    return num_vertices;
}

std::vector<int32_t> Base_Calculate_Boundaries(const std::vector<int32_t>& in_deg, int32_t num_vertices) {
    std::vector<int32_t> boundaries;
    boundaries.push_back(0);

    int64_t total_edges = std::accumulate(in_deg.begin(), in_deg.end(), 0LL);
    int64_t target_edges_per_batch = total_edges / NUM_THREADS;

    int64_t current_sum = 0;
    for (int32_t v = 0; v < num_vertices; ++v) {
        current_sum += in_deg[v];
        if (current_sum >= target_edges_per_batch && boundaries.size() < NUM_THREADS) {
            boundaries.push_back(v + 1);
            current_sum = 0;
        }
    }
    boundaries.push_back(num_vertices);
    return boundaries;
}

void shard_edges(const std::string& csv_path, const std::vector<int32_t>& boundaries, int32_t num_vertices) {
    std::vector<int32_t> vertex_to_batch(num_vertices);
    for (int32_t b = 0; b < NUM_THREADS; ++b) {
        for (int32_t v = boundaries[b]; v < boundaries[b + 1]; ++v) {
            vertex_to_batch[v] = b;
        }
    }

    std::vector<std::ofstream> batch_files(NUM_THREADS);
    for (int32_t i = 0; i < NUM_THREADS; ++i) {
        batch_files[i].open("batch_" + std::to_string(i) + ".bin", std::ios::binary);
    }

    std::ifstream file(csv_path);
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t comma = line.find(',');
        int32_t from = std::stoi(line.substr(0, comma));
        int32_t to = std::stoi(line.substr(comma + 1));

        int32_t batch_id = vertex_to_batch[to];
        batch_files[batch_id].write(reinterpret_cast<const char*>(&from), sizeof(int32_t));
        batch_files[batch_id].write(reinterpret_cast<const char*>(&to), sizeof(int32_t));
    }
}

void worker_process_batch(int32_t batch_id, int32_t start_v, const std::vector<double>& current_rank, const std::vector<int32_t>& out_deg,
    double ground_share, std::vector<double>& local_next) {
    std::fill(local_next.begin(), local_next.end(), ground_share);

    std::ifstream infile("batch_" + std::to_string(batch_id) + ".bin", std::ios::binary);
    int32_t from, to;

    while (infile.read(reinterpret_cast<char*>(&from), sizeof(int32_t))) {
        infile.read(reinterpret_cast<char*>(&to), sizeof(int32_t));
        local_next[to - start_v] += current_rank[from] / (out_deg[from] + 1);
    }
}

void run_leader_rank(int32_t num_vertices, const std::vector<int32_t>& out_deg, const std::vector<int32_t>& boundaries) {
    std::vector<double> current_rank(num_vertices, 1.0);
    double ground_rank = 0.0;

    std::vector<std::vector<double>> local_buffers(NUM_THREADS);
    for (int32_t i = 0; i < NUM_THREADS; ++i) {
        local_buffers[i].resize(boundaries[i + 1] - boundaries[i]);
    }

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double ground_share = ground_rank / num_vertices;
        std::vector<std::thread> threads;

        for (int32_t i = 0; i < NUM_THREADS; ++i) {
            threads.emplace_back(worker_process_batch, i, boundaries[i],
                std::cref(current_rank), std::cref(out_deg),
                ground_share, std::ref(local_buffers[i]));
        }

        double next_ground_rank = 0.0;
        for (int32_t v = 0; v < num_vertices; ++v) {
            next_ground_rank += current_rank[v] / (out_deg[v] + 1);
        }

        for (auto& t : threads) t.join();

        double delta = 0.0;
        int32_t v_global = 0;
        for (int32_t i = 0; i < NUM_THREADS; ++i) {
            for (double val : local_buffers[i]) {
                delta += std::abs(val - current_rank[v_global]);
                current_rank[v_global] = val;
                v_global++;
            }
        }
        ground_rank = next_ground_rank;

        std::cout << "Iteration " << iter << " | Delta: " << delta << "\n";
        if (delta < EPSILON) break;
    }

    for (int32_t v = 0; v < num_vertices; ++v) {
        current_rank[v] += ground_rank / num_vertices;
    }
}

int main() {
    std::string csv_path = "edges.csv";

    std::vector<int32_t> out_deg, in_deg;
    int32_t num_vertices = compute_degrees(csv_path, out_deg, in_deg);

    std::vector<int32_t> boundaries = Base_Calculate_Boundaries(in_deg, num_vertices);

    shard_edges(csv_path, boundaries, num_vertices);

    run_leader_rank(num_vertices, out_deg, boundaries);

    return 0;
}
