#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>

using namespace std;

__constant__ float tau = 20.0f, v_rest = -65.0f, v_reset = -65.0f, v_th = -50.0f, R = 1.0f, dt = 0.1f;
__constant__ int ref_steps = 20;

__global__ void update_neurons(float* v, int* ref, int* d_sensor_data, int* d_spike_counts, int sensors_per_region, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int r = i / sensors_per_region; // Identifikasi Wilayah
        int current_spike = d_sensor_data[r];

        if (ref[i] > 0) {
            ref[i]--;
            v[i] = v_reset;
        } else {
            float I_ext = current_spike * 50.0f;
            v[i] += (dt / tau) * (-(v[i] - v_rest) + R * I_ext);
        }

        if (v[i] >= v_th) {
            v[i] = v_reset;
            ref[i] = ref_steps;
            atomicAdd(&d_spike_counts[r], 1); // GPU Menjumlahkan Intensitas Paralel
        }
    }
}

int main() {
    const int NUM_REGIONS = 10;
    const int SENSORS_PER_REGION = 50000;
    const int N_SENSORS = NUM_REGIONS * SENSORS_PER_REGION;
    const int TIMESTEPS = 5000;

    string region_names[] = {"Sumatera   ", "Jawa Barat ", "Jawa Tengah", "Jawa Timur ", "Bali       ", 
                             "Nusa Tnggra", "Kalimantan ", "Sulawesi   ", "Maluku     ", "Papua      "};

    // Baca Data
    vector<vector<int>> h_sensor_data(TIMESTEPS, vector<int>(NUM_REGIONS));
    ifstream infile("data/seismic_data.txt");
    for (int t = 0; t < TIMESTEPS; ++t) {
        for (int r = 0; r < NUM_REGIONS; ++r) infile >> h_sensor_data[t][r];
    }

    float *d_v; int *d_ref, *d_sensor_data_t, *d_spike_counts;
    cudaMalloc(&d_v, N_SENSORS * sizeof(float));
    cudaMalloc(&d_ref, N_SENSORS * sizeof(int));
    cudaMalloc(&d_sensor_data_t, NUM_REGIONS * sizeof(int));
    cudaMalloc(&d_spike_counts, NUM_REGIONS * sizeof(int));

    vector<float> h_v(N_SENSORS, -65.0f);
    vector<int> h_ref(N_SENSORS, 0);
    int h_spike_counts[NUM_REGIONS] = {0};

    cudaMemcpy(d_v, h_v.data(), N_SENSORS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, h_ref.data(), N_SENSORS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_spike_counts, 0, NUM_REGIONS * sizeof(int));

    cout << "\033[2J\033[1;1H"; // Clear screen
    cout << "\033[1;36m========================================================\033[0m\n";
    cout << "\033[1;36m📡 SISTEM PERINGATAN DINI GEMPA BMKG (GPU CUDA) 📡\033[0m\n";
    cout << "\033[1;36m========================================================\033[0m\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N_SENSORS + threadsPerBlock - 1) / threadsPerBlock;

    cudaEventRecord(start);

    for (int t = 0; t < TIMESTEPS; ++t) {
        // Copy data sensor per timestep ke GPU
        cudaMemcpy(d_sensor_data_t, h_sensor_data[t].data(), NUM_REGIONS * sizeof(int), cudaMemcpyHostToDevice);
        
        update_neurons<<<blocksPerGrid, threadsPerBlock>>>(d_v, d_ref, d_sensor_data_t, d_spike_counts, SENSORS_PER_REGION, N_SENSORS);
        
        if ((t + 1) % 1000 == 0) {
            cudaMemcpy(h_spike_counts, d_spike_counts, NUM_REGIONS * sizeof(int), cudaMemcpyDeviceToHost);
            cout << "\n\033[1;33m[WAKTU SIMULASI: " << t + 1 << " ms] - ANALISIS REGIONAL\033[0m\n";
            for (int r = 0; r < NUM_REGIONS; ++r) {
                float magnitude = (h_spike_counts[r] / (float)SENSORS_PER_REGION) * 0.8f;
                cout << "[" << r+1 << (r==9?"": " ") << "] " << region_names[r] << " : ";
                if (magnitude > 5.0f) {
                    cout << "\033[1;31m🔴 AWAS GEMPA! (Mag: " << fixed << setprecision(1) << magnitude << " SR)\033[0m\n";
                } else {
                    cout << "\033[1;32m🟢 Aman        (Mag: " << fixed << setprecision(1) << magnitude << " SR)\033[0m\n";
                }
            }
            // Reset counter di GPU untuk detik selanjutnya
            cudaMemset(d_spike_counts, 0, NUM_REGIONS * sizeof(int));
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0; cudaEventElapsedTime(&milliseconds, start, stop);

    cout << "\n\033[1;36m--------------------------------------------------------\033[0m\n";
    cout << "⏱️  \033[1;37mWaktu Eksekusi GPU CUDA : \033[1;32m" << fixed << setprecision(4) << (milliseconds / 1000.0f) << " detik\033[0m\n";
    cout << "\033[1;36m--------------------------------------------------------\033[0m\n";

    cudaFree(d_v); cudaFree(d_ref); cudaFree(d_sensor_data_t); cudaFree(d_spike_counts);
    return 0;
}
