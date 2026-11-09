#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;

const float tau = 20.0f, v_rest = -65.0f, v_reset = -65.0f, v_th = -50.0f, R = 1.0f, dt = 0.1f;
const int ref_steps = 20;
const int NUM_REGIONS = 10;
const int SENSORS_PER_REGION = 50000;
const int N_SENSORS = NUM_REGIONS * SENSORS_PER_REGION;
const int TIMESTEPS = 5000;

string region_names[] = {"Sumatera   ", "Jawa Barat ", "Jawa Tengah", "Jawa Timur ", "Bali       ", 
                         "Nusa Tnggra", "Kalimantan ", "Sulawesi   ", "Maluku     ", "Papua      "};

int main() {
    vector<vector<int>> sensor_data(TIMESTEPS, vector<int>(NUM_REGIONS));
    ifstream infile("data/seismic_data.txt");
    for (int t = 0; t < TIMESTEPS; ++t) {
        for (int r = 0; r < NUM_REGIONS; ++r) infile >> sensor_data[t][r];
    }
    
    vector<float> v(N_SENSORS, v_rest);
    vector<int> ref(N_SENSORS, 0);
    vector<int> region_spikes(NUM_REGIONS, 0);

    cout << "\033[2J\033[1;1H"; // Clear screen
    cout << "\033[1;36m========================================================\033[0m\n";
    cout << "\033[1;36m📡 SISTEM PERINGATAN DINI GEMPA BMKG (CPU/OpenMP) 📡\033[0m\n";
    cout << "\033[1;36m========================================================\033[0m\n";

    auto start_time = chrono::high_resolution_clock::now();

    for (int t = 0; t < TIMESTEPS; ++t) {
        #pragma omp parallel for
        for (int i = 0; i < N_SENSORS; ++i) {
            int r = i / SENSORS_PER_REGION; // Menentukan sensor ini masuk wilayah mana
            int current_spike = sensor_data[t][r]; 

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
                #pragma omp atomic
                region_spikes[r]++;
            }
        }

        // Tampilkan Dashboard setiap 1000 ms
        if ((t + 1) % 1000 == 0) {
            cout << "\n\033[1;33m[WAKTU SIMULASI: " << t + 1 << " ms] - ANALISIS REGIONAL\033[0m\n";
            for (int r = 0; r < NUM_REGIONS; ++r) {
                // Rumus dummy mengubah jumlah spike SNN menjadi Skala Richter (Magnitudo)
                float magnitude = (region_spikes[r] / (float)SENSORS_PER_REGION) * 0.8f;
                
                cout << "[" << r+1 << (r==9?"": " ") << "] " << region_names[r] << " : ";
                if (magnitude > 5.0f) {
                    cout << "\033[1;31m🔴 AWAS GEMPA! (Mag: " << fixed << setprecision(1) << magnitude << " SR)\033[0m\n";
                } else {
                    cout << "\033[1;32m🟢 Aman        (Mag: " << fixed << setprecision(1) << magnitude << " SR)\033[0m\n";
                }
                region_spikes[r] = 0; // Reset hitungan untuk 1000 ms berikutnya
            }
        }
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end_time - start_time;
    cout << "\n\033[1;36m--------------------------------------------------------\033[0m\n";
    cout << "⏱️  \033[1;37mWaktu Eksekusi CPU/OpenMP : \033[1;31m" << fixed << setprecision(4) << diff.count() << " detik\033[0m\n";
    cout << "\033[1;36m--------------------------------------------------------\033[0m\n";

    return 0;
}
