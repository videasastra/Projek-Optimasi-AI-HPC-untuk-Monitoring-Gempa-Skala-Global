import random

TIMESTEPS = 5000
REGIONS = 10

print("Membangun data sensor seismik untuk 10 Wilayah Indonesia...")
with open("data/seismic_data.txt", "w") as f:
    for t in range(TIMESTEPS):
        row_data = []
        for r in range(REGIONS):
            # Noise alami bumi (Getaran kecil)
            vibration = random.uniform(0.0, 0.4) 
            
            # Skenario: Gempa di Region 1 (Jawa Barat) pada ms 2000-3000
            if r == 1 and 2000 <= t < 3000:
                vibration = random.uniform(0.8, 1.0)
            
            # Skenario: Gempa di Region 7 (Sulawesi) pada ms 4000-5000
            if r == 7 and 4000 <= t < 5000:
                vibration = random.uniform(0.7, 0.95)
                
            spike = 1 if vibration > 0.5 else 0
            row_data.append(str(spike))
        
        f.write(" ".join(row_data) + "\n")

print("Data seismik 10 Wilayah berhasil dibuat!")
