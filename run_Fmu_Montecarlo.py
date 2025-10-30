import numpy as np
import matplotlib.pyplot as plt
import tempfile
import shutil
import pickle
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import os

# ==============================
# CONFIGURAZIONE
# ==============================
FMU_FILENAME = 'GreenRoofSim.fmu'
N_RUNS = 100                  # Numero di simulazioni Monte Carlo
T_STOP = 3600.0               # Durata simulazione [s]
STEP_SIZE = 10.0              # Passo simulazione [s]
SEED = 42                     # Random seed per riproducibilità

GMM_DIR = 'data\pkl_models'        # cartella dove hai salvato i modelli GMM
np.random.seed(SEED)

# ==============================
# FUNZIONE: esegui una simulazione FMU
# ==============================
def run_fmu_with_noise(SolarRadiation, AmbientTemp, WindSpeed):
    unzipdir = tempfile.mkdtemp()
    model_description = read_model_description(FMU_FILENAME)
    extract(FMU_FILENAME, unzipdir)

    # Crea mappa delle variabili
    vrs = {v.name: v.valueReference for v in model_description.modelVariables}

    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=unzipdir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName='instance1'
    )

    fmu.instantiate()
    fmu.setupExperiment(startTime=0.0)
    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    time = 0.0
    times = []
    temps = []

    while time <= T_STOP:
        # Imposta gli input
        fmu.setReal([vrs['SolarRadiation']], [SolarRadiation[int(time/STEP_SIZE)]])
        fmu.setReal([vrs['AmbientTemp']], [AmbientTemp[int(time/STEP_SIZE)]])
        fmu.setReal([vrs['WindSpeed']], [WindSpeed[int(time/STEP_SIZE)]])

        # Esegui un passo
        fmu.doStep(currentCommunicationPoint=time, communicationStepSize=STEP_SIZE)
        y = fmu.getReal([vrs['SensorTemp']])[0]

        times.append(time)
        temps.append(y)
        time += STEP_SIZE

    fmu.terminate()
    fmu.freeInstance()
    shutil.rmtree(unzipdir)

    return np.array(times), np.array(temps)


# ==============================
# CARICA I MODELLI GMM
# ==============================
def load_gmm(name):
    path = os.path.join(GMM_DIR, f"{name.lower()}_gmm.pkl")
    with open(path, 'rb') as f:
        gmm = pickle.load(f)
    print(f"Caricato modello GMM: {path}")
    return gmm

gmm_rad = load_gmm('radiation')
gmm_temp = load_gmm('temperature')
gmm_wind = load_gmm('wind_speed')

# ==============================
# SIMULAZIONE MONTE CARLO
# ==============================
times = None
all_results = []

print(f"\nAvvio Monte Carlo con {N_RUNS} simulazioni...\n")

for i in range(N_RUNS):
    # Campiona rumore realistico da GMM
    rad_noise = gmm_rad.sample(int(T_STOP / STEP_SIZE) + 1)[0].flatten()
    temp_noise = gmm_temp.sample(int(T_STOP / STEP_SIZE) + 1)[0].flatten()
    wind_noise = gmm_wind.sample(int(T_STOP / STEP_SIZE) + 1)[0].flatten()

    # Crea profili temporali realistici (base + rumore)
    base_rad = np.full_like(rad_noise, 800.0)   # costante media
    base_temp = np.full_like(temp_noise, 25.0)
    base_wind = np.full_like(wind_noise, 2.0)

    SolarRadiation = base_rad + rad_noise
    AmbientTemp = base_temp + temp_noise
    WindSpeed = base_wind + wind_noise

    t, temp_output = run_fmu_with_noise(SolarRadiation, AmbientTemp, WindSpeed)
    all_results.append(temp_output)

    if times is None:
        times = t

    print(f"Simulazione {i+1}/{N_RUNS} completata")

all_results = np.array(all_results)

# ==============================
# ANALISI STATISTICA
# ==============================
mean_temp = np.mean(all_results, axis=0)
std_temp = np.std(all_results, axis=0)
p05 = np.percentile(all_results, 5, axis=0)
p95 = np.percentile(all_results, 95, axis=0)

# ==============================
# PLOT RISULTATI
# ==============================
plt.figure(figsize=(10,6))
plt.plot(times, mean_temp, 'k-', label='Media (Monte Carlo)')
plt.fill_between(times, p05, p95, color='lightblue', alpha=0.5, label='Intervallo 90%')
plt.xlabel('Tempo [s]')
plt.ylabel('Temperatura Sensore [°C]')
plt.title(f'FMU Monte Carlo ({N_RUNS} run) con rumore GMM')
plt.legend()
plt.grid(True)
plt.show()
