import numpy as np
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import tempfile
import shutil
import matplotlib.pyplot as plt
import os

# ==============================
# CONFIGURAZIONE
# ==============================
fmu_filename = 'GreenRoofSim.fmu'   # Nome del file FMU
t_stop = 3600.0                     # Durata simulazione [s]
step_size = 10.0                    # Passo di simulazione [s]

# ==============================
# CARICAMENTO MODELLO
# ==============================
# Crea una cartella temporanea dove estrarre il contenuto dell’FMU
unzipdir = tempfile.mkdtemp()

# Legge la descrizione del modello (modelDescription.xml)
model_description = read_model_description(fmu_filename)

# Estrae i file binari e XML
extract(fmu_filename, unzipdir)

# Crea un dizionario con i valueReference delle variabili
vrs = {}
for variable in model_description.modelVariables:
    vrs[variable.name] = variable.valueReference

# Istanzia il modello FMU
fmu = FMU2Slave(
    guid=model_description.guid,
    unzipDirectory=unzipdir,
    modelIdentifier=model_description.coSimulation.modelIdentifier,
    instanceName='instance1'
)

# Inizializzazione FMI
fmu.instantiate()
fmu.setupExperiment(startTime=0.0)
fmu.enterInitializationMode()
fmu.exitInitializationMode()

# ==============================
# SIMULAZIONE
# ==============================
time = 0.0
times = []
sensor_temp = []

# Valori fissi di input (puoi renderli dinamici più avanti)
SolarRadiation = 800.0  # [W/m²]
AmbientTemp = 25.0      # [°C]
WindSpeed = 2.0         # [m/s]

print("Simulazione avviata...")

while time <= t_stop:
    # Imposta gli input
    fmu.setReal([vrs['SolarRadiation']], [SolarRadiation])
    fmu.setReal([vrs['AmbientTemp']], [AmbientTemp])
    fmu.setReal([vrs['WindSpeed']], [WindSpeed])

    # Esegui un passo di co-simulazione
    status = fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)
    if status != 0:
        print(f"⚠️ Avviso: passo a t={time} restituisce status={status}")

    # Leggi l’output
    y = fmu.getReal([vrs['SensorTemp']])[0]

    # Salva i risultati
    times.append(time)
    sensor_temp.append(y)

    time += step_size

# ==============================
# TERMINA
# ==============================
fmu.terminate()
fmu.freeInstance()
shutil.rmtree(unzipdir)

print("Simulazione completata ✅")

# ==============================
# RISULTATI
# ==============================
plt.figure()
plt.plot(times, sensor_temp, '-o')
plt.xlabel('Time [s]')
plt.ylabel('Sensor Temperature [°C]')
plt.title('Simulazione GreenRoofSim.fmu')
plt.grid(True)
plt.show()
