import numpy as np
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import shutil
import tempfile
import matplotlib.pyplot as plt

# ==============================
# CONFIGURAZIONE
# ==============================
fmu_filename = 'GreenRoofModel.fmu'
t_stop = 3600.0      # durata simulazione (s)
step_size = 10.0     # passo simulazione (s)

# ==============================
# CARICAMENTO MODELLO
# ==============================
unzipdir = tempfile.mkdtemp()
model_description = read_model_description(fmu_filename)
extract(fmu_filename, unzipdir)

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

# ==============================
# SIMULAZIONE
# ==============================
time = 0.0
times = []
sensor_temp = []

# valori fissi per esempio (puoi renderli dinamici)
SolarRadiation = 800.0  # W/m2
AmbientTemp = 25.0      # °C
WindSpeed = 2.0         # m/s

while time < t_stop:
    # imposta gli input
    fmu.setReal([fmu.getVariableByName('SolarRadiation').valueReference], [SolarRadiation])
    fmu.setReal([fmu.getVariableByName('AmbientTemp').valueReference], [AmbientTemp])
    fmu.setReal([fmu.getVariableByName('WindSpeed').valueReference], [WindSpeed])

    # esegui un passo
    fmu.doStep(currentCommunicationPoint=time, communicationStepSize=step_size)

    # leggi l’output
    y = fmu.getReal([fmu.getVariableByName('SensorTemp').valueReference])[0]

    # salva i risultati
    times.append(time)
    sensor_temp.append(y)

    time += step_size

# ==============================
# TERMINA
# ==============================
fmu.terminate()
fmu.freeInstance()
shutil.rmtree(unzipdir)

# ==============================
# RISULTATI
# ==============================
plt.plot(times, sensor_temp)
plt.xlabel('Time [s]')
plt.ylabel('Sensor Temperature [°C]')
plt.title('Simulazione greenroofmodel.fmu')
plt.grid(True)
plt.show()
