# Green Rooftop Simulation & Performance Evaluation
Project for the course of Simulation &amp; Performance Evaluation 2024/2025, done by Luca Bissoli and Alberto Messa

## Project Structure
```plaintext
├── data/
│   ├── csv/                            # Output CSV files: trend, residuals, tuning params
│   ├── pkl_models/
│   ├── plots/                          # Output plots: bic scores, residuals, gmm plots
│   ├── air_quality.csv                 # Raw air quality data
│   ├── analyze_data.py                 # Script to clean data and generate trend/params CSVs
│   ├── build_model.py                  # Script to build and save GMM models
│   └── open-meteo.csv                  # Raw weather data
│
├── evaluation/
│   ├── results/
|   |   ├── air_quality/                # Evaluation results for generated air quality data
|   |   ├── simulation_eval/            # Evaluation results for simulation data
|   |   ├── weather/                    # Evaluation results for generated weather data
│   |   └── pollutants_eval.png
│   ├── ecology_removal_analysis.py
│   ├── simulation_eval.py
│   └── weather_air_quality_eval.py
│
├── simulation/
│   ├── blocks/
│   ├── csv_results/
│   |   ├── air_quality_samples.csv     # Generated air quality samples
│   |   ├── simulation_results.csv      # Generated simulation results
│   |   └── weather_samples.csv         # Generated weather samples
│   ├── model/
│   └── run_MC_simulation.py
│
├── README.md
├── requirements.txt
└── theory.md                           # Physical/biological theory/formulas behind the simulator
└── theory.pdf
```

## Data analysis
 ```bash
  cd data
  python analyze_data.py
 ```

Set up parameters in `build_model.py` before running, based on BIC scores.
 ```bash
  cd data
  python build_model.py
 ```

## Run Monte Carlo Simulation
From the project root:

 ```bash
  cd simulation
  python run_MC_simulation.py
 ```

## Evaluate
From the project root:

 ```bash
  cd evaluation
  python simulation_eval.py
  python weather_air_quality_eval.py
  python ecology_removal_analysis.py
 ```