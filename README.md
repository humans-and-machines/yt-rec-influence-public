# The Impact of Algorithmic Memory on Content Consumption and Attitudes: A Longitudinal Field Experiment on YouTube

## Project Layout

- `data/` contains the raw survey files, watch history CSVs, cleaned analysis outputs, and supporting inputs such as the Qualtrics export and propensity-matched participant IDs.
- `scripts/` contains the main analysis scripts and notebooks used to process watch histories, score surveys, run the registered analyses, and generate plots.
- `figures/` contains exported charts and analysis visuals generated from the notebooks and scripts.
- `surveys/` contains the ```.qsf''' files for each of our four Qualtrics surveys.
- `requirements.txt` lists the Python dependencies needed to run the scripts and notebooks.

## Key Files

- `scripts/compute_ai_outcomes.py` builds the AI watch-history outcome dataframes.
- `scripts/registered_analysis.ipynb` runs the main registered analysis pipeline.
- `scripts/propensity_weighting.ipynb` covers the propenstiy weighting workflow.
- `scripts/exploratory_channel_topics.py` and `scripts/exploratory_plots.ipynb` support exploratory analysis.
