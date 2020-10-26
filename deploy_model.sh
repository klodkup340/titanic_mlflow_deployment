#!/usr/bin/env sh

echo "Deplying Production model name=TitanicModel"

# Set enviorment variable for the tracking URL where the Model Registry is
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db

# Serve the production model from the model registry
mlflow models serve --model-uri models:/TitaticModel/production --no-conda
