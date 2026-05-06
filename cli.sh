# Sujet II

python -m pipeline.run_clean --config config.yaml --level all
python -m pipeline.run_train --config config.yaml --level all --target all
python -m pipeline.run_evaluate --config config.yaml --level all --target all
python -m pipeline.run_report --config config.yaml --level all --target all --site all
python -m pipeline.run_baseline_report --config config.yaml --level all --target all --site all
python -m pipeline.run_predict --config config.yaml --level all --target all --days 7
