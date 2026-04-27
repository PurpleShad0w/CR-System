# Sujet II

python -m pipeline.run_clean --config config.yaml --level site
python -m pipeline.run_train --config config.yaml --level site --target elecTotalKwh
python -m pipeline.run_evaluate --config config.yaml --level site --target elecTotalKwh
python -m pipeline.run_report --config config.yaml --level site --target elecTotalKwh
