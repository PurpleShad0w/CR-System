# Sujet I

python process_onenote.py name --transcribe


# Sujet II

python run_enrich_from_consumptiondrift.py --db-dir "db"

python -m pipeline.run_clean --config config.yaml --level all
python -m pipeline.run_train --config config.yaml --level all --target all
python -m pipeline.run_evaluate --config config.yaml --level all --target all
python -m pipeline.run_report --config config.yaml --level all --target all --site all
python -m pipeline.run_baseline_report --config config.yaml --level all --target all --site all
python -m pipeline.run_predict --config config.yaml --level all --target all --days 7


# Sujet III

python -m src.pipeline --input-dwg data/input_plans/sample.dwg --oda-exe "C:/Program Files (x86)/ODA/Teigha File Converter 4.3.2/TeighaFileConverter.exe"

