# Sujet I

python process_onenote.py {name} --transcribe


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
python -m src.layer_review_prepare --dxf data/work/dxf_out/sample.dxf --config config/default_rules.yaml --out-dir data/work/layer_review
streamlit run src/layer_review_app.py -- --review-dir data/work/layer_review

python cli_clean_plan.py data/input_plans/sample.dxf   --rules config/default_rules.yaml   --decisions data/work/layer_review/layer_decisions.yaml   --out output/rendered_clean.png   --entities-csv output/entities_df.csv   --html-25d output/rendered_25d.html
streamlit run app_layer_review.py
