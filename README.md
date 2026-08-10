- n ----> Name
- d ----> dep.
- p ----> params
- o ----> output

dvc stage add -n data_ingestion -d .\src\data_ingestion.py -d .\data\sample_data.csv -p data_ingestion.test_size -p data_ingestion.random_state -o data/raw python .\src\data_ingestion.py
