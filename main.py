import pipeline


features_file_path = "data/features.csv"
datapath = "data/voice_project_data"
# pipeline.dev(model="svc", features_file_path=features_file_path, train=True)
pipeline.final_out("./data", "svc")
