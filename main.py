import pipeline


features_file_path = "data/features.csv"
datapath = "data/voice_project_data"
pipeline.dev(model="svc", datapath=datapath, train=False)
