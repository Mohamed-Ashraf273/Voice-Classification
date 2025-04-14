import pipeline


features_file_path = "data/features.csv"
datapath = "data/voice_project_data"
pipeline.dev(model="svc", datapath=datapath, train=False)

# pipeline.predict_all(
#     test_file_path="data/voice_project_data/sampled_50k.tsv",
#     model_selected="svc",
# )
