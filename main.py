import pipeline


features_file_path = "data/features.csv"
datapath = "data/voice_project_data"
# pipeline.dev(
#     model="xgboost",
#     features_file_path=features_file_path,
#     train=True,
# )

pipeline.predict_all("./data/test_data.json")
