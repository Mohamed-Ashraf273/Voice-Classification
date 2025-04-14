import utlis


features_file_path = "data/f_v5/features.csv"
datapath = "data/voice_project_data"
utlis.dev(model="svc", features_file_path=features_file_path, train=True)

# production_phase(
#     test_file_path="./data/voice_project_data",
#     model_selected="svc",
# )
