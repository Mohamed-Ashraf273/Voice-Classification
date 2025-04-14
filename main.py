from src import utlis


features_file_path = "./data/f_v5_best/features.csv"
datapath = "data/voice_project_data"
utlis.dev(datapath, features_file_path, model="svc", train=False)

# production_phase(
#     test_file_path="./data/voice_project_data",
#     model_selected="svc",
# )
