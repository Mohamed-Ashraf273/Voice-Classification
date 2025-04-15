import pipeline


features_file_path = "data/features_accents.csv"
datapath = "data"
pipeline.dev(
    model="svc",
    features_file_path=features_file_path,
    train=True,
    accent_train=True,
)
