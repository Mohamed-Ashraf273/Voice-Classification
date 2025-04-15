import pipeline
import multiprocessing



def main():
    features_file_path = "data/features.csv"
    datapath = "data/uncompressed/"
    pipeline.dev(model="svc", datapath=datapath,features_file_path=features_file_path, train=True)

    # pipeline.predict_all(
    #     test_file_path="data/voice_project_data/sampled_50k.tsv",
    #     model_selected="svc",
    # )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()