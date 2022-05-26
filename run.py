from src.scripts.preparation import preprocessing
from src.scripts.modelling.k_means_model import k_means_modelling
from src.scripts.modelling.aggmolative_model import agglomerative_modelling
from src.scripts.evaluation import evaluate
import shutil

if __name__ == "__main__":
    input_file = input("Enter File Path: ")
    print("Running pipeline...")
    shutil.copyfile(input_file, "../CS385_3HorseTea-project/data/raw data/data_raw.csv")
    print("[START] preposessing")
    preprocessing()
    print("[START] Training model")
    kmeans = k_means_modelling()
    agg_ward = agglomerative_modelling('ward')
    agg_complete = agglomerative_modelling('complete')
    agg_average = agglomerative_modelling('average')
    print("[FINISH TRAINING]")
    evaluate(kmeans, agg_ward, agg_complete, agg_average)

