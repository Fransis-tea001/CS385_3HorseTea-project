import warnings
warnings.filterwarnings("ignore")

from src.scripts.preparation import preprocessing
from src.scripts.modelling.k_means_model import k_means_modelling
from src.scripts.modelling.aggmolative_model import agglomerative_modelling
from src.scripts.evaluation import evaluate
import shutil

if __name__ == "__main__":
    input_file = input("Enter File Path: ")
    try:
        shutil.copyfile(input_file, "../CS385_3HorseTea-project/data/raw data/data_raw.csv")

        print("Running pipeline...")
    
        print("[START] preposessing")
        preprocessing()
        print("[START] Training model")
        kmeans, X_kmeans = k_means_modelling()
        agg_ward, X_agg_ward = agglomerative_modelling('ward')
        agg_complete, X_agg_complete = agglomerative_modelling('complete')
        agg_average, X_agg_average = agglomerative_modelling('average')
        print("[FINISH TRAINING]")
        
        model = [kmeans, agg_ward, agg_complete, agg_average]
        model_result = [X_kmeans, X_agg_ward, X_agg_complete, X_agg_average]
        model_names = ['k-Means', 'Agglomerative (Ward)', 'Agglomerative (Complete)', 'Agglomerative (Average)']
        evaluate(model, model_result, model_names)
    except:
        print("File not found exception")

