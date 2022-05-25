from src.scripts.preparation import *
import shutil

if __name__ == "__main__":
    input_file = input("Enter File Path: ")
    shutil.copyfile(input_file, "../CS385_3HorseTea-project/data/raw data/data_raw.csv")
    preprocessing()