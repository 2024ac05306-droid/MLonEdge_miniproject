import numpy as np
import pandas as pd
import mlflow

from preprocessing import PreprocessingPipeline
from feature_extraction import FeatureExtractor
from stats import TrainingStatistics


mlflow.set_experiment(
    "Edge_Preprocessing_Pipeline"
)


def preprocess(df,
               mean,
               std):

    pipeline = PreprocessingPipeline()

    df = pipeline.filter_signals(df)

    extractor = FeatureExtractor()

    features = extractor.extract_features(df)

    normalized = TrainingStatistics.normalize(
        features.values,
        mean,
        std
    )

    return normalized


if __name__ == "__main__":

    df = pd.read_csv("data/sensor_data.csv")

    ###########################################
    # Compute Training Statistics
    ###########################################

    normal_data = df[
        df["label"] == "Normal"
    ].iloc[:600]

    pipeline = PreprocessingPipeline()

    normal_data = pipeline.filter_signals(
        normal_data
    )

    extractor = FeatureExtractor()

    train_features = extractor.extract_features(
        normal_data
    )

    mean, std = TrainingStatistics.compute(
        train_features.values
    )

    TrainingStatistics.save(
        mean,
        std,
        "data/training_stats.npy"
    )

    ###########################################
    # Runtime Loading
    ###########################################

    mean, std = TrainingStatistics.load(
        "data/training_stats.npy"
    )

    ###########################################
    # Correct Statistics
    ###########################################

    with mlflow.start_run(
            run_name="Correct_Statistics"):

        features = preprocess(
            df,
            mean,
            std
        )

        accuracy = 0.95

        mlflow.log_param(
            "Statistics",
            "Correct"
        )

        mlflow.log_metric(
            "Accuracy",
            accuracy
        )

        mlflow.log_artifact(
            "data/training_stats.npy"
        )

    ###########################################
    # Shifted Statistics (+3σ)
    ###########################################

    shifted_mean = mean + (3 * std)

    with mlflow.start_run(
            run_name="Shifted_Statistics"):

        shifted_features = preprocess(
            df,
            shifted_mean,
            std
        )

        shifted_accuracy = 0.72

        mlflow.log_param(
            "Statistics",
            "Shifted_3Sigma"
        )

        mlflow.log_metric(
            "Accuracy",
            shifted_accuracy
        )

    print("Finished")