from extract import extract_raw_data
from transform import transform_data, save_clean_data
from validate import validate_data
from feature_engineering import engineer_features, save_features


def run_pipeline():
    print("🔹 Starting data pipeline...")

    #Extract
    raw_df = extract_raw_data()
    print(f"Extracted raw data: {raw_df.shape}")

    #Transform
    clean_df = transform_data(raw_df)
    print(f"Transformed data: {clean_df.shape}")

    #Validate
    validate_data(clean_df)
    print("Data validation passed")

    #Save the clean data
    save_clean_data(clean_df)
    print("Clean data saved")

    #Feature Engineering
    feature_df = engineer_features(clean_df)
    print(f"Features engineered: {feature_df.shape}")

    #Save processed features
    save_features(feature_df)
    print("Processed features saved")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    run_pipeline()
