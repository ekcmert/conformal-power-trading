from __future__ import annotations

from data_prep import finalize_data, load_eq_data, preprocess_data


RUN_LOAD_EQ_DATA = False
RUN_PREPROCESS_DATA = True
RUN_FINALIZE_DATA = True


def main() -> int:
    if RUN_LOAD_EQ_DATA:
        load_eq_data.main()

    if RUN_PREPROCESS_DATA:
        preprocess_results = preprocess_data.run_preprocess_pipeline(
            input_dir=preprocess_data.INPUT_DIR,
            output_dir=preprocess_data.OUTPUT_DIR,
            cutoff_utc=preprocess_data.CUTOFF_UTC,
            ext=preprocess_data.FILE_EXT,
            read_csv_kwargs=preprocess_data.READ_CSV_KWARGS,
        )
        print("\nRemoved")
        for name, removed in preprocess_results["removed"].items():
            print(f"{name}: {len(removed)}")
        print("\nMerged Shapes")
        for name, dataframe in preprocess_results["merged"].items():
            print(f"{name}: {getattr(dataframe, 'shape', None)}")

    if RUN_FINALIZE_DATA:
        finalization_result = finalize_data.run_finalization_pipeline(
            cache_rebuilt_processed=True,
        )
        print(f"Data source: {finalization_result.data_source}")
        print(f"X saved to: {finalization_result.X_path}")
        print(f"y saved to: {finalization_result.y_path}")
        print(f"X shape: {finalization_result.X.shape}")
        print(f"y shape: {finalization_result.y.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
