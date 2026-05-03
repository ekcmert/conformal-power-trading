from __future__ import annotations

from generate.generate_regimes import generate_all_regimes
from generate.generate_residuals import generate_residuals
from generate.generate_scales import generate_scales


RUN_GENERATE_SCALES = True
RUN_GENERATE_REGIMES = True
RUN_GENERATE_RESIDUALS = True


def main() -> int:
    if RUN_GENERATE_SCALES:
        scale_summary = generate_scales()
        print(f"Scale files written: {scale_summary.files_written}")

    if RUN_GENERATE_REGIMES:
        regime_summaries = generate_all_regimes(
            skip_existing=True,
            show_progress=True,
        )
        generated_count = sum(not summary.skipped for summary in regime_summaries)
        skipped_count = sum(summary.skipped for summary in regime_summaries)
        print(f"Generated regime files: {generated_count}")
        print(f"Skipped regime files  : {skipped_count}")

    if RUN_GENERATE_RESIDUALS:
        residual_summary = generate_residuals()
        print(f"Processed model-loss directories : {residual_summary.model_loss_directories}")
        print(f"Point residual files written     : {residual_summary.point_files_written}")
        print(f"Interval residual files written  : {residual_summary.interval_files_written}")
        print(f"Skipped prediction files         : {residual_summary.skipped_prediction_files}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
