from __future__ import annotations

from pathlib import Path

from tqdm.auto import tqdm

from energy_quantified.eq_client import EQClient
from energy_quantified.eq_curves import (
    INSTANCE_CURVES,
    OHLC_CURVES,
    SCENARIO_TIMESERIES_CURVES,
    TIMESERIES_CURVES,
)
from energy_quantified.eq_helper import (
    DEFAULT_BEGIN,
    DEFAULT_END,
    DEFAULT_OUTPUT_DIR,
    RelativeInstanceSettings,
    save_curve_dataframe,
)


START_TIME = DEFAULT_BEGIN
END_TIME = DEFAULT_END
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
FILE_EXT = ".csv"
OVERWRITE = True
API_KEY = None

INSTANCE_SETTINGS = RelativeInstanceSettings()


def _new_summary() -> dict[str, int | list[str]]:
    return {
        "saved": 0,
        "skipped": 0,
        "errors": 0,
        "error_messages": [],
    }


def fetch_curve_group(
    name: str,
    curves: list[str],
    loader,
    output_dir: str | Path,
    ext: str,
    overwrite: bool,
) -> dict[str, int | list[str]]:
    summary = _new_summary()

    for curve in tqdm(curves, desc=f"Fetching {name}", unit="curve"):
        try:
            dataframe = loader(curve)
            path = save_curve_dataframe(
                curve_name=curve,
                dataframe=dataframe,
                output_dir=output_dir,
                ext=ext,
                overwrite=overwrite,
            )
            if path is None:
                summary["skipped"] += 1
                print(f"[skip] {name}: {curve} (no data)")
                continue

            summary["saved"] += 1
            print(f"[saved] {name}: {curve} -> {path}")
        except Exception as exc:
            summary["errors"] += 1
            message = f"{name}: {curve} -> {exc}"
            summary["error_messages"].append(message)
            print(f"[error] {message}")

    return summary


def fetch_instance_curves(
    client: EQClient,
    curves: list[str],
    begin: str,
    end: str,
    settings: RelativeInstanceSettings,
    output_dir: str | Path,
    ext: str,
    overwrite: bool,
) -> dict[str, int | list[str]]:
    summary = _new_summary()

    for curve in curves:
        try:
            tags = client.list_instance_tags(curve)
        except Exception as exc:
            summary["errors"] += 1
            message = f"instance-tags: {curve} -> {exc}"
            summary["error_messages"].append(message)
            print(f"[error] {message}")
            continue

        if not tags:
            summary["skipped"] += 1
            print(f"[skip] instance: {curve} (no tags)")
            continue

        for tag in tags:
            try:
                dataframe = client.load_instance(
                    curve=curve,
                    begin=begin,
                    end=end,
                    tag=tag,
                    days_ahead=settings.days_ahead,
                    before_time_of_day=settings.before_time_of_day,
                    issued=settings.issued,
                    frequency=settings.frequency,
                )
                path = save_curve_dataframe(
                    curve_name=curve,
                    dataframe=dataframe,
                    output_dir=output_dir,
                    ext=ext,
                    prefix=tag,
                    overwrite=overwrite,
                )
                if path is None:
                    summary["skipped"] += 1
                    print(f"[skip] instance: {curve} [{tag}] (no data)")
                    continue

                summary["saved"] += 1
                print(f"[saved] instance: {curve} [{tag}] -> {path}")
            except Exception as exc:
                summary["errors"] += 1
                message = f"instance: {curve} [{tag}] -> {exc}"
                summary["error_messages"].append(message)
                print(f"[error] {message}")

    return summary


def raise_on_fetch_errors(summaries: dict[str, dict[str, int | list[str]]]) -> None:
    failing = {
        name: summary
        for name, summary in summaries.items()
        if int(summary["errors"]) > 0
    }
    if not failing:
        return

    lines = ["Energy Quantified fetch completed with errors."]
    for name, summary in failing.items():
        lines.append(
            f"- {name}: saved={summary['saved']}, skipped={summary['skipped']}, errors={summary['errors']}"
        )

        messages = summary.get("error_messages", [])
        if isinstance(messages, list):
            for message in messages[:10]:
                lines.append(f"  {message}")
            if len(messages) > 10:
                lines.append(f"  ... {len(messages) - 10} more error(s)")

    lines.append("Check the messages above, then rerun after fixing the failing curves or tags.")
    raise RuntimeError("\n".join(lines))


def main() -> None:
    client = EQClient(api_key=API_KEY)

    summaries = {
        "timeseries": fetch_curve_group(
            name="timeseries",
            curves=TIMESERIES_CURVES,
            loader=lambda curve: client.load_timeseries(
                curve=curve,
                begin=START_TIME,
                end=END_TIME,
            ),
            output_dir=OUTPUT_DIR,
            ext=FILE_EXT,
            overwrite=OVERWRITE,
        ),
        "scenario": fetch_curve_group(
            name="scenario",
            curves=SCENARIO_TIMESERIES_CURVES,
            loader=lambda curve: client.load_scenario(
                curve=curve,
                begin=START_TIME,
                end=END_TIME,
            ),
            output_dir=OUTPUT_DIR,
            ext=FILE_EXT,
            overwrite=OVERWRITE,
        ),
        "ohlc": fetch_curve_group(
            name="ohlc",
            curves=OHLC_CURVES,
            loader=lambda curve: client.load_ohlc(
                curve=curve,
                begin=START_TIME,
                end=END_TIME,
            ),
            output_dir=OUTPUT_DIR,
            ext=FILE_EXT,
            overwrite=OVERWRITE,
        ),
        "instance": fetch_instance_curves(
            client=client,
            curves=INSTANCE_CURVES,
            begin=START_TIME,
            end=END_TIME,
            settings=INSTANCE_SETTINGS,
            output_dir=OUTPUT_DIR,
            ext=FILE_EXT,
            overwrite=OVERWRITE,
        ),
    }

    print("\nSummary")
    for name, result in summaries.items():
        print(
            f"{name}: saved={result['saved']}, "
            f"skipped={result['skipped']}, errors={result['errors']}"
        )

    raise_on_fetch_errors(summaries)


if __name__ == "__main__":
    main()
