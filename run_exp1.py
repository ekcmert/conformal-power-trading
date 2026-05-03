from __future__ import annotations

from exp1 import (
    base_learner_benchmarking,
    feature_selection,
    initial_tests,
    initial_training,
    iterative_retraining,
    train_interval_models,
    train_point_models,
)


RUN_INITIAL_TESTS = False
RUN_FEATURE_SELECTION = False
RUN_TRAIN_POINT_MODELS = False
RUN_TRAIN_INTERVAL_MODELS = False
RUN_INITIAL_TRAINING = True
RUN_ITERATIVE_RETRAINING = False
RUN_BASE_LEARNER_BENCHMARKING = True


def main() -> int:
    if RUN_INITIAL_TESTS:
        initial_tests.main()

    if RUN_FEATURE_SELECTION:
        feature_selection.main()

    if RUN_TRAIN_POINT_MODELS:
        train_point_models.main()

    if RUN_TRAIN_INTERVAL_MODELS:
        train_interval_models.main()

    if RUN_INITIAL_TRAINING:
        initial_training_result = initial_training.main()
        if initial_training_result:
            return int(initial_training_result)

    if RUN_ITERATIVE_RETRAINING:
        iterative_retraining_result = iterative_retraining.main()
        if iterative_retraining_result:
            return int(iterative_retraining_result)

    if RUN_BASE_LEARNER_BENCHMARKING:
        benchmark_result = base_learner_benchmarking.main()
        if benchmark_result:
            return int(benchmark_result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
