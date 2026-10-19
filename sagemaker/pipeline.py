"""
DEMO ITEM 4 — SageMaker Pipelines (CI/CD DAG) with an evaluation gate.

Builds a pipeline:  evaluate -> ConditionStep(PR-AUC >= threshold)
                                   |-- True  -> register model
                                   |-- False -> fail step

Run it twice to demonstrate both states:
  * SUCCESS:  python sagemaker/pipeline.py --threshold 0.25   (0.321 >= 0.25 -> registers)
  * FAILURE:  python sagemaker/pipeline.py --threshold 0.99   (0.321 <  0.99 -> gate fails)

Record: SageMaker Studio -> Pipelines -> the DAG graph of a green (successful)
        execution, then a red (failed) execution stopped at the evaluation gate.

Keeps cost low: the "evaluate" step just reads the existing metrics from the
model metadata instead of retraining.
"""
from __future__ import annotations

import argparse

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.function_step import step

PIPELINE_NAME = "heart-disease-screening-pipeline"


@step(instance_type="ml.m5.large")
def evaluate() -> float:
    """Evaluation step — returns the model's PR-AUC from training metadata.

    (In a full build this would re-evaluate on a fixed hold-out; here we read
    the metric produced by src/train.py to keep the Learner Lab cost minimal.)
    """
    import json
    meta = json.loads(open("best_model_meta.json").read())
    return float(meta["pr_auc"])


@step(instance_type="ml.m5.large")
def register() -> str:
    """Registration step — placeholder that marks promotion succeeded."""
    return "model-registered"


def build_pipeline(threshold: float) -> Pipeline:
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    thr = ParameterFloat(name="PrAucThreshold", default_value=threshold)

    eval_result = evaluate()
    reg_result = register()

    gate = ConditionStep(
        name="EvaluationGate",
        conditions=[ConditionGreaterThanOrEqualTo(left=eval_result, right=thr)],
        if_steps=[reg_result],
        else_steps=[FailStep(
            name="PromotionBlocked",
            error_message="PR-AUC below threshold — model promotion blocked.",
        )],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[thr],
        steps=[gate],
        sagemaker_session=sess,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.25,
                    help="PR-AUC gate threshold. 0.25 passes, 0.99 forces a fail.")
    args = ap.parse_args()

    role = sagemaker.get_execution_role()
    pipe = build_pipeline(args.threshold)
    pipe.upsert(role_arn=role)
    print(f"[pipeline] upserted {PIPELINE_NAME}  threshold={args.threshold}")
    execution = pipe.start()
    print(f"[pipeline] started execution: {execution.arn}")
    print("[pipeline] watch the DAG in Studio -> Pipelines. "
          "threshold 0.25 -> SUCCESS (registers); 0.99 -> FAIL at the gate.")


if __name__ == "__main__":
    main()
