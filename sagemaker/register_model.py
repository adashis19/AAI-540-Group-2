"""
DEMO ITEM 5 — SageMaker Model Registry.

Packages the EXISTING trained pipeline (models/best_model.joblib) and registers
it as a versioned model package in the `heart-disease-screening` model package
group, with metrics attached and approval status = Approved. No retraining is
done in SageMaker, which keeps the Learner Lab cost minimal.

Run:  python sagemaker/register_model.py
Record: SageMaker -> Model Registry -> heart-disease-screening -> version ->
        Metrics + Approval status = Approved.
"""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

import sagemaker
from sagemaker.sklearn import SKLearnModel

GROUP = "heart-disease-screening"
HERE = Path(__file__).resolve().parent


def build_model_tar() -> str:
    """Package best_model.joblib as model.tar.gz (arcname model.joblib)."""
    src = Path("models/best_model.joblib")
    if not src.exists():
        raise FileNotFoundError("models/best_model.joblib not found. Clone the repo first.")
    tmp = HERE / "model.joblib"
    shutil.copy(src, tmp)
    tar = HERE / "model.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        t.add(tmp, arcname="model.joblib")
    tmp.unlink()
    return str(tar)


def main():
    sess = sagemaker.Session()
    role = sagemaker.get_execution_role()

    tar = build_model_tar()
    model_data = sess.upload_data(tar, key_prefix="model")
    print(f"[reg] uploaded model artifact -> {model_data}")

    # Metrics captured at training time (src/train.py / best_model_meta.json).
    meta = json.loads(Path("models/best_model_meta.json").read_text())
    print(f"[reg] selected={meta['best_model']}  pr_auc={meta['pr_auc']:.4f}")

    model = SKLearnModel(
        model_data=model_data,
        role=role,
        entry_point="inference.py",
        source_dir=str(HERE),
        framework_version="1.2-1",
        sagemaker_session=sess,
    )

    mp = model.register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=GROUP,
        approval_status="Approved",
        description=(
            f"Heart-disease screening {meta['best_model']} "
            f"(PR-AUC={meta['pr_auc']:.3f}); trained {meta['train_years']} "
            f"-> tested {meta['test_years']}."
        ),
    )
    print(f"[reg] registered model package: {mp.model_package_arn}")
    print(f"[reg] group={GROUP}  approval=Approved")


if __name__ == "__main__":
    main()
