import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

EVAL_THRESHOLD = 0.70
DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
LABELS = [0, 1, 2]


def configure_mlflow() -> None:
    """Use the README tracking store unless the caller provides another one."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)


def get_label_distribution(y: pd.Series) -> dict[str, float]:
    distribution = y.value_counts(normalize=True).to_dict()
    return {str(label): float(distribution.get(label, 0.0)) for label in LABELS}


def write_report(y_true: pd.Series, preds, path: str) -> None:
    matrix = confusion_matrix(y_true, preds, labels=LABELS)
    report = classification_report(
        y_true,
        preds,
        labels=LABELS,
        zero_division=0,
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("Confusion matrix (rows=true, columns=predicted)\n")
        f.write("Labels: 0, 1, 2\n")
        f.write(str(matrix))
        f.write("\n\nClassification report\n")
        f.write(report)


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """

    configure_mlflow()

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen RandomForestClassifier
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap danh gia va tinh chi so
        preds = model.predict(X_eval)
        acc   = accuracy_score(y_eval, preds)
        f1    = f1_score(y_eval, preds, average="weighted")
        label_distribution = get_label_distribution(y_train)

        for label, ratio in label_distribution.items():
            if ratio < 0.10:
                print(
                    f"WARNING: class {label} is only "
                    f"{ratio:.2%} of the training data"
                )

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Luu metrics ra file outputs/metrics.json
        os.makedirs("outputs", exist_ok=True)
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "label_distribution": label_distribution,
        }
        with open("outputs/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        report_path = "outputs/report.txt"
        write_report(y_eval, preds, report_path)
        mlflow.log_artifact(report_path)

        # TODO 9: Luu mo hinh ra file models/model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Tra ve acc
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
