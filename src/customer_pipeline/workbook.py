"""Read only the organizer-provided workbook and resolve Stage 1 test IDs."""
from pathlib import Path

import pandas as pd


EXPECTED_SHEETS = {
    "README", "Stage1_Train", "Stage1_Test", "Stage2_Data", "Stage2_Constraints"
}


def _read_mapping(path):
    mapping_path = Path(path)
    if mapping_path.suffix.lower() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(mapping_path, engine="openpyxl")
    else:
        frame = pd.read_csv(mapping_path)
    required = {"test_row", "customer_id"}
    if not required.issubset(frame.columns):
        raise ValueError("ID mapping file must contain test_row (1-based) and customer_id columns.")
    frame = frame[["test_row", "customer_id"]].copy()
    frame["test_row"] = pd.to_numeric(frame["test_row"], errors="raise").astype(int)
    frame["customer_id"] = frame["customer_id"].astype(str).str.strip()
    if frame["test_row"].duplicated().any() or frame["customer_id"].duplicated().any():
        raise ValueError("ID mapping rows and customer IDs must both be unique.")
    return frame.sort_values("test_row").reset_index(drop=True)


def load_participant_workbook(workbook_path, id_mapping=None):
    path = Path(workbook_path)
    if not path.exists():
        raise FileNotFoundError(f"Organizer workbook not found: {path}")
    excel = pd.ExcelFile(path, engine="openpyxl")
    missing_sheets = EXPECTED_SHEETS.difference(excel.sheet_names)
    if missing_sheets:
        raise ValueError(f"Workbook is missing required sheets: {sorted(missing_sheets)}")

    train = pd.read_excel(excel, sheet_name="Stage1_Train")
    test = pd.read_excel(excel, sheet_name="Stage1_Test")
    stage2 = pd.read_excel(excel, sheet_name="Stage2_Data")
    constraints_frame = pd.read_excel(excel, sheet_name="Stage2_Constraints")
    excel.close()

    required_train = {"customer_id", "y"}
    if not required_train.issubset(train.columns):
        raise ValueError("Stage1_Train must contain customer_id and y.")
    if "y" in test.columns:
        raise ValueError("Stage1_Test must not contain target labels; refusing to use them.")
    if "customer_id" not in stage2.columns:
        raise ValueError("Stage2_Data must contain customer_id.")

    train["customer_id"] = train["customer_id"].astype(str).str.strip()
    stage2["customer_id"] = stage2["customer_id"].astype(str).str.strip()
    train["y"] = train["y"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if train["y"].isna().any():
        raise ValueError("Stage1_Train contains a target value other than yes/no.")
    if train["customer_id"].isna().any() or train["customer_id"].duplicated().any():
        raise ValueError("Stage1_Train customer IDs must be present and unique.")
    if stage2["customer_id"].isna().any() or stage2["customer_id"].duplicated().any():
        raise ValueError("Stage2_Data customer IDs must be present and unique.")

    training_ids = set(train["customer_id"])
    stage2_ids = set(stage2["customer_id"])
    if not training_ids.issubset(stage2_ids):
        raise ValueError("Some Stage1_Train IDs are missing from Stage2_Data.")
    missing_ids = sorted(stage2_ids - training_ids)
    if len(missing_ids) != len(test):
        raise ValueError(
            f"Expected {len(test)} test IDs absent from train, found {len(missing_ids)}."
        )

    id_mapping = str(id_mapping).strip() if id_mapping is not None else ""
    id_mapping_path = None if not id_mapping or id_mapping.lower() in {"none", "null"} else Path(id_mapping)
    workbook_has_ids = (
        "customer_id" in test.columns
        and test["customer_id"].notna().all()
        and test["customer_id"].astype(str).str.strip().ne("").all()
    )

    if id_mapping_path is not None:
        mapping = _read_mapping(id_mapping_path)
        if len(mapping) != len(test) or mapping["test_row"].tolist() != list(range(1, len(test) + 1)):
            raise ValueError("ID mapping file must map each test row exactly once (test_row 1..N).")
        test_ids = mapping["customer_id"].tolist()
        mapping_status = "EXPLICIT_MAPPING_FILE"
        mapping_note = f"ID mapping file {id_mapping_path} supplied; mapping is explicit."
    elif workbook_has_ids:
        test_ids = test["customer_id"].astype(str).str.strip().tolist()
        mapping_status = "WORKBOOK_IDS"
        mapping_note = (
            "ID mapping check: this workbook contains explicit Stage1_Test customer_id values; "
            "they are used directly, verified unique, disjoint from train, and present in Stage2_Data."
        )
    else:
        # The organizer's stated fallback: test row order paired with sorted IDs absent from train.
        test_ids = missing_ids
        mapping_status = "PROVISIONAL_ROW_ORDER"
        mapping_note = (
            "WARNING: PROVISIONAL ID MAPPING. Stage1_Test row order is paired with the sorted "
            "Stage2_Data IDs absent from Stage1_Train. This row-to-ID association is unverified; "
            "replace it with an id_mapping file before treating customer-level results as final."
        )

    test_ids = [str(value).strip() for value in test_ids]
    if len(test_ids) != len(test) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Resolved Stage1_Test IDs must be present and unique.")
    if set(test_ids).intersection(training_ids):
        raise ValueError("Resolved Stage1_Test IDs overlap Stage1_Train.")
    if set(test_ids) != set(missing_ids):
        raise ValueError("Resolved Stage1_Test IDs must equal Stage2_Data IDs absent from Stage1_Train.")
    test["customer_id"] = test_ids
    test["test_row"] = range(1, len(test) + 1)

    if len(train) != 300 or len(test) != 100 or len(stage2) != 400:
        raise ValueError(
            f"Workbook row counts do not match the release contract: train={len(train)}, "
            f"test={len(test)}, Stage2_Data={len(stage2)}."
        )
    if "duration" in train.columns or "duration" in test.columns:
        raise ValueError("duration is forbidden as a predictor.")

    constraints = {}
    for row in constraints_frame.itertuples(index=False):
        symbol = str(getattr(row, "symbol")).strip()
        value = getattr(row, "value")
        if symbol in {"N", "K", "C", "H"}:
            constraints[symbol] = int(value)
    if set(constraints) != {"N", "K", "C", "H"}:
        raise ValueError("Stage2_Constraints must define N, K, C, and H.")
    if constraints["N"] != 10 or constraints["K"] != 6:
        raise ValueError("Workbook rules expected N=10 candidates and K=6 selected.")

    stage2_required = {
        "customer_id", "contact_cost", "expected_deposit_value",
        "capacity_units", "customer_segment",
    }
    if not stage2_required.issubset(stage2.columns):
        raise ValueError(f"Stage2_Data is missing columns: {sorted(stage2_required.difference(stage2.columns))}")

    return {
        "train": train.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "stage2_data": stage2.reset_index(drop=True),
        "constraints": constraints,
        "mapping_status": mapping_status,
        "mapping_note": mapping_note,
        "workbook_path": str(path),
        "train_base_rate": float(train["y"].mean()),
    }
