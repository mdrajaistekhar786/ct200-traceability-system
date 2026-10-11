"""
CT-200 Traceability System — Detailed Gold Standard
====================================================

Retrieval evaluation:
    relevant_node_ids = ALL manually verified evidence nodes required
    to answer the query completely.

End-to-end evaluation:
    ground_truth_facts = atomic facts that a generated answer/test case
    must preserve.

Document versions:
    ct200_manual.pdf       -> v1
    ct200_manual_v2.pdf    -> v2
    ct200_manual_v3.pdf    -> v3

Node convention:
    Section 3.2 -> REQ-3.2
"""

GOLD_STANDARD = [

    # ============================ V1 ============================

    {
        "id": "GS-V1-01",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "What arm circumference is supported, and what cuff sizes are available for the CT-200?",
        "source_sections": ["1.1", "2.2"],
        "source_pages": [1, 2],
        "relevant_node_ids": ["REQ-1.1", "REQ-2.2"],
        "ground_truth_facts": [
            "The CT-200 is intended for adults with an arm circumference of 22–42 cm.",
            "The standard cuff fits arm circumferences of 22–32 cm.",
            "A separate large cuff (CT200-LC) is available for 32–42 cm.",
            "The large cuff must be ordered separately.",
            "Using the standard cuff outside its rated range can produce inaccurate readings.",
        ],
        "critical_facts": [
            "The CT-200 is intended for adults with an arm circumference of 22–42 cm.",
            "The standard cuff fits arm circumferences of 22–32 cm.",
            "A separate large cuff (CT200-LC) is available for 32–42 cm.",
        ],
    },

    {
        "id": "GS-V1-02",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "What is the maximum pressure and what happens if the CT-200 experiences an overpressure condition?",
        "source_sections": ["2.1", "3.2", "4.1", "4.2"],
        "source_pages": [2, 3, 4],
        "relevant_node_ids": ["REQ-2.1", "REQ-3.2", "REQ-4.1", "REQ-4.2"],
        "ground_truth_facts": [
            "The pressure range is 0–299 mmHg.",
            "The initial inflation target is 180 mmHg.",
            "Inflation proceeds in 40 mmHg increments if pulse is not detected by 180 mmHg.",
            "The maximum inflation pressure is 299 mmHg.",
            "If cuff pressure exceeds 299 mmHg, emergency deflation is triggered.",
            "If pressure exceeds 300 mmHg for longer than 3 seconds due to sensor fault, emergency deflation is triggered.",
            "The emergency deflation valve halts inflation and vents the cuff within 2 seconds.",
            "E3 represents an overpressure condition and auto-deflates within 2 seconds.",
        ],
        "critical_facts": [
            "The pressure range is 0–299 mmHg.",
            "The initial inflation target is 180 mmHg.",
            "Inflation proceeds in 40 mmHg increments if pulse is not detected by 180 mmHg.",
            "The maximum inflation pressure is 299 mmHg.",
            "If cuff pressure exceeds 299 mmHg, emergency deflation is triggered.",
            "The emergency deflation valve halts inflation and vents the cuff within 2 seconds.",
            "E3 represents an overpressure condition and auto-deflates within 2 seconds.",
        ],
    },

    {
        "id": "GS-V1-03",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "What happens when an E3 overpressure error occurs, and can the E3 alarm be silenced?",
        "source_sections": ["4.2", "4.3"],
        "source_pages": [4],
        "relevant_node_ids": ["REQ-4.2", "REQ-4.3"],
        "ground_truth_facts": [
            "E3 means an overpressure condition.",
            "The device auto-deflates within 2 seconds.",
            "The device displays E3.",
            "E3 cannot be silenced for safety reasons.",
        ],
        "critical_facts": [
            "E3 means an overpressure condition.",
            "The device auto-deflates within 2 seconds.",
            "The device displays E3.",
            "E3 cannot be silenced for safety reasons.",
        ],
    },

    {
        "id": "GS-V1-04",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "What should the user do if an error code persists after the retry prompts?",
        "source_sections": ["7.1"],
        "source_pages": [6],
        "relevant_node_ids": ["REQ-7.1"],
        "ground_truth_facts": [
            "If an error code persists after the on-screen retry prompt twice, the user should discontinue use.",
            "The user should contact CardioTrack support.",
            "The user should not attempt further self-diagnosis.",
            "E5 indicates an internal sensor fault.",
        ],
        "critical_facts": [
            "If an error code persists after the on-screen retry prompt twice, the user should discontinue use.",
            "The user should contact CardioTrack support.",
            "The user should not attempt further self-diagnosis.",
        ],
    },

    {
        "id": "GS-V1-05",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "How long do the batteries last and when does the CT-200 automatically shut off?",
        "source_sections": ["2.1.1.1", "3.4"],
        "source_pages": [2, 3],
        "relevant_node_ids": ["REQ-2.1.1.1", "REQ-3.4"],
        "ground_truth_facts": [
            "Four AA alkaline batteries provide approximately 300 measurement cycles under typical use.",
            "Typical use is three measurements per day.",
            "The low-battery icon appears below 15% remaining capacity.",
            "The device powers off after 60 seconds of inactivity on the home screen.",
            "The device powers off after 3 minutes of inactivity when a measurement screen is left open without starting a reading.",
        ],
        "critical_facts": [
            "Four AA alkaline batteries provide approximately 300 measurement cycles under typical use.",
            "The low-battery icon appears below 15% remaining capacity.",
            "The device powers off after 60 seconds of inactivity on the home screen.",
            "The device powers off after 3 minutes of inactivity when a measurement screen is left open without starting a reading.",
        ],
    },

    {
        "id": "GS-V1-06",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "How are user profiles selected and how many readings can the CT-200 store?",
        "source_sections": ["3.1", "5.1"],
        "source_pages": [3, 5],
        "relevant_node_ids": ["REQ-3.1", "REQ-5.1"],
        "ground_truth_facts": [
            "The user selects User 1 or User 2 before beginning a measurement.",
            "Readings are stored against the active profile.",
            "The CT-200 stores up to 100 readings per user profile.",
            "When storage is full, the oldest reading for that profile is overwritten automatically.",
            "There is no user-facing warning before an old reading is overwritten.",
        ],
        "critical_facts": [
            "The user selects User 1 or User 2 before beginning a measurement.",
            "Readings are stored against the active profile.",
            "The CT-200 stores up to 100 readings per user profile.",
            "When storage is full, the oldest reading for that profile is overwritten automatically.",
        ],
    },

    # ============================ V2 ============================

    {
        "id": "GS-V2-01",
        "version": "v2",
        "document": "ct200_manual_v2.pdf",
        "query": "What is the cuff inflation sequence and maximum pressure in version 2?",
        "source_sections": ["3.2"],
        "source_pages": [3],
        "relevant_node_ids": ["REQ-3.2"],
        "ground_truth_facts": [
            "The initial inflation target is 180 mmHg.",
            "Inflation increases in 30 mmHg increments when pulse is not detected by 180 mmHg.",
            "The maximum pressure is 299 mmHg.",
            "The increment size was reduced from 40 mmHg to 30 mmHg to improve pulse-detection reliability.",
        ],
        "critical_facts": [
            "The initial inflation target is 180 mmHg.",
            "Inflation increases in 30 mmHg increments when pulse is not detected by 180 mmHg.",
            "The maximum pressure is 299 mmHg.",
        ],
    },

    {
        "id": "GS-V2-02",
        "version": "v2",
        "document": "ct200_manual_v2.pdf",
        "query": "What happens during an E3 overpressure condition in version 2, and can the alarm be silenced?",
        "source_sections": ["4.2", "4.3"],
        "source_pages": [4],
        "relevant_node_ids": ["REQ-4.2", "REQ-4.3"],
        "ground_truth_facts": [
            "E3 means an overpressure condition.",
            "The device auto-deflates within 1.5 seconds.",
            "The device displays E3.",
            "E3 cannot be silenced for safety reasons.",
        ],
        "critical_facts": [
            "E3 means an overpressure condition.",
            "The device auto-deflates within 1.5 seconds.",
            "The device displays E3.",
            "E3 cannot be silenced for safety reasons.",
        ],
    },

    {
        "id": "GS-V2-03",
        "version": "v2",
        "document": "ct200_manual_v2.pdf",
        "query": "Can CT-200 readings be exported, and what information is included in the export?",
        "source_sections": ["5.3"],
        "source_pages": [5],
        "relevant_node_ids": ["REQ-5.3"],
        "ground_truth_facts": [
            "Firmware 1.4 and later supports exporting stored readings as a CSV file.",
            "The CSV contains timestamp, profile, systolic, diastolic, pulse, and classification columns.",
            "Export requires at least one successful Bluetooth sync in the current session.",
        ],
        "critical_facts": [
            "Firmware 1.4 and later supports exporting stored readings as a CSV file.",
            "The CSV contains timestamp, profile, systolic, diastolic, pulse, and classification columns.",
            "Export requires at least one successful Bluetooth sync in the current session.",
        ],
    },

    {
        "id": "GS-V2-04",
        "version": "v2",
        "document": "ct200_manual_v2.pdf",
        "query": "What is the new E6 error code and what happens when it occurs?",
        "source_sections": ["4.2"],
        "source_pages": [4],
        "relevant_node_ids": ["REQ-4.2"],
        "ground_truth_facts": [
            "E6 means Bluetooth sync failure.",
            "The device displays E6 on the next sync attempt.",
            "E6 does not affect measurement.",
        ],
        "critical_facts": [
            "E6 means Bluetooth sync failure.",
            "The device displays E6 on the next sync attempt.",
            "E6 does not affect measurement.",
        ],
    },

    # ============================ V3 ============================

    {
        "id": "GS-V3-01",
        "version": "v3",
        "document": "ct200_manual_v3.pdf",
        "query": "What are the revised battery life and low-battery threshold in version 3?",
        "source_sections": ["2.1.1.1"],
        "source_pages": [2],
        "relevant_node_ids": ["REQ-2.1.1.1"],
        "ground_truth_facts": [
            "Four AA alkaline batteries provide approximately 300 measurement cycles.",
            "The low-battery icon appears when remaining capacity falls below 28%.",
        ],
        "critical_facts": [
            "Four AA alkaline batteries provide approximately 300 measurement cycles.",
            "The low-battery icon appears when remaining capacity falls below 28%.",
        ],
    },

    {
        "id": "GS-V3-02",
        "version": "v3",
        "document": "ct200_manual_v3.pdf",
        "query": "What are the cuff inflation increment and maximum pressure in version 3?",
        "source_sections": ["3.2"],
        "source_pages": [3],
        "relevant_node_ids": ["REQ-3.2"],
        "ground_truth_facts": [
            "The initial inflation target is 180 mmHg.",
            "Inflation increases in 40 mmHg increments when pulse is not detected by 180 mmHg.",
            "The maximum pressure is 299 mmHg.",
        ],
        "critical_facts": [
            "The initial inflation target is 180 mmHg.",
            "Inflation increases in 40 mmHg increments when pulse is not detected by 180 mmHg.",
            "The maximum pressure is 299 mmHg.",
        ],
    },

    {
        "id": "GS-V3-03",
        "version": "v3",
        "document": "ct200_manual_v3.pdf",
        "query": "How quickly does the CT-200 auto-deflate during an E3 overpressure condition?",
        "source_sections": ["4.2"],
        "source_pages": [4],
        "relevant_node_ids": ["REQ-4.2"],
        "ground_truth_facts": [
            "E3 represents an overpressure condition.",
            "The device auto-deflates within 2 seconds.",
            "The device displays E3.",
        ],
        "critical_facts": [
            "E3 represents an overpressure condition.",
            "The device auto-deflates within 2 seconds.",
        ],
    },

    # ====================== TABLE-BASED CASES ======================

    {
        "id": "GS-V1-T01",
        "version": "v1",
        "document": "ct200_manual.pdf",
        "query": "What are the CT-200 pressure range, pulse range, and measurement accuracy?",
        "source_sections": ["2.1"],
        "source_pages": [2],
        "relevant_node_ids": ["REQ-2.1"],
        "ground_truth_facts": [
            "Pressure range is 0–299 mmHg.",
            "Pulse range is 40–199 bpm.",
            "Pressure accuracy is ±3 mmHg.",
            "Pulse accuracy is ±5%.",
        ],
        "critical_facts": [
            "Pressure range is 0–299 mmHg.",
            "Pulse range is 40–199 bpm.",
            "Pressure accuracy is ±3 mmHg.",
            "Pulse accuracy is ±5%.",
        ],
        "retrieval_type": "table",
    },

    {
        "id": "GS-V2-T01",
        "version": "v2",
        "document": "ct200_manual_v2.pdf",
        "query": "What is the E6 Bluetooth sync failure behavior in version 2?",
        "source_sections": ["4.2"],
        "source_pages": [4],
        "relevant_node_ids": ["REQ-4.2"],
        "ground_truth_facts": [
            "E6 means Bluetooth sync failure.",
            "The device displays E6 on the next sync attempt.",
            "E6 does not affect measurement.",
        ],
        "critical_facts": [
            "E6 means Bluetooth sync failure.",
            "The device displays E6 on the next sync attempt.",
            "E6 does not affect measurement.",
        ],
        "retrieval_type": "table",
    },
]


# ======================= VERSION CHANGE TRUTH =======================

VERSION_CHANGES = {
    "v1": {
        "REQ-2.1.1.1": {
            "battery_cycles": "approximately 300",
            "low_battery_threshold": "15%",
        },
        "REQ-3.2": {
            "initial_target": "180 mmHg",
            "inflation_increment": "40 mmHg",
            "maximum_pressure": "299 mmHg",
        },
        "REQ-4.2": {
            "E3_auto_deflation": "within 2 seconds",
            "error_codes": "E1–E5",
        },
    },

    "v2": {
        "REQ-2.1.1.1": {
            "battery_cycles": "approximately 250",
            "low_battery_threshold": "10%",
        },
        "REQ-3.2": {
            "initial_target": "180 mmHg",
            "inflation_increment": "30 mmHg",
            "maximum_pressure": "299 mmHg",
        },
        "REQ-4.2": {
            "E3_auto_deflation": "within 1.5 seconds",
            "error_codes": "E1–E6",
        },
        "REQ-5.3": {
            "firmware": "1.4",
            "export_format": "CSV",
        },
    },

    "v3": {
        "REQ-2.1.1.1": {
            "battery_cycles": "approximately 300",
            "low_battery_threshold": "28%",
        },
        "REQ-3.2": {
            "initial_target": "180 mmHg",
            "inflation_increment": "40 mmHg",
            "maximum_pressure": "299 mmHg",
        },
        "REQ-4.2": {
            "E3_auto_deflation": "within 2 seconds",
            "error_codes": "E1–E5",
        },
    },
}


def get_gold_case(case_id: str):
    for case in GOLD_STANDARD:
        if case["id"] == case_id:
            return case
    raise KeyError(f"Gold-standard case not found: {case_id}")


def validate_gold_standard():
    assert len(GOLD_STANDARD) == 15
    ids = [case["id"] for case in GOLD_STANDARD]
    assert len(ids) == len(set(ids)), "Duplicate gold-standard IDs."

    for case in GOLD_STANDARD:
        assert case["relevant_node_ids"], f"{case['id']} has no relevant nodes."
        assert case["ground_truth_facts"], f"{case['id']} has no facts."
        assert case["critical_facts"], f"{case['id']} has no critical facts."
        assert set(case["critical_facts"]).issubset(
            set(case["ground_truth_facts"])
        ), f"{case['id']} critical_facts must come from ground_truth_facts."


if __name__ == "__main__":
    validate_gold_standard()
    print(f"Gold Standard validated: {len(GOLD_STANDARD)} cases")
