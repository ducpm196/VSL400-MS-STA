#!/usr/bin/env python3
"""Fail-closed helpers for MS-STA v5 D0 full-corpus execution (v14).

Does not authorize M0/train. Binary outputs live under data/features/;
Docs/results holds evidence JSON only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

from ms_sta_v5_d0_lib import (  # noqa: E402
    D0_DIR_REL,
    ELIGIBLE_REL,
    EXPECTED_ELIGIBLE_SHA256,
    EXPECTED_METADATA_SHA256,
    EXPECTED_R4_FULL_PROTOCOL_SHA256,
    EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
    EXPECTED_SLOWFAST_CKPT_REL,
    EXPECTED_SLOWFAST_CKPT_SHA256,
    EXPECTED_SPLIT_SHA256,
    FULL_PROTOCOL_REL,
    FULL_PROTOCOL_R7C_AMENDMENT_REL,
    METADATA_REL,
    PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL,
    R7C_AMBIGUITY_ROI_LOCK,
    SPLIT_REL,
    executable_tree_sha256,
    executable_tree_sha256_at_revision,
    git_commit_exists,
    git_ls_files,
    git_porcelain,
    git_rev,
    load_json,
    sha256_file,
    verify_code_lock_revision_binding,
    write_json,
)

CORPUS_N = 24752
SPLIT_COUNTS = {"train": 15992, "val": 2595, "test": 6165}
PHASES_EXACT = ("skeleton_v3", "rgb_token_v3", "quality_scan", "completion")
MAX_RESUMES = 20
OUTPUT_VERSION = "v14"
CAMPAIGN_TOKEN = "D0_FULL_CORPUS_V14"
OUTPUT_VERSION_V15 = "v15"
CAMPAIGN_TOKEN_V15 = "D0_FULL_CORPUS_V15"
OUTPUT_VERSION_V13 = "v13"
CAMPAIGN_TOKEN_V13 = "D0_FULL_CORPUS_V13"
OUTPUT_VERSION_V12 = "v12"
CAMPAIGN_TOKEN_V12 = "D0_FULL_CORPUS_V12"
OUTPUT_VERSION_V11 = "v11"
CAMPAIGN_TOKEN_V11 = "D0_FULL_CORPUS_V11"
OUTPUT_VERSION_V10 = "v10"
CAMPAIGN_TOKEN_V10 = "D0_FULL_CORPUS_V10"
OUTPUT_VERSION_V9 = "v9"
CAMPAIGN_TOKEN_V9 = "D0_FULL_CORPUS_V9"
OUTPUT_VERSION_V8 = "v8"
CAMPAIGN_TOKEN_V8 = "D0_FULL_CORPUS_V8"
OUTPUT_VERSION_V7 = "v7"
CAMPAIGN_TOKEN_V7 = "D0_FULL_CORPUS_V7"
OUTPUT_VERSION_V6 = "v6"
CAMPAIGN_TOKEN_V6 = "D0_FULL_CORPUS_V6"
OUTPUT_VERSION_V5 = "v5"
CAMPAIGN_TOKEN_V5 = "D0_FULL_CORPUS_V5"
OUTPUT_VERSION_V4 = "v4"
CAMPAIGN_TOKEN_V4 = "D0_FULL_CORPUS_V4"
WATCHDOG_INTERVAL_SEC = 600
# RGB RUNNING with zero workers/progress beyond this age is a material startup stall.
STARTUP_STALL_SEC = 2 * WATCHDOG_INTERVAL_SEC

RUNNER_SCRIPT_MARKER = "run_ms_sta_v5_d0_full_corpus"
NON_RUNNER_CMDLINE_MARKERS = (
    "monitor_ms_sta_v5_d0_full_corpus",
    "watch_ms_sta_v5_d0_full_corpus",
    "validate_ms_sta_v5_d0_full_corpus",
    "smoke_ms_sta_v5_d0_full_corpus",
    "write_ms_sta_v5_d0_full_corpus",
    "material_",
)
PSUTIL_PIN = "psutil==7.2.2"

EVIDENCE_ROOT_V2_REL = "Docs/results/ms_sta_v5_d0_full/v2"
EVIDENCE_ROOT_V3_REL = "Docs/results/ms_sta_v5_d0_full/v3"
EVIDENCE_ROOT_V4_REL = "Docs/results/ms_sta_v5_d0_full/v4"
EVIDENCE_ROOT_V5_REL = "Docs/results/ms_sta_v5_d0_full/v5"
EVIDENCE_ROOT_V6_REL = "Docs/results/ms_sta_v5_d0_full/v6"
EVIDENCE_ROOT_V7_REL = "Docs/results/ms_sta_v5_d0_full/v7"
EVIDENCE_ROOT_V8_REL = "Docs/results/ms_sta_v5_d0_full/v8"
EVIDENCE_ROOT_V9_REL = "Docs/results/ms_sta_v5_d0_full/v9"
EVIDENCE_ROOT_V10_REL = "Docs/results/ms_sta_v5_d0_full/v10"
EVIDENCE_ROOT_V11_REL = "Docs/results/ms_sta_v5_d0_full/v11"
EVIDENCE_ROOT_V12_REL = "Docs/results/ms_sta_v5_d0_full/v12"
EVIDENCE_ROOT_V13_REL = "Docs/results/ms_sta_v5_d0_full/v13"
EVIDENCE_ROOT_REL = "Docs/results/ms_sta_v5_d0_full/v14"
EVIDENCE_ROOT_V15_REL = "Docs/results/ms_sta_v5_d0_full/v15"
BINARY_ROOT_PREFIX_V7_REL = "data/features/ms_sta_v5_d0_full_v7_"
BINARY_ROOT_PREFIX_V8_REL = "data/features/ms_sta_v5_d0_full_v8_"
BINARY_ROOT_PREFIX_V9_REL = "data/features/ms_sta_v5_d0_full_v9_"
BINARY_ROOT_PREFIX_V10_REL = "data/features/ms_sta_v5_d0_full_v10_"
BINARY_ROOT_PREFIX_V11_REL = "data/features/ms_sta_v5_d0_full_v11_"
BINARY_ROOT_PREFIX_V12_REL = "data/features/ms_sta_v5_d0_full_v12_"
BINARY_ROOT_PREFIX_V13_REL = "data/features/ms_sta_v5_d0_full_v13_"
BINARY_ROOT_PREFIX_REL = "data/features/ms_sta_v5_d0_full_v14_"
BINARY_ROOT_PREFIX_V15_REL = "data/features/ms_sta_v5_d0_full_v15_"
BINARY_ROOT_PREFIX_V6_REL = "data/features/ms_sta_v5_d0_full_v6_"
BINARY_ROOT_PREFIX_V5_REL = "data/features/ms_sta_v5_d0_full_v5_"
BINARY_ROOT_PREFIX_V4_REL = "data/features/ms_sta_v5_d0_full_v4_"
V14_SEALED_SKELETON_BINARY_ROOT_REL = "data/features/ms_sta_v5_d0_full_v14_2adbdf3d2bb5"
SKEL_JOINT_COUNT = 75
RGB_TOKEN_DIM = 2304
RGB_PATHWAY_SLOW_DIM = 2048
RGB_PATHWAY_FAST_DIM = 256
MANIFEST_PHASES = ("skeleton_v3", "rgb_token_v3")
RUNTIME_REL_SUFFIX = "runtime"

D0_FULL_CORPUS_PRE_EXECUTION_AUDIT_REL = f"{D0_DIR_REL}/D0_FULL_CORPUS_PRE_EXECUTION_AUDIT.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_AUDIT_V2_REL = (
    f"{EVIDENCE_ROOT_V2_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_AUDIT.json"
)
D0_FULL_CORPUS_EXEC_PROTOCOL_V2_REL = f"{EVIDENCE_ROOT_V2_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V2.json"
D0_FULL_CORPUS_APPROVAL_V2_REL = f"{EVIDENCE_ROOT_V2_REL}/D0_FULL_CORPUS_APPROVAL_V2.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V2_REL = f"{EVIDENCE_ROOT_V2_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V2.json"
D0_FULL_CORPUS_RUN_CONTRACT_V2_REL = f"{EVIDENCE_ROOT_V2_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V2.json"

D0_FULL_CORPUS_EXEC_PROTOCOL_V3_REL = f"{EVIDENCE_ROOT_V3_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V3.json"
D0_FULL_CORPUS_APPROVAL_V3_REL = f"{EVIDENCE_ROOT_V3_REL}/D0_FULL_CORPUS_APPROVAL_V3.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V3_REL = f"{EVIDENCE_ROOT_V3_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V3.json"
D0_FULL_CORPUS_RUN_CONTRACT_V3_REL = f"{EVIDENCE_ROOT_V3_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V3.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V3_REL = (
    f"{EVIDENCE_ROOT_V3_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V4_REL = f"{EVIDENCE_ROOT_V4_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V4.json"
D0_FULL_CORPUS_APPROVAL_V4_REL = f"{EVIDENCE_ROOT_V4_REL}/D0_FULL_CORPUS_APPROVAL_V4.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V4_REL = f"{EVIDENCE_ROOT_V4_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V4.json"
D0_FULL_CORPUS_RUN_CONTRACT_V4_REL = f"{EVIDENCE_ROOT_V4_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V4.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V4_REL = (
    f"{EVIDENCE_ROOT_V4_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V5_REL = f"{EVIDENCE_ROOT_V5_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V5.json"
D0_FULL_CORPUS_APPROVAL_V5_REL = f"{EVIDENCE_ROOT_V5_REL}/D0_FULL_CORPUS_APPROVAL_V5.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V5_REL = f"{EVIDENCE_ROOT_V5_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V5.json"
D0_FULL_CORPUS_RUN_CONTRACT_V5_REL = f"{EVIDENCE_ROOT_V5_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V5.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V5_REL = (
    f"{EVIDENCE_ROOT_V5_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V6_REL = f"{EVIDENCE_ROOT_V6_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V6.json"
D0_FULL_CORPUS_APPROVAL_V6_REL = f"{EVIDENCE_ROOT_V6_REL}/D0_FULL_CORPUS_APPROVAL_V6.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V6_REL = f"{EVIDENCE_ROOT_V6_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V6.json"
D0_FULL_CORPUS_RUN_CONTRACT_V6_REL = f"{EVIDENCE_ROOT_V6_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V6.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V6_REL = (
    f"{EVIDENCE_ROOT_V6_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V7_REL = f"{EVIDENCE_ROOT_V7_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V7.json"
D0_FULL_CORPUS_APPROVAL_V7_REL = f"{EVIDENCE_ROOT_V7_REL}/D0_FULL_CORPUS_APPROVAL_V7.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V7_REL = f"{EVIDENCE_ROOT_V7_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V7.json"
D0_FULL_CORPUS_RUN_CONTRACT_V7_REL = f"{EVIDENCE_ROOT_V7_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V7.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V7_REL = (
    f"{EVIDENCE_ROOT_V7_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V8_REL = f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V8.json"
D0_FULL_CORPUS_APPROVAL_V8_REL = f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_APPROVAL_V8.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V8_REL = f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V8.json"
D0_FULL_CORPUS_RUN_CONTRACT_V8_REL = f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V8.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V8_REL = (
    f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V8_REL = (
    f"{EVIDENCE_ROOT_V8_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V9_REL = f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V9.json"
D0_FULL_CORPUS_APPROVAL_V9_REL = f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_APPROVAL_V9.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V9_REL = f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V9.json"
D0_FULL_CORPUS_RUN_CONTRACT_V9_REL = f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V9.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V9_REL = (
    f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V9_REL = (
    f"{EVIDENCE_ROOT_V9_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V10_REL = f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V10.json"
D0_FULL_CORPUS_APPROVAL_V10_REL = f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_APPROVAL_V10.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V10_REL = f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V10.json"
D0_FULL_CORPUS_RUN_CONTRACT_V10_REL = f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V10.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V10_REL = (
    f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V10_REL = (
    f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)
D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL = (
    f"{EVIDENCE_ROOT_V10_REL}/D0_FULL_CORPUS_EXECUTION_INTERRUPTION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V11_REL = f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V11.json"
D0_FULL_CORPUS_APPROVAL_V11_REL = f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_APPROVAL_V11.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V11_REL = f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V11.json"
D0_FULL_CORPUS_RUN_CONTRACT_V11_REL = f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V11.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V11_REL = (
    f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V11_REL = (
    f"{EVIDENCE_ROOT_V11_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V12_REL = f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V12.json"
D0_FULL_CORPUS_APPROVAL_V12_REL = f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_APPROVAL_V12.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V12_REL = f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V12.json"
D0_FULL_CORPUS_RUN_CONTRACT_V12_REL = f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V12.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V12_REL = (
    f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V12_REL = (
    f"{EVIDENCE_ROOT_V12_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL = f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V13.json"
D0_FULL_CORPUS_APPROVAL_V13_REL = f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_APPROVAL_V13.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V13_REL = f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V13.json"
D0_FULL_CORPUS_RUN_CONTRACT_V13_REL = f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V13.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL = (
    f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V13_REL = (
    f"{EVIDENCE_ROOT_V13_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)

D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V14.json"
D0_FULL_CORPUS_APPROVAL_V14_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_APPROVAL_V14.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V14.json"
D0_FULL_CORPUS_RUN_CONTRACT_V14_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V14.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL = (
    f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL = (
    f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)
D0_FULL_CORPUS_EXECUTION_INTERRUPTION_REL = (
    f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_EXECUTION_INTERRUPTION.json"
)
D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL = f"{EVIDENCE_ROOT_REL}/D0_FULL_CORPUS_V14_HALT_CLOSEOUT.json"

D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL = f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_EXECUTION_PROTOCOL_V15.json"
D0_FULL_CORPUS_APPROVAL_V15_REL = f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_APPROVAL_V15.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL = f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT_V15.json"
D0_FULL_CORPUS_RUN_CONTRACT_V15_REL = f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_RUN_CONTRACT_V15.json"
D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)
D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_PROTOCOL.json"
)
D0_FULL_CORPUS_V15_REMEDIATION_STATUS_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_REMEDIATION_STATUS.json"
)
D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_RGB_F0_SMOKE_REPORT.json"
)
D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT.json"
)
D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_PRELAUNCH_BLOCKER.json"
)
D0_FULL_CORPUS_V15_SUPERSESSION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SEAL_SUPERSESSION.json"
)
D0_FULL_CORPUS_V15_INDEPENDENT_AUDIT_FAIL_ORCHESTRATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_INDEPENDENT_AUDIT_FAIL_ORCHESTRATION.json"
)
D0_FULL_CORPUS_V15_WATCHDOG_COLD_START_FAIL_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_WATCHDOG_COLD_START_FAIL.json"
)
D0_FULL_CORPUS_V15_INTERRUPTED_RGB_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_INTERRUPTED_RGB_WORKER_EXIT_WITHOUT_PROGRESS.json"
)
D0_FULL_CORPUS_V15_INTERRUPTED_BOUND_WORKER_RECHECKED_COLD_START_GATE_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_INTERRUPTED_BOUND_WORKER_RECHECKED_COLD_START_GATE.json"
)
D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_QUALITY_SCAN_USED_LOCAL_SKEL_MANIFEST_UNDER_CARRY_FORWARD.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R2_AUDIT_FAIL_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R2_AUDIT_FAIL_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R3_AUDIT_FAIL_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SCAN_RECOVERY_R3_AUDIT_FAIL_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_EXECUTION_PROVENANCE_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_EXECUTION_PROVENANCE_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R5_META_CONTRACT_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_SCAN_RECOVERY_R5_META_CONTRACT_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R6_FAIL_CLOSED_REMEDIATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_SCAN_RECOVERY_R6_FAIL_CLOSED_REMEDIATION.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_FAIL_ATTEMPT_PRESERVATION_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_SCAN_RECOVERY_R4_FAIL_ATTEMPT_PRESERVATION.json"
)
D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/"
    "D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC.json"
)
D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE.json"
)
D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT.json"
)
D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL = (
    f"{EVIDENCE_ROOT_V15_REL}/D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_EXECUTION_AUTHORIZATION.json"
)
SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15 = (
    "ms_sta_v5_d0_full_corpus_scan_recovery_external_reaudit_v15"
)
SCAN_RECOVERY_POST_AUDIT_AUTH_SCHEMA_V15 = (
    "ms_sta_v5_d0_full_corpus_scan_recovery_post_audit_execution_authorization_v15"
)

EXTRACT_PY_REL = "_venvs/extract/Scripts/python.exe"
F0_PY_REL = "_venvs/f0_v5/Scripts/python.exe"
CI_PY_REL = "_venvs/ci/Scripts/python.exe"

EXPECTED_V3_APPROVAL_SHA256 = "4c4579322159e208d8e1a5e4301fce6ae4e062a4701f9c2b80c279c07559084e"
EXPECTED_V3_EXTERNAL_REAUDIT_SHA256 = (
    "1a079e19e7a57ef3684de15cff4b217062094f7efd329613594943b585d16461"
)
EXPECTED_V4_APPROVAL_SHA256 = "4f9a23a002b738dcaf5d58c787a3d8660b355f2ffb25a31cdf20ab2d861cb1f1"
EXPECTED_V4_EXTERNAL_REAUDIT_SHA256 = (
    "329c5046c03677fa1d958528528795baf5cb6c060f0ca611f71745da10078bdc"
)
EXPECTED_V5_APPROVAL_SHA256 = "345340c1089810d5b7b4b87ed77cf50e2e8512846acd3e5d96fa87fc68668b7b"
EXPECTED_V5_EXTERNAL_REAUDIT_SHA256 = (
    "929388bbae3266485cf80a9563ebfa05219926cac4edc33e19683f48306e6e69"
)
EXPECTED_V6_APPROVAL_SHA256 = "371ed1c4c0845d9bca3612d2ab115c0eb9d6878a5e9868bd243ed6a47f68713c"
EXPECTED_V6_EXTERNAL_REAUDIT_SHA256 = (
    "17a45c5e603b7af5eb15e9799bc4b752d09a7cc0438a2a827eb98df8406c4c65"
)
EXPECTED_V7_APPROVAL_SHA256 = "13a9bc8902664a76b5b4ff59c4aa33f40cbc107278a40bca214d36050c94093a"
EXPECTED_V7_EXTERNAL_REAUDIT_SHA256 = (
    "e319c3fcad431bf4108db2e033e578b13c1ae7162acc530111eb7fbe71572752"
)
EXPECTED_V8_APPROVAL_SHA256 = "e8210afe9ff18263e78a343d86cf8c339b9f949aa1cf4a87d3ba4df761cef691"
EXPECTED_V8_EXTERNAL_REAUDIT_SHA256 = (
    "17d9f18351ad45559795e2360633a8bb663abcc56aacc9b29494cde3522b1568"
)
EXPECTED_V9_APPROVAL_SHA256 = "3438d258e4dba7eea50b150ce36a844b7f42a6e2a981b9167c6a919573f59888"
EXPECTED_V9_EXTERNAL_REAUDIT_SHA256 = (
    "2868ed69468e0c19068e3674d3388521506bb258a97188280c1c89e6621cd19b"
)
EXPECTED_V10_APPROVAL_SHA256 = "436984ddf2dc46e01f8cf34adbeb7845fae966fc7950f18f5b8e9c1f3bfa1918"
EXPECTED_V10_INTERRUPTION_SHA256 = (
    "e9c86c860d08248950651c830754575607b22f617c6a48335f905894e57b3f13"
)
EXPECTED_V11_APPROVAL_SHA256 = "de92083d5dc8fc77c4679b261b688ebf97652eae8e6e31d3722f92e4f7c5183c"
EXPECTED_V11_EXTERNAL_REAUDIT_SHA256 = (
    "c9fe3a09e7d19b677fe2c8b800e0731640e469391764b6c2b5f994c9158a0efb"
)
EXPECTED_V12_APPROVAL_SHA256 = "5faa1d784f1eec963842466f59216aba2728c76b36f6f2eccb2854a584b56080"
EXPECTED_V12_EXTERNAL_REAUDIT_SHA256 = (
    "31991f7e5f0a8618c02fb40961f30405c51bca6480d0ae4c41bcb0646269f1ee"
)
EXPECTED_V13_APPROVAL_SHA256 = "19e80e95548b9288f6e62a01e37a17415fd244c42eb1796aab0258c5bca49d62"
EXPECTED_V13_EXTERNAL_REAUDIT_SHA256 = (
    "a4b6e138923cec697a65ccf68c60e8a27f85b2444c6e34578e9145f2530566d8"
)

EXTERNAL_REAUDIT_PASS_SCHEMA = "ms_sta_v5_d0_full_corpus_execution_external_reaudit_pass_v1"
EXTERNAL_REAUDIT_PASS_SCHEMA_V15 = "ms_sta_v5_d0_full_corpus_execution_external_reaudit_pass_v15"
EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION: dict[str, bool] = {
    "post_audit_execution_authorization_allowed": True,
    "launch_authorized": False,
    "full_corpus_extraction_started": False,
    "m0_authorized": False,
    "training_authorized": False,
    "test_authorized": False,
    "demo_authorized": False,
    "auto_chain": False,
}
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

DEFAULT_QUALITY_THRESHOLDS = {
    "max_missing_roi_rate_face": 0.05,
    "max_missing_roi_rate_left_hand": 0.15,
    "max_missing_roi_rate_right_hand": 0.15,
    "max_partial_missing_rate_any_region": 0.25,
}

SKEL_REQUIRED_KEYS = (
    "coordinates",
    "joints_raw",
    "velocity",
    "bone_vectors",
    "missing_mask_prefill",
    "validity",
    "confidence",
    "padding_mask",
    "joint_part",
    "joint_chirality",
    "bone_edges",
    "sequence_length",
    "original_frame_index",
    "timestamps_sec",
    "raw_landmarks",
    "effective_landmarks",
    "raw_holistic_missing",
    "effective_missing",
    "observation_source_left_hand",
    "observation_source_right_hand",
)
RGB_REQUIRED_KEYS = (
    "token_global",
    "token_face",
    "token_left_hand",
    "token_right_hand",
    "missing_face",
    "missing_left_hand",
    "missing_right_hand",
    "partial_face",
    "partial_left_hand",
    "partial_right_hand",
    "pathway_slow",
    "pathway_fast",
    "original_frame_index",
)

SKEL_MANIFEST_REQUIRED_FIELDS = (
    "sample_id",
    "split",
    "signer_id",
    "source_sha256",
    "output_sha256",
    "sidecar_sha256",
    "output_rel",
    "sidecar_rel",
    "producer",
)
RGB_MANIFEST_REQUIRED_FIELDS = (
    "sample_id",
    "split",
    "signer_id",
    "source_sha256",
    "skel_sha256",
    "output_sha256",
    "sidecar_sha256",
    "output_rel",
    "sidecar_rel",
    "producer",
)

FULL_CORPUS_EXECUTABLE_PATHS_V4 = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/data_prep/joint_layout_v3.py",
    "src/data_prep/raw_video_resolve.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/ms_sta_v5_d0_full_corpus_lib.py",
    "scripts/run_ms_sta_v5_d0_full_corpus.py",
    "scripts/scan_ms_sta_v5_d0_full_corpus.py",
    "scripts/monitor_ms_sta_v5_d0_full_corpus.py",
    "scripts/watch_ms_sta_v5_d0_full_corpus.ps1",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V5 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V6 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V5[:-2],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V7 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V6[:-2],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V8 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V9 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V10 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v10.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V11 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v11.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V12 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v12.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v12.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V13 = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v12.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v12.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v13.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v13.py",
    "scripts/audit_ms_sta_v5_d0_full_corpus_live_readiness.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS = [
    *FULL_CORPUS_EXECUTABLE_PATHS_V4[:-3],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v4.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v5.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v6.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v7.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v8.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v9.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v10.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v11.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v12.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v12.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v13.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v13.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v14.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v14.py",
    "scripts/audit_ms_sta_v5_d0_full_corpus_live_readiness.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_c3_parity.py",
    "scripts/verify_ms_sta_v5_d0_full_corpus_v14_halt_closeout.py",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
]

FULL_CORPUS_EXECUTABLE_PATHS_V15 = [
    *FULL_CORPUS_EXECUTABLE_PATHS[:-2],
    "scripts/write_ms_sta_v5_d0_full_corpus_approval_v15.py",
    "scripts/write_ms_sta_v5_d0_full_corpus_post_audit_auth_v15.py",
    "scripts/validate_ms_sta_v5_d0_full_corpus_external_reaudit_v15.py",
    "scripts/validate_ms_sta_v5_d0_full_corpus_v15_skeleton_carry_forward.py",
    "scripts/smoke_ms_sta_v5_d0_full_corpus_v15_rgb_f0.py",
    "scripts/watch_ms_sta_v5_d0_full_corpus_v15.ps1",
    "requirements-d0-full-corpus.txt",
    "tests/test_ms_sta_v5_d0_full_corpus.py",
    "tests/test_ms_sta_v5_d0_v15_rgb_remediation.py",
    "tests/test_ms_sta_v5_d0_v15_lifecycle.py",
    "tests/test_ms_sta_v5_d0_v15_prelaunch_remediation.py",
    "tests/test_ms_sta_v5_d0_v15_carry_forward_orchestration.py",
    "tests/test_ms_sta_v5_d0_v15_watchdog_cold_start.py",
    "tests/test_ms_sta_v5_d0_v15_worker_liveness.py",
    "tests/test_ms_sta_v5_d0_v15_bound_worker_attach.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r1.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r2.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r3.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r4.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r5.py",
    "tests/test_ms_sta_v5_d0_v15_scan_recovery_r6.py",
    "scripts/run_ms_sta_v5_d0_full_corpus_scan_recovery_v15.py",
    "scripts/smoke_ms_sta_v5_d0_full_corpus_scan_recovery_perf_v15.py",
    "scripts/diagnose_ms_sta_v5_d0_v15_rgb_region_validity.py",
]

SKEL_ARRAY_CONTRACT: dict[str, dict[str, Any]] = {
    # Matches src/data_prep/skel_v3.build_skel_v3_arrays SoT dtypes.
    "coordinates": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "joints_raw": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "velocity": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "bone_vectors": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "missing_mask_prefill": {"ndim": 2, "dtype": "bool", "align_group": "frames"},
    "validity": {"ndim": 2, "dtype": "float32", "align_group": "frames"},
    "confidence": {"ndim": 2, "dtype": "float32", "align_group": "frames"},
    "padding_mask": {"ndim": 1, "dtype": "bool", "align_group": "frames"},
    "original_frame_index": {"ndim": 1, "dtype": "int32", "align_group": "frames"},
    "timestamps_sec": {"ndim": 1, "dtype": "float64", "align_group": "frames"},
    "raw_landmarks": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "effective_landmarks": {"ndim": 3, "dtype": "float32", "align_group": "frames"},
    "raw_holistic_missing": {"ndim": 2, "dtype": "bool", "align_group": "frames"},
    "effective_missing": {"ndim": 2, "dtype": "bool", "align_group": "frames"},
    "observation_source_left_hand": {"ndim": 1, "dtype": "uint8", "align_group": "frames"},
    "observation_source_right_hand": {"ndim": 1, "dtype": "uint8", "align_group": "frames"},
    "joint_part": {"ndim": 1},
    "joint_chirality": {"ndim": 1},
    "bone_edges": {"ndim": 2, "dtype": "int32"},
    "sequence_length": {"scalar": True, "dtype": "int32"},
}
RGB_ARRAY_CONTRACT: dict[str, dict[str, Any]] = {
    # Tokens/pathways are (T_windows, D); original_frame_index is native-frame length.
    "token_global": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "token_face": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "token_left_hand": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "token_right_hand": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "pathway_slow": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "pathway_fast": {"ndim": 2, "dtype": "float32", "align_group": "windows"},
    "missing_face": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "missing_left_hand": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "missing_right_hand": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "partial_face": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "partial_left_hand": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "partial_right_hand": {"ndim": 1, "dtype": "bool", "align_group": "windows"},
    "original_frame_index": {"ndim": 1, "dtype": "int32"},
}
COMPLETION_BOUND_FIELDS = (
    "verdict",
    "n_paired",
    "failed_count",
    "quality_stats_sha256",
    "scan_report_sha256",
)
COMPLETION_INTENT_IDENTITY_FIELDS = (
    "verdict",
    "n_paired",
    "failed_count",
    "quality_stats_sha256",
    "scan_report_sha256",
    "binary_root_rel",
    "closed_at",
    "campaign_token",
    "output_version",
)

FORBIDDEN_IMPORT_MARKERS = (
    "run_ms_sta_m0",
    "train_ms_sta",
    "evaluate_ms_sta",
    "demo_ms_sta",
    "auto_chain_to_m0",
)


def code_lock_short(rev: str | None = None, *, root: Path = _ROOT) -> str:
    rev = rev or git_rev(root)
    return rev[:12]


def binary_root_rel(rev: str | None = None, *, root: Path = _ROOT, version: str = OUTPUT_VERSION) -> str:
    if version == OUTPUT_VERSION_V4:
        prefix = BINARY_ROOT_PREFIX_V4_REL
    elif version == OUTPUT_VERSION_V5:
        prefix = BINARY_ROOT_PREFIX_V5_REL
    elif version == OUTPUT_VERSION_V6:
        prefix = BINARY_ROOT_PREFIX_V6_REL
    elif version == OUTPUT_VERSION_V7:
        prefix = BINARY_ROOT_PREFIX_V7_REL
    elif version == OUTPUT_VERSION_V8:
        prefix = BINARY_ROOT_PREFIX_V8_REL
    elif version == OUTPUT_VERSION_V9:
        prefix = BINARY_ROOT_PREFIX_V9_REL
    elif version == OUTPUT_VERSION_V10:
        prefix = BINARY_ROOT_PREFIX_V10_REL
    elif version == OUTPUT_VERSION_V11:
        prefix = BINARY_ROOT_PREFIX_V11_REL
    elif version == OUTPUT_VERSION_V12:
        prefix = BINARY_ROOT_PREFIX_V12_REL
    elif version == OUTPUT_VERSION_V13:
        prefix = BINARY_ROOT_PREFIX_V13_REL
    elif version == OUTPUT_VERSION_V15:
        prefix = BINARY_ROOT_PREFIX_V15_REL
    else:
        prefix = BINARY_ROOT_PREFIX_REL
    short = code_lock_short(rev, root=root)
    return f"{prefix}{short}"


def binary_root_path(
    rev: str | None = None, *, root: Path = _ROOT, version: str = OUTPUT_VERSION
) -> Path:
    return root / binary_root_rel(rev, root=root, version=version)


def evidence_root_path(*, root: Path = _ROOT) -> Path:
    return root / EVIDENCE_ROOT_REL


def sha256_git_blob_at_revision(rel: str, rev: str, *, root: Path = _ROOT) -> str:
    """SHA-256 of file bytes at an exact git revision via `git show rev:rel`."""
    norm = rel.replace("\\", "/")
    try:
        blob = subprocess.check_output(["git", "show", f"{rev}:{norm}"], cwd=root)
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"git_blob_missing_at_rev:{norm}@{rev}") from exc
    return hashlib.sha256(blob).hexdigest()


def runtime_dir(binary_root: Path) -> Path:
    return binary_root / RUNTIME_REL_SUFFIX


def atomic_create_json(path: Path, obj: dict[str, Any]) -> str:
    """Create-only atomic JSON: temp write + validate + hard link publish; never partial final bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{path}")
    content_sha = hashlib.sha256(raw).hexdigest()

    if path.is_file():
        try:
            load_json(path)
            existing_sha = sha256_file(path)
            if existing_sha == content_sha:
                return existing_sha
            raise FileExistsError(f"exists_with_different_content:{path}")
        except json.JSONDecodeError:
            path.unlink(missing_ok=True)

    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{time.time_ns()}")

    def _publish_from_tmp() -> None:
        try:
            os.link(tmp, path)
            tmp.unlink(missing_ok=True)
            return
        except OSError:
            pass
        try:
            with path.open("xb") as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
        except FileExistsError:
            if not path.is_file():
                raise
            try:
                load_json(path)
                existing_sha = sha256_file(path)
            except json.JSONDecodeError:
                path.unlink(missing_ok=True)
                with path.open("xb") as fh:
                    fh.write(raw)
                    fh.flush()
                    os.fsync(fh.fileno())
            else:
                if existing_sha == content_sha:
                    return
                raise FileExistsError(f"exists_with_different_content:{path}")
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    try:
        with tmp.open("wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        load_json(tmp)
        tmp_sha = sha256_file(tmp)
        if tmp_sha != content_sha:
            raise RuntimeError(f"temp_sha_mismatch:{path}")
        _publish_from_tmp()
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return sha256_file(path)


def atomic_replace_bytes(dest: Path, data: bytes) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return sha256_file(dest)


def atomic_replace_json(dest: Path, obj: dict[str, Any]) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{dest}")
    return atomic_replace_bytes(dest, raw)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def load_eligible_identity(*, root: Path = _ROOT) -> dict[str, Any]:
    eligible = load_json(root / ELIGIBLE_REL)
    if sha256_file(root / ELIGIBLE_REL) != EXPECTED_ELIGIBLE_SHA256:
        raise RuntimeError("eligible_manifest_sha_drift")
    ids = list(eligible.get("eligible_ids") or [])
    if len(ids) != CORPUS_N:
        raise RuntimeError(f"eligible_n_mismatch:{len(ids)}")
    if len(set(ids)) != CORPUS_N:
        raise RuntimeError("eligible_ids_not_unique")
    by_split = eligible.get("eligible_by_split") or {}
    for k, n in SPLIT_COUNTS.items():
        if int(by_split.get(k) or -1) != n:
            raise RuntimeError(f"split_count_mismatch:{k}")
    return {
        "eligible_ids": ids,
        "eligible_id_set": set(ids),
        "eligible_by_split": dict(by_split),
        "eligible_n": CORPUS_N,
        "eligible_sha256": EXPECTED_ELIGIBLE_SHA256,
    }


def env_provenance(*, root: Path = _ROOT) -> dict[str, Any]:
    extract_py = root / EXTRACT_PY_REL
    f0_py = root / F0_PY_REL
    ci_py = root / CI_PY_REL
    out: dict[str, Any] = {
        "extract_python_rel": EXTRACT_PY_REL,
        "f0_python_rel": F0_PY_REL,
        "ci_python_rel": CI_PY_REL,
        "extract_python_exists": extract_py.is_file(),
        "f0_python_exists": f0_py.is_file(),
        "ci_python_exists": ci_py.is_file(),
        "psutil_pin": PSUTIL_PIN,
    }
    for key, py in (("extract", extract_py), ("f0", f0_py), ("ci", ci_py)):
        if not py.is_file():
            out[f"{key}_python_sha256"] = None
            out[f"{key}_pip_freeze_sha256"] = None
            out[f"{key}_psutil"] = False
            continue
        out[f"{key}_python_sha256"] = sha256_file(py)
        try:
            freeze = subprocess.check_output([str(py), "-m", "pip", "freeze"], cwd=root)
            out[f"{key}_pip_freeze_sha256"] = hashlib.sha256(freeze).hexdigest()
            out[f"{key}_psutil"] = any(
                line.lower().startswith("psutil==7.2.") for line in freeze.decode("utf-8", errors="ignore").splitlines()
            )
        except Exception as exc:  # noqa: BLE001
            out[f"{key}_pip_freeze_error"] = str(exc)
            out[f"{key}_pip_freeze_sha256"] = None
            out[f"{key}_psutil"] = False
    if not (out.get("extract_psutil") and out.get("f0_psutil") and out.get("ci_psutil")):
        out["psutil_all_envs_ok"] = False
    else:
        out["psutil_all_envs_ok"] = True
    return out


def assert_no_forbidden_symbols(paths: list[str], *, root: Path = _ROOT) -> None:
    for rel in paths:
        text = (root / rel).read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        for marker in FORBIDDEN_IMPORT_MARKERS:
            if marker.lower() in lower and "FORBIDDEN" not in text:
                # Allow explicit forbid lists / comments that mention markers with FORBIDDEN context
                if f'"{marker}"' in text or f"'{marker}'" in text:
                    # string constants listing forbid names are OK
                    continue
                if marker in text and "forbid" in lower:
                    continue
                raise RuntimeError(f"forbidden_symbol:{rel}:{marker}")


def assert_no_out_of_provenance_sitecustomize(*, root: Path = _ROOT) -> None:
    """Fail-closed if sitecustomize.py exists outside code-lock provenance in campaign venvs."""
    offenders = [
        "_venvs/f0_v5/Lib/site-packages/sitecustomize.py",
        "_venvs/extract/Lib/site-packages/sitecustomize.py",
    ]
    found = [rel for rel in offenders if (root / rel).is_file()]
    if found:
        raise RuntimeError("sitecustomize_outside_provenance:" + ",".join(found))


def process_tree_alive(pid: int) -> bool:
    """Return True if pid refers to a live (non-zombie) process."""
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psutil_required_for_process_guard") from exc
    try:
        proc = psutil.Process(pid)
        if not proc.is_running():
            return False
        return proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _cmdline_blob(proc: Any) -> str:
    try:
        parts = proc.cmdline() or []
        return " ".join(str(x) for x in parts).lower()
    except Exception:
        try:
            return str(proc.name() or "").lower()
        except Exception:
            return ""


def _process_identity(pid: int) -> dict[str, Any] | None:
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psutil_required_for_process_guard") from exc
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return None
        return {
            "pid": int(pid),
            "create_time": float(proc.create_time()),
            "cmdline": _cmdline_blob(proc),
            "name": str(proc.name() or ""),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None


def worker_entry_alive(entry: dict[str, Any]) -> bool:
    """True iff registered PID is live and create_time matches (blocks PID reuse)."""
    pid = int(entry.get("pid") or 0)
    expected_ct = entry.get("create_time")
    ident = _process_identity(pid)
    if ident is None:
        return False
    if expected_ct is None:
        return False
    # Float compare with small tolerance for serialization.
    if abs(float(ident["create_time"]) - float(expected_ct)) >= 1e-3:
        return False
    token = str(entry.get("campaign_token") or "")
    cmdline = str(ident.get("cmdline") or entry.get("cmdline") or "")
    if token and not is_campaign_runner_cmdline(cmdline, token):
        return False
    return True


def is_campaign_runner_cmdline(blob: str, campaign_token: str) -> bool:
    """True only for the full-corpus runner with the exact campaign token.

    Monitor/watchdog/observer/tooling cmdlines that merely mention the campaign
    token must never count as orchestrator or worker liveness.
    """
    text = (blob or "").lower().replace("\\", "/")
    if RUNNER_SCRIPT_MARKER not in text:
        return False
    for marker in NON_RUNNER_CMDLINE_MARKERS:
        if marker in text:
            return False
    token = (campaign_token or "").strip().lower()
    if not token or token not in text:
        return False
    return True


def load_worker_registry_document(runtime: Path) -> dict[str, Any] | None:
    path = runtime / "WORKER_REGISTRY.json"
    if not path.is_file():
        return None
    return load_json(path)


def load_worker_registry(runtime: Path) -> list[dict[str, Any]]:
    doc = load_worker_registry_document(runtime)
    if doc is None:
        return []
    return list(doc.get("workers") or [])


def save_worker_registry(
    runtime: Path,
    workers: list[dict[str, Any]],
    *,
    campaign_token: str,
) -> None:
    token = str(campaign_token or "").strip()
    if not token:
        raise ValueError("worker_registry_campaign_token_required")
    for entry in workers:
        entry_token = str(entry.get("campaign_token") or "").strip()
        if entry_token and entry_token != token:
            raise ValueError(
                f"worker_registry_campaign_mismatch:top={token}:entry={entry_token}"
            )
        if not entry_token:
            entry["campaign_token"] = token
    atomic_replace_json(
        runtime / "WORKER_REGISTRY.json",
        {
            "schema": "ms_sta_v5_d0_full_corpus_worker_registry_v1",
            "campaign_token": token,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "workers": workers,
        },
    )


def register_worker(
    runtime: Path,
    *,
    pid: int,
    phase: str,
    campaign_token: str = CAMPAIGN_TOKEN,
) -> dict[str, Any]:
    token = str(campaign_token or "").strip() or CAMPAIGN_TOKEN
    existing = load_worker_registry_document(runtime)
    if existing is not None:
        existing_token = str(existing.get("campaign_token") or "").strip()
        if existing_token and existing_token != token:
            raise ValueError(
                f"worker_registry_campaign_mismatch:top={existing_token}:register={token}"
            )
    ident = _process_identity(pid)
    if ident is None:
        raise RuntimeError(f"cannot_register_dead_worker:{pid}")
    if not is_campaign_runner_cmdline(ident["cmdline"], token):
        raise RuntimeError(f"cannot_register_non_runner_worker:{pid}")
    entry = {
        "pid": int(pid),
        "create_time": float(ident["create_time"]),
        "phase": phase,
        "campaign_token": token,
        "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cmdline": ident["cmdline"][:500],
    }
    workers = [w for w in load_worker_registry(runtime) if worker_entry_alive(w)]
    workers.append(entry)
    save_worker_registry(runtime, workers, campaign_token=token)
    return entry


def unregister_worker(
    runtime: Path,
    pid: int,
    *,
    campaign_token: str | None = None,
) -> None:
    existing = load_worker_registry_document(runtime)
    token = str(campaign_token or "").strip()
    if not token and existing is not None:
        token = str(existing.get("campaign_token") or "").strip()
    if not token:
        token = CAMPAIGN_TOKEN
    if existing is not None:
        existing_token = str(existing.get("campaign_token") or "").strip()
        if existing_token and existing_token != token:
            raise ValueError(
                f"worker_registry_campaign_mismatch:top={existing_token}:unregister={token}"
            )
    workers = [w for w in load_worker_registry(runtime) if int(w.get("pid") or 0) != int(pid)]
    # Drop cross-campaign entries rather than rewriting under the wrong token.
    workers = [w for w in workers if str(w.get("campaign_token") or token) == token]
    save_worker_registry(runtime, workers, campaign_token=token)


def scan_campaign_processes(*, campaign_token: str = CAMPAIGN_TOKEN) -> list[dict[str, Any]]:
    """Scan process table for exact full-corpus runner identity only."""
    try:
        import psutil
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("psutil_required_for_process_guard") from exc
    token = (campaign_token or "").strip()
    hits: list[dict[str, Any]] = []
    for proc in psutil.process_iter(["pid", "create_time", "name"]):
        try:
            blob = _cmdline_blob(proc)
            if not is_campaign_runner_cmdline(blob, token):
                continue
            hits.append(
                {
                    "pid": int(proc.pid),
                    "create_time": float(proc.create_time()),
                    "cmdline": blob[:500],
                    "match": "runner_script",
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return hits


def _self_process_tree_pids() -> set[int]:
    """PIDs in the same process tree as this interpreter (self + parents + children).

    Used to ignore Windows python-launcher / parent-shell false positives when
    scanning the process table for campaign tokens. Registry and root_pid checks
    remain authoritative for live workers/orchestrators.
    """
    pids: set[int] = {int(os.getpid())}
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return pids
    try:
        me = psutil.Process(os.getpid())
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
        return pids
    try:
        for parent in me.parents():
            pids.add(int(parent.pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
        pass
    try:
        for child in me.children(recursive=True):
            pids.add(int(child.pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):  # pragma: no cover
        pass
    return pids


def campaign_related_process_alive(
    root_pid: int = 0,
    *,
    campaign_token: str = CAMPAIGN_TOKEN,
    runtime: Path | None = None,
    orchestrator_create_time: float | None = None,
) -> bool:
    """True if a registered runner worker or bound orchestrator/runner is alive.

    Monitor/watchdog/observer processes that only mention the campaign token are
    never treated as live campaign workers. Dead or reused orchestrator PIDs with
    missing/inaccessible identity or mismatched create_time report False.
    """
    token = str(campaign_token or CAMPAIGN_TOKEN)
    if runtime is not None:
        doc = load_worker_registry_document(runtime)
        if doc is not None:
            top = str(doc.get("campaign_token") or "").strip()
            if top and top != token:
                # Mismatched registry must not authorize liveness under the wrong campaign.
                pass
            else:
                for entry in list(doc.get("workers") or []):
                    if str(entry.get("campaign_token") or token) != token:
                        continue
                    if worker_entry_alive(entry):
                        return True
    if root_pid > 0:
        ident = _process_identity(int(root_pid))
        if ident is not None:
            if orchestrator_create_time is not None:
                if abs(float(ident["create_time"]) - float(orchestrator_create_time)) >= 1e-3:
                    ident = None
            if ident is not None and is_campaign_runner_cmdline(ident["cmdline"], token):
                return True
    hits = scan_campaign_processes(campaign_token=token)
    ignore_pids = _self_process_tree_pids()
    for h in hits:
        if int(h["pid"]) in ignore_pids:
            continue
        return True
    return False


def descendants_alive(root_pid: int, *, runtime: Path | None = None) -> bool:
    return campaign_related_process_alive(root_pid, campaign_token=CAMPAIGN_TOKEN, runtime=runtime)


def require_manifest_fields(row: dict[str, Any], required: tuple[str, ...]) -> None:
    for key in required:
        val = row.get(key)
        if val is None or val == "":
            raise RuntimeError(f"manifest_missing_required_field:{key}")


def staging_dir(binary_root: Path, phase: str, sample_id: str) -> Path:
    return binary_root / "_staging" / phase / sample_id


def manifest_dir(runtime: Path, phase: str) -> Path:
    if phase not in MANIFEST_PHASES:
        raise RuntimeError(f"manifest_phase_forbidden:{phase}")
    return runtime / "manifests" / phase


def manifest_record_path(runtime: Path, phase: str, sample_id: str) -> Path:
    return manifest_dir(runtime, phase) / "samples" / f"{sample_id}.json"


def manifest_index_path(runtime: Path, phase: str) -> Path:
    return manifest_dir(runtime, phase) / "INDEX.json"


def canonical_sample_paths(
    *,
    root: Path,
    binary: Path,
    phase: str,
    sample_id: str,
) -> tuple[str, str]:
    root_r = Path(root).resolve()
    binary_r = Path(binary).resolve()
    if phase == "skeleton_v3":
        output_rel = str((binary_r / "skeleton_v3" / f"{sample_id}.npz").relative_to(root_r)).replace(
            "\\", "/"
        )
        sidecar_rel = str(
            (binary_r / "skeleton_v3" / f"{sample_id}.recovery.json").relative_to(root_r)
        ).replace("\\", "/")
        return output_rel, sidecar_rel
    if phase == "rgb_token_v3":
        output_rel = str((binary_r / "rgb_token_v3" / f"{sample_id}.npz").relative_to(root_r)).replace(
            "\\", "/"
        )
        sidecar_rel = str(
            (binary_r / "rgb_token_v3" / f"{sample_id}.meta.json").relative_to(root_r)
        ).replace("\\", "/")
        return output_rel, sidecar_rel
    raise RuntimeError(f"canonical_paths_phase_forbidden:{phase}")


def _assert_manifest_filename_matches(path: Path, sample_id: str) -> None:
    if path.stem != sample_id:
        raise RuntimeError(f"manifest_filename_sample_id_mismatch:{path.name}:{sample_id}")


def rebuild_manifest_index(runtime: Path, phase: str) -> str:
    """Rebuild INDEX.json from create-only per-sample records (atomic temp+replace)."""
    samples_dir = manifest_dir(runtime, phase) / "samples"
    index: dict[str, dict[str, Any]] = {}
    if samples_dir.is_dir():
        for path in sorted(samples_dir.glob("*.json")):
            row = load_json(path)
            sid = row.get("sample_id") or path.stem
            if not sid:
                raise RuntimeError(f"manifest_missing_sample_id:{path}")
            _assert_manifest_filename_matches(path, str(sid))
            if sid in index:
                raise RuntimeError(f"manifest_duplicate_sample_id:{sid}")
            index[sid] = row
    payload = {
        "schema": "ms_sta_v5_d0_full_corpus_manifest_index_v1",
        "campaign_token": CAMPAIGN_TOKEN,
        "phase": phase,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_records": len(index),
        "records": index,
    }
    dest = manifest_index_path(runtime, phase)
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = dest.with_suffix(dest.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        fh.write(raw)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    return sha256_file(dest)


def commit_manifest_record(runtime: Path, phase: str, row: dict[str, Any]) -> str:
    """Create-only per-sample manifest JSON (INDEX rebuilt separately at phase end)."""
    sid = row.get("sample_id")
    if not sid:
        raise RuntimeError("manifest_missing_sample_id")
    record_path = manifest_record_path(runtime, phase, str(sid))
    _assert_manifest_filename_matches(record_path, str(sid))
    return atomic_create_json(record_path, row)


def count_manifest_records(runtime: Path, phase: str) -> int:
    """Count committed per-sample manifest JSON files (source of truth)."""
    samples_dir = manifest_dir(runtime, phase) / "samples"
    if not samples_dir.is_dir():
        return 0
    return len(list(samples_dir.glob("*.json")))


def load_manifest_index(runtime_or_path: Path, phase: str | None = None) -> dict[str, dict[str, Any]]:
    """Load per-sample manifest records. JSONL is forbidden for v5."""
    path = Path(runtime_or_path)
    if path.suffix == ".jsonl":
        raise RuntimeError("jsonl_manifest_forbidden")
    if phase is None:
        if path.name == "INDEX.json" and path.parent.name in MANIFEST_PHASES:
            phase = path.parent.name
            runtime = path.parent.parent.parent
        elif path.name in MANIFEST_PHASES:
            phase = path.name
            runtime = path.parent
        else:
            raise RuntimeError("manifest_phase_required")
    else:
        runtime = path
    samples_dir = manifest_dir(runtime, phase) / "samples"
    if not samples_dir.is_dir():
        return {}
    index: dict[str, dict[str, Any]] = {}
    for rec_path in sorted(samples_dir.glob("*.json")):
        try:
            row = load_json(rec_path)
        except json.JSONDecodeError:
            continue
        sid = row.get("sample_id") or rec_path.stem
        if not sid:
            raise RuntimeError(f"manifest_missing_sample_id:{rec_path}")
        _assert_manifest_filename_matches(rec_path, str(sid))
        if sid in index:
            raise RuntimeError(f"manifest_duplicate_sample_id:{sid}")
        index[sid] = row
    return index


def write_ready_descriptor(path: Path, obj: dict[str, Any]) -> str:
    return atomic_replace_json(path, obj)


def promote_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.replace(src, dest)


def heal_or_resume_sample(
    *,
    dest: Path,
    sidecar: Path,
    ready_path: Path,
    manifest_index: dict[str, dict[str, Any]],
    sample_id: str,
    required_keys: tuple[str, ...],
    required_manifest_fields: tuple[str, ...],
    resume: bool,
    source_sha: str,
    skel_sha: str | None = None,
) -> str | None:
    """Return output sha to skip, or None to extract. Heals crash windows; exit 2 on mismatch."""
    row = manifest_index.get(sample_id)

    # Heal: READY + staged/final outputs without manifest.
    if ready_path.is_file():
        ready = load_json(ready_path)
        require_manifest_fields(ready.get("manifest_row") or ready, required_manifest_fields)
        man_row = dict(ready.get("manifest_row") or ready)
        if man_row.get("source_sha256") != source_sha:
            raise SystemExit(2)
        if skel_sha is not None and man_row.get("skel_sha256") != skel_sha:
            raise SystemExit(2)
        bound_record_sha = ready.get("manifest_record_sha256") or man_row.get("manifest_record_sha256")
        # Ensure finals exist: promote from staging if needed.
        stage_npz = ready_path.parent / dest.name
        stage_side = ready_path.parent / sidecar.name
        if not dest.is_file() and stage_npz.is_file():
            promote_file(stage_npz, dest)
        if not sidecar.is_file() and stage_side.is_file():
            promote_file(stage_side, sidecar)
        if not dest.is_file() or not sidecar.is_file():
            raise SystemExit(2)
        if sha256_file(dest) != man_row["output_sha256"]:
            raise SystemExit(2)
        if sha256_file(sidecar) != man_row["sidecar_sha256"]:
            raise SystemExit(2)
        validate_npz_keys(dest, required_keys)
        if row is None:
            if not resume and dest.is_file():
                raise RuntimeError("existing_output_without_resume")
            return "HEAL_MANIFEST:" + man_row["output_sha256"]
        require_manifest_fields(row, required_manifest_fields)
        if row["output_sha256"] != man_row["output_sha256"]:
            raise SystemExit(2)
        if bound_record_sha is not None:
            binary_root = ready_path.parent.parent.parent.parent
            phase_name = ready_path.parent.parent.name
            rec_path = manifest_record_path(runtime_dir(binary_root), phase_name, sample_id)
            if rec_path.is_file() and sha256_file(rec_path) != bound_record_sha:
                raise SystemExit(2)
        if not resume:
            raise RuntimeError("existing_output_without_resume")
        return str(row["output_sha256"])

    if dest.is_file():
        if row is None:
            raise SystemExit(2)  # orphan final without READY/manifest
        require_manifest_fields(row, required_manifest_fields)
        if row.get("source_sha256") != source_sha:
            raise SystemExit(2)
        if skel_sha is not None and row.get("skel_sha256") != skel_sha:
            raise SystemExit(2)
        try:
            resume_output_ok(path=dest, expected_sha256=row["output_sha256"], required_keys=required_keys)
        except RuntimeError:
            raise SystemExit(2)
        if not sidecar.is_file() or sha256_file(sidecar) != row["sidecar_sha256"]:
            raise SystemExit(2)
        if not resume:
            raise RuntimeError("existing_output_without_resume")
        return str(row["output_sha256"])

    if row is not None:
        raise SystemExit(2)
    return None


def _resolve_completion_closed_at(
    *,
    evidence: Path,
    runtime: Path,
    completion: dict[str, Any],
) -> str:
    seal_path = evidence / "D0_FULL_CORPUS_COMPLETION_SEAL.json"
    intent_path = runtime / "COMPLETION_INTENT.json"
    closed_at: str | None = None
    if seal_path.is_file():
        closed_at = load_json(seal_path).get("closed_at")
    if closed_at is None and intent_path.is_file():
        closed_at = load_json(intent_path).get("closed_at")
    if closed_at is None:
        closed_at = completion.get("closed_at")
    if closed_at is None:
        closed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return str(closed_at)


def _completion_intent_payload(completion: dict[str, Any], *, closed_at: str) -> dict[str, Any]:
    return {
        "schema": "ms_sta_v5_d0_full_corpus_completion_intent_v1",
        "campaign_token": CAMPAIGN_TOKEN,
        "output_version": OUTPUT_VERSION,
        "closed_at": closed_at,
        "verdict": completion.get("verdict"),
        "n_paired": completion.get("n_paired"),
        "failed_count": completion.get("failed_count"),
        "quality_stats_sha256": completion.get("quality_stats_sha256"),
        "scan_report_sha256": completion.get("scan_report_sha256"),
        "binary_root_rel": completion.get("binary_root_rel"),
    }


def _completion_intent_matches(existing: dict[str, Any], computed: dict[str, Any]) -> bool:
    return all(existing.get(field) == computed.get(field) for field in COMPLETION_INTENT_IDENTITY_FIELDS)


def publish_completion_pack(
    *,
    evidence: Path,
    runtime: Path,
    completion: dict[str, Any],
    scan_report: dict[str, Any],
    quality_stats: dict[str, Any],
) -> dict[str, Any]:
    """Crash-healable completion transaction with COMPLETION_INTENT + COMPLETION_READY SHA binding."""
    seal_path = evidence / "D0_FULL_CORPUS_COMPLETION_SEAL.json"
    intent_path = runtime / "COMPLETION_INTENT.json"
    completion_ready_path = runtime / "COMPLETION_READY.json"
    closed_at = _resolve_completion_closed_at(evidence=evidence, runtime=runtime, completion=completion)

    stage = runtime / "_staging_completion"
    stage.mkdir(parents=True, exist_ok=True)
    stage_stats = stage / "QUALITY_STATS.json"
    stage_scan = stage / "SCAN_REPORT.json"
    stage_comp = stage / "COMPLETION.json"
    atomic_replace_json(stage_stats, quality_stats)
    atomic_replace_json(stage_scan, scan_report)
    completion = dict(completion)
    completion["closed_at"] = closed_at
    atomic_replace_json(stage_comp, completion)
    stats_sha = sha256_file(stage_stats)
    scan_sha = sha256_file(stage_scan)
    completion["quality_stats_sha256"] = stats_sha
    completion["scan_report_sha256"] = scan_sha
    atomic_replace_json(stage_comp, completion)
    comp_sha = sha256_file(stage_comp)
    intent_obj = _completion_intent_payload(completion, closed_at=closed_at)

    if seal_path.is_file():
        existing = load_json(seal_path)
        for field in COMPLETION_BOUND_FIELDS:
            if existing.get(field) != completion.get(field):
                raise SystemExit(2)
        if intent_path.is_file():
            intent_existing = load_json(intent_path)
            if not _completion_intent_matches(intent_existing, intent_obj):
                raise SystemExit(2)
        else:
            atomic_create_json(intent_path, intent_obj)
        atomic_replace_json(runtime / "QUALITY_STATS.json", quality_stats)
        atomic_replace_json(runtime / "SCAN_REPORT.json", scan_report)
        atomic_replace_json(runtime / "COMPLETION.json", existing)
        ready_obj = {
            "schema": "ms_sta_v5_d0_full_corpus_completion_ready_v1",
            "campaign_token": CAMPAIGN_TOKEN,
            "quality_stats_sha256": stats_sha,
            "scan_report_sha256": scan_sha,
            "completion_sha256": sha256_file(runtime / "COMPLETION.json"),
            "seal_sha256": sha256_file(seal_path),
            "intent_sha256": sha256_file(intent_path),
            "bound_fields": {k: existing.get(k) for k in COMPLETION_BOUND_FIELDS},
        }
        atomic_replace_json(completion_ready_path, ready_obj)
        return existing

    if intent_path.is_file():
        intent_existing = load_json(intent_path)
        if not _completion_intent_matches(intent_existing, intent_obj):
            raise SystemExit(2)
    else:
        atomic_create_json(intent_path, intent_obj)

    promote_file(stage_stats, runtime / "QUALITY_STATS.json")
    atomic_replace_json(runtime / "QUALITY_STATS.json", quality_stats)
    atomic_replace_json(runtime / "SCAN_REPORT.json", scan_report)
    atomic_replace_json(runtime / "COMPLETION.json", completion)
    atomic_create_json(seal_path, completion)
    seal_sha = sha256_file(seal_path)
    ready_obj = {
        "schema": "ms_sta_v5_d0_full_corpus_completion_ready_v1",
        "campaign_token": CAMPAIGN_TOKEN,
        "quality_stats_sha256": sha256_file(runtime / "QUALITY_STATS.json"),
        "scan_report_sha256": sha256_file(runtime / "SCAN_REPORT.json"),
        "completion_sha256": sha256_file(runtime / "COMPLETION.json"),
        "seal_sha256": seal_sha,
        "intent_sha256": sha256_file(intent_path),
        "bound_fields": {k: completion.get(k) for k in COMPLETION_BOUND_FIELDS},
    }
    atomic_replace_json(completion_ready_path, ready_obj)
    return completion


def assert_psutil_available() -> None:
    try:
        import psutil  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"psutil_required_pin:{PSUTIL_PIN}") from exc
    # version pin soft-check
    import psutil

    if not str(psutil.__version__).startswith("7.2."):
        raise RuntimeError(f"psutil_version_unexpected:{psutil.__version__}")


def validate_array_contracts(npz_path: Path, kind: str = "skel") -> dict[str, Any]:
    """Enforce per-array ndim/dtype/alignment contracts for skeleton or rgb NPZ."""
    import numpy as np

    contract = SKEL_ARRAY_CONTRACT if kind == "skel" else RGB_ARRAY_CONTRACT
    if kind not in ("skel", "rgb"):
        raise RuntimeError(f"array_contract_kind_forbidden:{kind}")
    data = np.load(npz_path, allow_pickle=False)
    info: dict[str, Any] = {"kind": kind, "arrays": {}}
    align_lens: dict[str, int] = {}
    token_shape: tuple[int, ...] | None = None
    for key, spec in contract.items():
        if key not in data.files:
            raise RuntimeError(f"array_contract_missing_key:{key}")
        arr = data[key]
        if spec.get("scalar"):
            flat = np.asarray(arr).reshape(-1)
            if flat.size != 1:
                raise RuntimeError(f"array_contract_scalar_multi:{key}")
            if "dtype" in spec and arr.dtype != np.dtype(spec["dtype"]):
                raise RuntimeError(f"array_contract_dtype:{key}:{arr.dtype}")
            info["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "scalar": int(flat[0])}
            continue
        if arr.ndim != int(spec["ndim"]):
            raise RuntimeError(f"array_contract_ndim:{key}:{arr.ndim}")
        if "dtype" in spec:
            want_dtype = np.dtype(spec["dtype"])
            if arr.dtype != want_dtype:
                raise RuntimeError(f"array_contract_dtype:{key}:{arr.dtype}")
        group = spec.get("align_group")
        if group is not None:
            axis_len = int(arr.shape[0])
            prev = align_lens.get(str(group))
            if prev is None:
                align_lens[str(group)] = axis_len
            elif axis_len != prev:
                raise RuntimeError(f"array_contract_alignment:{key}")
        if kind == "rgb" and key.startswith("token_"):
            if token_shape is None:
                token_shape = tuple(arr.shape)
            elif tuple(arr.shape) != token_shape:
                raise RuntimeError(f"array_contract_token_shape:{key}")
        info["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if kind == "skel":
        from src.data_prep.joint_layout_v3 import JOINT_CHIRALITY, JOINT_PARTS, bone_edge_list

        t_len = align_lens.get("frames")
        if t_len is None or int(t_len) < 1:
            raise RuntimeError("array_contract_frame_T")
        for key in ("coordinates", "joints_raw", "raw_landmarks", "effective_landmarks", "velocity"):
            arr = data[key]
            if arr.shape[1] != SKEL_JOINT_COUNT or arr.shape[2] != 3:
                raise RuntimeError(f"array_contract_joint_dims:{key}")
        for key in ("missing_mask_prefill", "raw_holistic_missing", "effective_missing", "validity", "confidence"):
            arr = data[key]
            if arr.shape[1] != SKEL_JOINT_COUNT:
                raise RuntimeError(f"array_contract_mask_dims:{key}")
        canon_part = np.asarray(JOINT_PARTS)
        canon_chir = np.asarray(JOINT_CHIRALITY)
        canon_edges = np.asarray(bone_edge_list(), dtype=np.int32)
        joint_part = np.asarray(data["joint_part"])
        joint_chirality = np.asarray(data["joint_chirality"])
        if joint_part.shape != (SKEL_JOINT_COUNT,) or not np.array_equal(joint_part, canon_part):
            raise RuntimeError("array_contract_joint_part_identity")
        if joint_chirality.shape != (SKEL_JOINT_COUNT,) or not np.array_equal(joint_chirality, canon_chir):
            raise RuntimeError("array_contract_joint_chirality_identity")
        bone_edges = np.asarray(data["bone_edges"], dtype=np.int32)
        if bone_edges.ndim != 2 or bone_edges.shape[1] != 2:
            raise RuntimeError("array_contract_bone_edges")
        if not np.array_equal(bone_edges, canon_edges):
            raise RuntimeError("array_contract_bone_edges_identity")
        n_bones = int(canon_edges.shape[0])
        bv = data["bone_vectors"]
        if bv.shape[0] != t_len or bv.shape[1] != n_bones or bv.shape[2] != 3:
            raise RuntimeError("array_contract_bone_vectors")
        if not np.isfinite(bv).all():
            raise RuntimeError("array_contract_bone_vectors_nonfinite")
        seq_flat = np.asarray(data["sequence_length"], dtype=np.int32).reshape(-1)
        if seq_flat.size != 1 or int(seq_flat[0]) != int(t_len):
            raise RuntimeError("array_contract_sequence_length")
        ofi = np.asarray(data["original_frame_index"], dtype=np.int32)
        if ofi.size < 1:
            raise RuntimeError("array_contract_frame_index_empty")
        if not np.all(np.isfinite(ofi)):
            raise RuntimeError("array_contract_frame_index_nonfinite")
        if ofi.size > 1 and not np.all(ofi[1:] > ofi[:-1]):
            raise RuntimeError("array_contract_frame_index_not_strictly_increasing")
        ts = np.asarray(data["timestamps_sec"], dtype=np.float64)
        if ts.shape[0] != int(t_len):
            raise RuntimeError("array_contract_timestamps_len")
        if not np.all(np.isfinite(ts)):
            raise RuntimeError("array_contract_timestamps_nonfinite")
        if ts.size > 1 and not np.all(ts[1:] >= ts[:-1]):
            raise RuntimeError("array_contract_timestamps_not_monotonic")
        pad = np.asarray(data["padding_mask"], dtype=bool)
        if pad.shape != (int(t_len),) or not np.all(pad):
            raise RuntimeError("array_contract_padding_mask_not_all_true")
    if kind == "rgb":
        window_l = align_lens.get("windows")
        if window_l is None or int(window_l) < 1:
            raise RuntimeError("array_contract_window_L")
        for key in contract:
            if key.startswith("token_"):
                if int(data[key].shape[1]) != RGB_TOKEN_DIM:
                    raise RuntimeError(f"array_contract_token_dim:{key}")
        if int(data["pathway_slow"].shape[1]) != RGB_PATHWAY_SLOW_DIM:
            raise RuntimeError("array_contract_pathway_slow_dim")
        if int(data["pathway_fast"].shape[1]) != RGB_PATHWAY_FAST_DIM:
            raise RuntimeError("array_contract_pathway_fast_dim")
        ofi = np.asarray(data["original_frame_index"])
        if ofi.size < 1:
            raise RuntimeError("array_contract_frame_index_empty")
        if not np.all(np.isfinite(ofi)):
            raise RuntimeError("array_contract_frame_index_nonfinite")
        if ofi.size > 1 and not np.all(ofi[1:] >= ofi[:-1]):
            raise RuntimeError("array_contract_frame_index_not_monotonic")
    return info


def atomic_savez_compressed(dest: Path, **arrays: Any) -> str:
    """Write NPZ via binary handle (avoids NumPy double-.npz suffix), validate, os.replace."""
    import numpy as np

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{dest.stem}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        # Caller validates required keys before/after; here ensure readable.
        with np.load(tmp, allow_pickle=False) as _:
            pass
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return sha256_file(dest)


def validate_npz_keys(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    import numpy as np

    data = np.load(path, allow_pickle=False)
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"npz_missing_keys:{missing}")
    info = {"keys": list(data.files), "arrays": {}}
    for k in required:
        arr = data[k]
        if not np.isfinite(arr).all() if np.issubdtype(arr.dtype, np.floating) else True:
            if np.issubdtype(arr.dtype, np.floating) and not np.isfinite(arr).all():
                raise RuntimeError(f"npz_nonfinite:{k}")
        info["arrays"][k] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        }
    return info


def resume_output_ok(
    *,
    path: Path,
    expected_sha256: str | None,
    required_keys: tuple[str, ...],
) -> bool:
    if not path.is_file():
        return False
    dig = sha256_file(path)
    if expected_sha256 is not None and dig != expected_sha256:
        raise RuntimeError(f"resume_sha_mismatch:{path.name}")
    validate_npz_keys(path, required_keys)
    return True


def disk_preflight(min_free_gb: float, *, root: Path = _ROOT) -> dict[str, Any]:
    usage = shutil.disk_usage(str(root))
    free_gb = usage.free / (1024**3)
    ok = free_gb >= float(min_free_gb)
    return {
        "ok": ok,
        "min_free_gb": float(min_free_gb),
        "observed_free_gb": round(free_gb, 2),
    }


def default_quality_thresholds() -> dict[str, float]:
    return dict(DEFAULT_QUALITY_THRESHOLDS)


def compute_quality_stats_from_rgb_tensors(
    *,
    rgb_dir: Path,
    sample_ids: list[str],
    split_by_id: dict[str, str],
    signer_by_id: dict[str, str],
) -> dict[str, Any]:
    """Compute missing/partial rates from RGB tensor missing_*/partial_* arrays.

    Denominator: count of (sample, window, region) triples from tensor window axis.
    Meta JSON is not used for rates.
    """
    import numpy as np

    regions = ("face", "left_hand", "right_hand")
    by_region = {r: {"missing": 0, "partial": 0, "total": 0} for r in regions}
    by_split: dict[str, dict[str, dict[str, int]]] = {}
    by_signer: dict[str, dict[str, dict[str, int]]] = {}

    def _bucket(store: dict, key: str) -> dict[str, dict[str, int]]:
        if key not in store:
            store[key] = {r: {"missing": 0, "partial": 0, "total": 0} for r in regions}
        return store[key]

    for sid in sample_ids:
        path = rgb_dir / f"{sid}.npz"
        data = np.load(path, allow_pickle=False)
        split = split_by_id.get(sid, "unknown")
        signer = signer_by_id.get(sid, "unknown")
        n_win = None
        for r in regions:
            miss = np.asarray(data[f"missing_{r}"], dtype=bool)
            part = np.asarray(data[f"partial_{r}"], dtype=bool)
            if miss.ndim != 1 or part.ndim != 1 or miss.shape != part.shape:
                raise RuntimeError(f"rgb_mask_shape_invalid:{sid}:{r}")
            if n_win is None:
                n_win = int(miss.shape[0])
            elif int(miss.shape[0]) != n_win:
                raise RuntimeError(f"rgb_window_axis_mismatch:{sid}:{r}")
            for i in range(n_win):
                by_region[r]["total"] += 1
                _bucket(by_split, split)[r]["total"] += 1
                _bucket(by_signer, signer)[r]["total"] += 1
                if bool(miss[i]):
                    by_region[r]["missing"] += 1
                    _bucket(by_split, split)[r]["missing"] += 1
                    _bucket(by_signer, signer)[r]["missing"] += 1
                if bool(part[i]):
                    by_region[r]["partial"] += 1
                    _bucket(by_split, split)[r]["partial"] += 1
                    _bucket(by_signer, signer)[r]["partial"] += 1

    def _rate(num: int, den: int) -> float:
        return float(num) / float(den) if den else 0.0

    missing_by_region = {r: _rate(by_region[r]["missing"], by_region[r]["total"]) for r in regions}
    partial_by_region = {r: _rate(by_region[r]["partial"], by_region[r]["total"]) for r in regions}
    missing_by_split = {
        sp: {r: _rate(v[r]["missing"], v[r]["total"]) for r in regions} for sp, v in by_split.items()
    }
    missing_by_signer = {
        sg: {r: _rate(v[r]["missing"], v[r]["total"]) for r in regions} for sg, v in by_signer.items()
    }
    return {
        "denominator": "window_region_triples_from_rgb_tensor_masks",
        "source_of_truth": "rgb_token_v3_npz_missing_and_partial_arrays",
        "counts_by_region": by_region,
        "missing_roi_rate_by_region_from_tensor_validity": missing_by_region,
        "missing_roi_rate_by_split_from_tensor_validity": missing_by_split,
        "missing_roi_rate_by_signer_from_tensor_validity": missing_by_signer,
        "partial_missing_rate_by_region": partial_by_region,
    }


def assert_meta_tensor_agreement(*, rgb_path: Path, meta: dict[str, Any], sample_id: str) -> None:
    """Fail-closed: region_validity all_missing/partial_missing must match tensor masks.

    Canonical metadata source is windows[].region_validity[region].{all_missing,partial_missing}.
    Legacy windows[].missing_region_mask is optional; when present it must have exact keys
    {face,left_hand,right_hand}, each value type(bool), and each must agree with tensors.
    n_slices must be > 0 and 0 <= n_valid <= n_slices.
    """
    import numpy as np

    data = np.load(rgb_path, allow_pickle=False)
    windows = meta.get("windows") or []
    if not isinstance(windows, list):
        raise RuntimeError(f"meta_windows_not_list:{sample_id}")
    regions = ("face", "left_hand", "right_hand")
    expected_legacy_keys = set(regions)
    miss_by = {r: np.asarray(data[f"missing_{r}"], dtype=bool) for r in regions}
    part_by = {r: np.asarray(data[f"partial_{r}"], dtype=bool) for r in regions}
    n_win = len(windows)
    for r in regions:
        if n_win != int(miss_by[r].shape[0]) or n_win != int(part_by[r].shape[0]):
            raise RuntimeError(f"meta_window_count_mismatch:{sample_id}:{r}")

    for i, w in enumerate(windows):
        if not isinstance(w, dict):
            raise RuntimeError(f"meta_window_not_object:{sample_id}:{i}")
        rv_root = w.get("region_validity")
        if not isinstance(rv_root, dict):
            raise RuntimeError(f"meta_region_validity_missing:{sample_id}:{i}")
        for r in regions:
            rv = rv_root.get(r)
            if not isinstance(rv, dict):
                raise RuntimeError(f"meta_region_validity_region_missing:{sample_id}:{r}:{i}")
            if "all_missing" not in rv or "partial_missing" not in rv:
                raise RuntimeError(f"meta_region_validity_flags_missing:{sample_id}:{r}:{i}")
            all_m = rv.get("all_missing")
            part_m = rv.get("partial_missing")
            if type(all_m) is not bool or type(part_m) is not bool:
                raise RuntimeError(f"meta_region_validity_flags_bad_type:{sample_id}:{r}:{i}")
            if "n_valid" not in rv or "n_slices" not in rv:
                raise RuntimeError(f"meta_region_validity_counts_missing:{sample_id}:{r}:{i}")
            n_valid = rv.get("n_valid")
            n_slices = rv.get("n_slices")
            if type(n_valid) is not int or type(n_slices) is not int:
                raise RuntimeError(f"meta_region_validity_counts_bad_type:{sample_id}:{r}:{i}")
            if n_slices <= 0 or n_valid < 0 or n_valid > n_slices:
                raise RuntimeError(f"meta_region_validity_counts_inconsistent:{sample_id}:{r}:{i}")
            expect_all = n_valid == 0
            expect_partial = 0 < n_valid < n_slices
            if all_m != expect_all:
                raise RuntimeError(
                    f"meta_region_validity_all_missing_vs_counts:{sample_id}:{r}:{i}"
                )
            if part_m != expect_partial:
                raise RuntimeError(
                    f"meta_region_validity_partial_vs_counts:{sample_id}:{r}:{i}"
                )
            if bool(all_m) != bool(miss_by[r][i]):
                raise RuntimeError(f"meta_tensor_missing_mismatch:{sample_id}:{r}:{i}")
            if bool(part_m) != bool(part_by[r][i]):
                raise RuntimeError(f"meta_tensor_partial_mismatch:{sample_id}:{r}:{i}")

        legacy = w.get("missing_region_mask", None)
        if legacy is not None:
            if not isinstance(legacy, dict):
                raise RuntimeError(
                    f"meta_legacy_missing_region_mask_bad_type:{sample_id}:{i}"
                )
            if set(legacy.keys()) != expected_legacy_keys:
                raise RuntimeError(
                    f"meta_legacy_missing_region_mask_keys:{sample_id}:{i}"
                )
            for r in regions:
                val = legacy[r]
                if type(val) is not bool:
                    raise RuntimeError(
                        f"meta_legacy_missing_region_mask_bad_type:{sample_id}:{r}:{i}"
                    )
                if bool(val) != bool(miss_by[r][i]):
                    raise RuntimeError(
                        f"meta_legacy_missing_region_mask_mismatch:{sample_id}:{r}:{i}"
                    )
    if meta.get("sample_id") not in (None, sample_id):
        raise RuntimeError(f"meta_sample_id_mismatch:{sample_id}")


def compute_quality_stats_from_rgb_metas(
    meta_rows: list[dict[str, Any]],
    *,
    split_by_id: dict[str, str],
    signer_by_id: dict[str, str],
) -> dict[str, Any]:
    """Deprecated for gate decisions — retained only for negative tests that meta cannot drive rates.

    Raises if called for authoritative gating; use compute_quality_stats_from_rgb_tensors.
    """
    raise RuntimeError("quality_stats_must_use_tensors_not_meta")


def evaluate_quality_thresholds(
    stats: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    thr = dict(thresholds)
    # Scanner must not mutate caller thresholds.
    missing = stats["missing_roi_rate_by_region_from_tensor_validity"]
    partial = stats["partial_missing_rate_by_region"]
    checks = {
        "face": missing.get("face", 1.0) <= thr["max_missing_roi_rate_face"],
        "left_hand": missing.get("left_hand", 1.0) <= thr["max_missing_roi_rate_left_hand"],
        "right_hand": missing.get("right_hand", 1.0) <= thr["max_missing_roi_rate_right_hand"],
        "partial_any": all(
            partial.get(r, 1.0) <= thr["max_partial_missing_rate_any_region"] for r in ("face", "left_hand", "right_hand")
        ),
    }
    return {
        "thresholds_used": thr,
        "checks": checks,
        "ok": all(checks.values()),
    }


def validate_execution_protocol_v4(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v4":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V4:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V4
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V4_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V4:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V4,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v4(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v4":
        errors.append("bad_approval_v4_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v4_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v4":
        errors.append("approval_v4_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v4_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v4_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V4:
        errors.append("approval_v4_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v4_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v4_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v4_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v4_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v4_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v4_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v4_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v4_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v4_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v4_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v4_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v4_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v4_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v4_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v4_sample_tx")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            list(obj.get("executable_paths") or FULL_CORPUS_EXECUTABLE_PATHS_V4),
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V4):
        errors.append("approval_v4_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V4_REL:
        errors.append("approval_v4_evidence_root")
    if obj.get("immutable_v3_approval_sha256") != EXPECTED_V3_APPROVAL_SHA256:
        errors.append("approval_v4_v3_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V3_REL) != EXPECTED_V3_APPROVAL_SHA256:
        errors.append("v3_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v3_sha256") != EXPECTED_V3_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v4_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V3_REL) != EXPECTED_V3_EXTERNAL_REAUDIT_SHA256:
        errors.append("v3_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v4(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v4":
        errors.append("bad_run_contract_v4_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v4_status")
    if obj.get("output_version") != OUTPUT_VERSION_V4:
        errors.append("run_contract_v4_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v4_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v4_autochain")
    if obj.get("approval_v4_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V4_REL):
        errors.append("run_contract_v4_approval_sha")
    if obj.get("execution_protocol_v4_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V4_REL):
        errors.append("run_contract_v4_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v5(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v5":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V5:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V5
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V5_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V5:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V5,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v5(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v5":
        errors.append("bad_approval_v5_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v5_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v5":
        errors.append("approval_v5_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v5_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v5_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V5:
        errors.append("approval_v5_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v5_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v5_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v5_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v5_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v5_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v5_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v5_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v5_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v5_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v5_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v5_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v5_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v5_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v5_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v5_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v5_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            list(obj.get("executable_paths") or FULL_CORPUS_EXECUTABLE_PATHS_V5),
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V5):
        errors.append("approval_v5_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V5_REL:
        errors.append("approval_v5_evidence_root")
    if obj.get("immutable_v4_approval_sha256") != EXPECTED_V4_APPROVAL_SHA256:
        errors.append("approval_v5_v4_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V4_REL) != EXPECTED_V4_APPROVAL_SHA256:
        errors.append("v4_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v4_sha256") != EXPECTED_V4_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v5_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V4_REL) != EXPECTED_V4_EXTERNAL_REAUDIT_SHA256:
        errors.append("v4_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v5(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v5":
        errors.append("bad_run_contract_v5_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v5_status")
    if obj.get("output_version") != OUTPUT_VERSION_V5:
        errors.append("run_contract_v5_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v5_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v5_autochain")
    if obj.get("approval_v5_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V5_REL):
        errors.append("run_contract_v5_approval_sha")
    if obj.get("execution_protocol_v5_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V5_REL):
        errors.append("run_contract_v5_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v6(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v6":
        errors.append("bad_run_contract_v6_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v6_status")
    if obj.get("output_version") != OUTPUT_VERSION_V6:
        errors.append("run_contract_v6_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v6_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v6_autochain")
    if obj.get("approval_v6_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V6_REL):
        errors.append("run_contract_v6_approval_sha")
    if obj.get("execution_protocol_v6_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V6_REL):
        errors.append("run_contract_v6_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v6(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v6":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V6:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V6
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V6_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V6:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V6,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v6(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v6":
        errors.append("bad_approval_v6_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v6_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v6":
        errors.append("approval_v6_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v6_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v6_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V6:
        errors.append("approval_v6_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v6_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v6_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v6_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v6_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v6_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v6_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v6_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v6_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v6_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v6_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v6_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v6_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v6_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v6_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v6_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v6_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V6,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V6):
        errors.append("approval_v6_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V6_REL:
        errors.append("approval_v6_evidence_root")
    if obj.get("immutable_v5_approval_sha256") != EXPECTED_V5_APPROVAL_SHA256:
        errors.append("approval_v6_v5_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V5_REL) != EXPECTED_V5_APPROVAL_SHA256:
        errors.append("v5_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v5_sha256") != EXPECTED_V5_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v6_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V5_REL) != EXPECTED_V5_EXTERNAL_REAUDIT_SHA256:
        errors.append("v5_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v7(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v7":
        errors.append("bad_run_contract_v7_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v7_status")
    if obj.get("output_version") != OUTPUT_VERSION_V7:
        errors.append("run_contract_v7_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v7_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v7_autochain")
    if obj.get("approval_v7_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V7_REL):
        errors.append("run_contract_v7_approval_sha")
    if obj.get("execution_protocol_v7_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V7_REL):
        errors.append("run_contract_v7_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v7(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v7":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V7:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V7
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V7_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V7:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V7,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v7(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v7":
        errors.append("bad_approval_v7_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v7_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v7":
        errors.append("approval_v7_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v7_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v7_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V7:
        errors.append("approval_v7_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v7_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v7_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v7_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v7_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v7_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v7_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v7_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v7_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v7_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v7_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v7_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v7_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v7_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v7_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v7_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v7_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V7,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V7):
        errors.append("approval_v7_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V7_REL:
        errors.append("approval_v7_evidence_root")
    if obj.get("immutable_v6_approval_sha256") != EXPECTED_V6_APPROVAL_SHA256:
        errors.append("approval_v7_v6_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V6_REL) != EXPECTED_V6_APPROVAL_SHA256:
        errors.append("v6_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v6_sha256") != EXPECTED_V6_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v7_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V6_REL) != EXPECTED_V6_EXTERNAL_REAUDIT_SHA256:
        errors.append("v6_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_external_reaudit_pass(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != EXTERNAL_REAUDIT_PASS_SCHEMA:
        errors.append("external_reaudit_bad_schema")
    verdict = obj.get("audit_verdict")
    if verdict != "PASS":
        errors.append(f"external_reaudit_not_pass:{verdict}")
    if obj.get("auditor") != "independent":
        errors.append("external_reaudit_auditor")
    if obj.get("worktree_clean_before_audit_artifact") is not True:
        errors.append("external_reaudit_worktree_not_clean")
    head = obj.get("head_at_audit")
    origin = obj.get("origin_main_at_audit")
    if not (isinstance(head, str) and _HEX40.match(head)):
        errors.append("external_reaudit_head_at_audit_invalid")
    if not (isinstance(origin, str) and _HEX40.match(origin)):
        errors.append("external_reaudit_origin_main_at_audit_invalid")
    elif head != origin:
        errors.append("external_reaudit_head_origin_mismatch")
    elif isinstance(head, str) and _HEX40.match(head) and not git_commit_exists(head, root=root):
        errors.append("external_reaudit_head_not_found")

    approval_path = root / D0_FULL_CORPUS_APPROVAL_V14_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL
    run_contract_path = root / D0_FULL_CORPUS_RUN_CONTRACT_V14_REL
    approval_audit_path = root / D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL
    parent_reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL

    approval_code_lock: str | None = None
    if not approval_path.is_file():
        errors.append("approval_v14_missing")
    else:
        try:
            approval = load_json(approval_path)
            approval_code_lock = approval.get("code_lock_git_revision")
            if obj.get("code_lock_git_revision") != approval_code_lock:
                errors.append("external_reaudit_code_lock_mismatch")
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v14_unreadable")

    head_rev = head if isinstance(head, str) and _HEX40.match(head) else None
    for field, path, rel in (
        ("approval_v14_sha256", approval_path, D0_FULL_CORPUS_APPROVAL_V14_REL),
        ("execution_protocol_v14_sha256", protocol_path, D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL),
        ("run_contract_v14_sha256", run_contract_path, D0_FULL_CORPUS_RUN_CONTRACT_V14_REL),
        ("approval_audit_v14_sha256", approval_audit_path, D0_FULL_CORPUS_APPROVAL_AUDIT_V14_REL),
    ):
        expected = obj.get(field)
        if not path.is_file():
            errors.append(f"{field.replace('_sha256', '')}_missing")
            continue
        disk_sha = sha256_file(path)
        if expected != disk_sha:
            errors.append(f"{field}_mismatch")
            continue
        if head_rev and git_commit_exists(head_rev, root=root):
            try:
                git_sha = sha256_git_blob_at_revision(rel, head_rev, root=root)
            except FileNotFoundError:
                errors.append(f"external_reaudit_git_blob_missing:{field}")
            else:
                if expected != git_sha:
                    errors.append(f"external_reaudit_git_blob_mismatch:{field}")

    if obj.get("parent_external_reaudit_v13_sha256") != EXPECTED_V13_EXTERNAL_REAUDIT_SHA256:
        errors.append("external_reaudit_parent_v13_sha_mismatch")
    elif not parent_reaudit_path.is_file():
        errors.append("parent_external_reaudit_v13_missing")
    elif sha256_file(parent_reaudit_path) != EXPECTED_V13_EXTERNAL_REAUDIT_SHA256:
        errors.append("v13_reaudit_bytes_drift")

    blocking = obj.get("blocking_findings")
    if not isinstance(blocking, list):
        errors.append("external_reaudit_blocking_findings_not_list")
    elif blocking:
        errors.append("external_reaudit_blocking_findings_nonempty")

    decision = obj.get("decision")
    if not isinstance(decision, dict):
        errors.append("external_reaudit_decision_missing")
    else:
        for key, required in EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION.items():
            if key not in decision:
                errors.append(f"external_reaudit_decision_missing:{key}")
            elif decision[key] is not required:
                if required is True:
                    errors.append("external_reaudit_post_audit_not_allowed")
                else:
                    errors.append(f"external_reaudit_{key}_must_be_false")

    output_version = obj.get("output_version")
    if output_version is not None and output_version != OUTPUT_VERSION:
        errors.append("external_reaudit_output_version_mismatch")
    campaign = obj.get("campaign_token")
    if campaign is not None and campaign != CAMPAIGN_TOKEN:
        errors.append("external_reaudit_campaign_token_mismatch")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_interruption_v10(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_interruption_v10":
        errors.append("bad_execution_interruption_v10_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V10:
        errors.append("execution_interruption_v10_version")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V10:
        errors.append("execution_interruption_v10_campaign")
    if obj.get("status") != "INTERRUPTED_NON_CANONICAL_RUNTIME_PATCH":
        errors.append("execution_interruption_v10_status")
    if obj.get("canonical") is not False:
        errors.append("execution_interruption_v10_canonical_must_be_false")
    path = root / D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL
    if path.is_file() and sha256_file(path) != EXPECTED_V10_INTERRUPTION_SHA256:
        errors.append("execution_interruption_v10_sha_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v14":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V14_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
    if not approval_path.is_file():
        errors.append("approval_v14_missing")
    elif obj.get("approval_v14_sha256") != sha256_file(approval_path):
        errors.append("approval_v14_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v14_missing")
    elif obj.get("execution_protocol_v14_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v14_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v14_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    else:
        try:
            validate_external_reaudit_pass(load_json(reaudit_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v14_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v13(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v13":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V13:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V13:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V13_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL
    if not approval_path.is_file():
        errors.append("approval_v13_missing")
    elif obj.get("approval_v13_sha256") != sha256_file(approval_path):
        errors.append("approval_v13_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v13_missing")
    elif obj.get("execution_protocol_v13_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v13_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v13_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V13,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v13_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v12(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v12":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V12:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V12:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V12_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V12_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V12_REL
    if not approval_path.is_file():
        errors.append("approval_v12_missing")
    elif obj.get("approval_v12_sha256") != sha256_file(approval_path):
        errors.append("approval_v12_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v12_missing")
    elif obj.get("execution_protocol_v12_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v12_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v12_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V12,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v12_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v11(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v11":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V11:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V11:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V11_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V11_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V11_REL
    if not approval_path.is_file():
        errors.append("approval_v11_missing")
    elif obj.get("approval_v11_sha256") != sha256_file(approval_path):
        errors.append("approval_v11_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v11_missing")
    elif obj.get("execution_protocol_v11_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v11_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v11_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V11,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v11_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v10(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v10":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V10:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V10:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V10_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V10_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V10_REL
    if not approval_path.is_file():
        errors.append("approval_v10_missing")
    elif obj.get("approval_v10_sha256") != sha256_file(approval_path):
        errors.append("approval_v10_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v10_missing")
    elif obj.get("execution_protocol_v10_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v10_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v10_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V10,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v10_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v9(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v9":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V9:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V9:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V9_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V9_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V9_REL
    if not approval_path.is_file():
        errors.append("approval_v9_missing")
    elif obj.get("approval_v9_sha256") != sha256_file(approval_path):
        errors.append("approval_v9_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v9_missing")
    elif obj.get("execution_protocol_v9_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v9_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v9_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V9,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v9_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def evaluate_launch_readiness(*, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed readiness: approval v14 + protocol v14 + external PASS + post-audit auth.

    Sealed v10/v11/v12/v13 artifacts alone never make this ready.
    """
    errors: list[str] = []
    approval: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    try:
        try:
            assert_no_out_of_provenance_sitecustomize(root=root)
        except RuntimeError as exc:
            errors.append(str(exc))
        approval_path = root / D0_FULL_CORPUS_APPROVAL_V14_REL
        protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL
        if not approval_path.is_file():
            errors.append("approval_v14_missing")
        else:
            approval = load_json(approval_path)
            validate_approval_v14(approval, root=root)
        if not protocol_path.is_file():
            errors.append("protocol_v14_missing")
        else:
            protocol = load_json(protocol_path)
            validate_execution_protocol_v14(protocol, root=root)
        reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL
        if not reaudit_path.is_file():
            errors.append("external_reaudit_v14_missing")
        else:
            try:
                validate_external_reaudit_pass(load_json(reaudit_path), root=root)
            except ValueError as exc:
                errors.append(str(exc))
        auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL
        if not auth_path.is_file():
            errors.append("post_audit_execution_auth_missing")
        else:
            try:
                validate_post_audit_execution_authorization(load_json(auth_path), root=root)
            except ValueError as exc:
                errors.append(str(exc))
    except ValueError as exc:
        errors.append(str(exc))
    ready = len(errors) == 0
    return {
        "ready": ready,
        "errors": errors,
        "output_version": OUTPUT_VERSION,
        "campaign_token": CAMPAIGN_TOKEN,
        "approval": approval,
        "protocol": protocol,
    }


def require_launch_readiness(*, root: Path = _ROOT) -> dict[str, Any]:
    assert_no_out_of_provenance_sitecustomize(root=root)
    status = evaluate_launch_readiness(root=root)
    if not status["ready"]:
        raise RuntimeError("launch_gates_blocked:" + ";".join(status["errors"]))
    return status


def evaluate_launch_readiness_v15(*, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed v15 readiness: sealed approval + protocol + external PASS + post-auth.

    Draft approvals are rejected. Does not authorize M0/train. Never draft-friendly for launch.
    """
    errors: list[str] = []
    approval: dict[str, Any] | None = None
    protocol: dict[str, Any] | None = None
    try:
        try:
            assert_no_out_of_provenance_sitecustomize(root=root)
        except RuntimeError as exc:
            errors.append(str(exc))

        approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
        protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
        if not approval_path.is_file():
            errors.append("approval_v15_missing")
        else:
            try:
                approval = load_json(approval_path)
                status = str(approval.get("status") or "")
                decision = str(approval.get("decision") or "")
                if "DRAFT" in status.upper() or decision == "DRAFT_NOT_APPROVED":
                    errors.append("approval_v15_draft_rejected")
                else:
                    try:
                        validate_approval_v15(approval, root=root)
                    except ValueError as exc:
                        errors.append(str(exc))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"approval_v15_unreadable:{exc}")

        if not protocol_path.is_file():
            errors.append("protocol_v15_missing")
        else:
            try:
                protocol = load_json(protocol_path)
                try:
                    validate_execution_protocol_v15(protocol, root=root)
                except ValueError as exc:
                    errors.append(str(exc))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"protocol_v15_unreadable:{exc}")

        reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
        if not reaudit_path.is_file():
            errors.append("external_reaudit_v15_missing")
        else:
            try:
                validate_external_reaudit_pass_v15(load_json(reaudit_path), root=root)
            except ValueError as exc:
                errors.append(str(exc))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"external_reaudit_v15_unreadable:{exc}")

        auth_path = root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_V15_REL
        if not auth_path.is_file():
            errors.append("post_audit_execution_auth_v15_missing")
        else:
            try:
                validate_post_audit_execution_authorization_v15(load_json(auth_path), root=root)
            except ValueError as exc:
                errors.append(str(exc))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"post_audit_execution_auth_v15_unreadable:{exc}")

        carry_path = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
        if not carry_path.is_file():
            errors.append("carry_forward_audit_v15_missing")
        else:
            try:
                carry = load_json(carry_path)
                if carry.get("ok") is not True:
                    errors.append("carry_forward_audit_v15_not_ok")
                if carry.get("verify_all_shas") is not True:
                    errors.append("carry_forward_audit_v15_verify_all_shas")
                if carry.get("mutated") is True:
                    errors.append("carry_forward_audit_v15_mutated")
                if approval and approval.get("skeleton_carry_forward_audit_sha256"):
                    if approval.get("skeleton_carry_forward_audit_sha256") != sha256_file(carry_path):
                        errors.append("carry_forward_audit_v15_sha_bind_mismatch")
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"carry_forward_audit_v15_unreadable:{exc}")

        smoke_path = root / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL
        if not smoke_path.is_file():
            errors.append("rgb_f0_smoke_report_v15_missing")
        else:
            try:
                validate_rgb_f0_smoke_report_v2(load_json(smoke_path), root=root)
            except ValueError as exc:
                errors.append(str(exc))
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"rgb_f0_smoke_report_v15_unreadable:{exc}")

        if approval and approval.get("binary_root_rel"):
            bin_path = root / str(approval["binary_root_rel"])
            try:
                assert_binary_root_absent_or_empty(bin_path)
            except ValueError as exc:
                errors.append(str(exc))
            runtime = bin_path / RUNTIME_REL_SUFFIX
            if runtime.is_dir():
                if (runtime / "HALT.json").is_file():
                    errors.append("v15_binary_runtime_halt_present")
                if (runtime / "DETERMINISTIC_PHASE_FAILURE.json").is_file():
                    errors.append("v15_binary_runtime_deterministic_failure_present")
    except ValueError as exc:
        errors.append(str(exc))

    ready = len(errors) == 0
    return {
        "ready": ready,
        "errors": errors,
        "output_version": OUTPUT_VERSION_V15,
        "campaign_token": CAMPAIGN_TOKEN_V15,
        "approval": approval,
        "protocol": protocol,
        "draft_friendly": False,
        "extraction_authorized": bool(
            ready and (approval or {}).get("allow_full_corpus_extraction") is True
        ),
    }


def require_launch_readiness_v15(*, root: Path = _ROOT) -> dict[str, Any]:
    assert_no_out_of_provenance_sitecustomize(root=root)
    status = evaluate_launch_readiness_v15(root=root)
    if not status["ready"]:
        raise RuntimeError("launch_gates_blocked_v15:" + ";".join(status["errors"]))
    return status


def assert_binary_root_absent_or_empty(path: Path) -> None:
    """OK if missing; fail if any files exist under path."""
    if not path.exists():
        return
    if path.is_file():
        raise ValueError(f"binary_root_is_file:{path}")
    for child in path.rglob("*"):
        if child.is_file():
            raise ValueError(f"binary_root_not_empty:{path}")


def validate_rgb_f0_smoke_report_v2(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Validate v15 RGB F0 smoke report schema v2 (PASS/ok + provenance binds)."""
    del root  # reserved for future path binds; signature matches call sites
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_v15_rgb_f0_smoke_report_v2":
        errors.append("smoke_report_v2_bad_schema")
    if obj.get("ok") is not True:
        errors.append("smoke_report_v2_not_ok")
    if obj.get("status") != "PASS":
        errors.append("smoke_report_v2_not_pass")
    required = (
        "code_lock_git_revision",
        "executable_tree_sha256",
        "python_exe",
        "pip_freeze_sha256",
        "backbone_sha256",
        "source_video_sha256",
        "skeleton_sha256",
        "sample_id",
        "required_keys_ok",
        "shapes",
        "dtypes",
        "finite_ok",
        "token_content_sha256",
    )
    for key in required:
        if key not in obj or obj.get(key) is None:
            errors.append(f"smoke_report_v2_missing:{key}")
    if obj.get("required_keys_ok") is not True:
        errors.append("smoke_report_v2_required_keys_not_ok")
    if obj.get("finite_ok") is not True:
        errors.append("smoke_report_v2_finite_not_ok")
    if not isinstance(obj.get("shapes"), dict):
        errors.append("smoke_report_v2_shapes_not_dict")
    if not isinstance(obj.get("dtypes"), dict):
        errors.append("smoke_report_v2_dtypes_not_dict")
    rev = obj.get("code_lock_git_revision")
    if not (isinstance(rev, str) and _HEX40.match(rev)):
        errors.append("smoke_report_v2_code_lock_invalid")
    tree = obj.get("executable_tree_sha256")
    if not (isinstance(tree, str) and len(tree) == 64):
        errors.append("smoke_report_v2_executable_tree_invalid")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def publish_deterministic_phase_failure_create_only(
    runtime: Path,
    phase: str,
    exc: BaseException,
    *,
    n_ok: int,
    n_fail: int,
    n_resume: int,
    sample_id: str | None = None,
) -> dict[str, Any]:
    """Create-only DETERMINISTIC_PHASE_FAILURE; preserve phase counters in summary.

    Never resets n_ok/n_resume/n_fail to 0. Identical payload is idempotent; different
    existing content raises FileExistsError (refuse overwrite).
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "schema": "ms_sta_v5_d0_full_corpus_deterministic_phase_failure_v1",
        "phase": phase,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "sample_id": sample_id,
        "deterministic": True,
        "watchdog_resume_forbidden": True,
        "n_ok_at_halt": int(n_ok),
        "n_fail_at_halt": int(n_fail),
        "n_resume_at_halt": int(n_resume),
    }
    det_path = runtime / "DETERMINISTIC_PHASE_FAILURE.json"
    created = False
    if det_path.is_file():
        existing = load_json(det_path)
        identity_keys = (
            "schema",
            "phase",
            "error",
            "error_type",
            "sample_id",
            "deterministic",
            "watchdog_resume_forbidden",
            "n_ok_at_halt",
            "n_fail_at_halt",
            "n_resume_at_halt",
        )
        same = all(existing.get(k) == payload.get(k) for k in identity_keys)
        if not same:
            raise FileExistsError(f"exists_with_different_content:{det_path}")
    else:
        atomic_create_json(det_path, payload)
        created = True
        append_jsonl(
            runtime / "FAILURE_LEDGER.jsonl",
            {
                "sample_id": sample_id or "_phase_",
                "phase": phase,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "deterministic": True,
                "ts": ts,
            },
        )

    summary = {
        "phase": phase,
        "n_ok": int(n_ok),
        "n_fail": int(n_fail),
        "n_resume": int(n_resume),
        "ok": False,
        "deterministic_dependency_failure": True,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "deterministic_failure_created": created,
    }
    if phase == "rgb_token_v3":
        atomic_replace_json(runtime / "RGB_PHASE_SUMMARY.json", summary)

    mutable_path = runtime / "RUNTIME_MUTABLE.json"
    if mutable_path.is_file():
        try:
            mutable = load_json(mutable_path)
        except (OSError, json.JSONDecodeError):
            mutable = {}
    else:
        mutable = {}
    mutable["status"] = "HALTED_DETERMINISTIC_DEPENDENCY_FAIL"
    mutable["phase"] = "halted"
    mutable["updated_at"] = ts
    mutable["close_out"] = "DETERMINISTIC_DEPENDENCY_FAIL"
    atomic_replace_json(mutable_path, mutable)
    return summary


def verify_skeleton_carry_forward_from_v14(
    *,
    root: Path = _ROOT,
    sealed_binary_root_rel: str = V14_SEALED_SKELETON_BINARY_ROOT_REL,
    verify_all_shas: bool = True,
) -> dict[str, Any]:
    """Read-only audit of sealed v14 skeleton for v15 carry-forward.

    NEVER copies, hardlinks, or mutates the v14 binary root / manifests / ledger.
    """
    errors: list[str] = []
    binary = root / sealed_binary_root_rel
    runtime = binary / "runtime"
    summary_path = runtime / "SKEL_PHASE_SUMMARY.json"
    index_path = runtime / "manifests" / "skeleton_v3" / "INDEX.json"
    closeout_path = root / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL

    if not binary.is_dir():
        return {
            "ok": False,
            "errors": [f"sealed_binary_root_missing:{sealed_binary_root_rel}"],
            "source_root_rel": sealed_binary_root_rel,
            "mutated": False,
            "copied": False,
            "hardlinked": False,
        }

    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"skel_phase_summary_unreadable:{exc}"],
            "source_root_rel": sealed_binary_root_rel,
            "mutated": False,
            "copied": False,
            "hardlinked": False,
        }
    try:
        index = load_json(index_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "errors": [f"skel_index_unreadable:{exc}"],
            "source_root_rel": sealed_binary_root_rel,
            "mutated": False,
            "copied": False,
            "hardlinked": False,
        }

    identity = load_eligible_identity(root=root)
    eligible_ids = list(identity["eligible_ids"])
    records = index.get("records") or {}
    if not isinstance(records, dict):
        errors.append("index_records_not_dict")
        records = {}

    if summary.get("ok") is not True:
        errors.append("summary_ok_false")
    n_ok = summary.get("n_ok")
    n_fail = summary.get("n_fail")
    if n_ok is None or int(n_ok) != CORPUS_N:
        errors.append(f"summary_n_ok:{n_ok}")
    if n_fail is None or int(n_fail) != 0:
        errors.append(f"summary_n_fail:{n_fail}")
    n_records = index.get("n_records")
    if n_records is None or int(n_records) != CORPUS_N:
        errors.append(f"index_n_records:{n_records}")
    if len(records) != CORPUS_N:
        errors.append(f"index_records_len:{len(records)}")
    if len(eligible_ids) != CORPUS_N:
        errors.append(f"eligible_len:{len(eligible_ids)}")

    sha_mismatches = 0
    sha_checked = 0
    missing_npz = 0
    missing_sidecar = 0
    sample_iter = eligible_ids if verify_all_shas else eligible_ids[:64]
    for sid in sample_iter:
        row = records.get(sid)
        if not isinstance(row, dict):
            errors.append(f"index_row_missing:{sid}")
            continue
        out_rel = str(row.get("output_rel") or f"skeleton_v3/{sid}.npz")
        side_rel = str(row.get("sidecar_rel") or f"skeleton_v3/{sid}.recovery.json")
        out_p = root / out_rel
        if not out_p.is_file():
            out_p = binary / "skeleton_v3" / f"{sid}.npz"
        side_p = root / side_rel
        if not side_p.is_file():
            side_p = binary / "skeleton_v3" / f"{sid}.recovery.json"
        if not out_p.is_file():
            missing_npz += 1
            errors.append(f"missing_npz:{sid}")
            continue
        declared_out = row.get("output_sha256")
        if not declared_out:
            errors.append(f"output_sha_missing:{sid}")
        else:
            sha_checked += 1
            if sha256_file(out_p) != declared_out:
                sha_mismatches += 1
                errors.append(f"output_sha_mismatch:{sid}")
        if side_p.is_file():
            declared_side = row.get("sidecar_sha256")
            if declared_side:
                if sha256_file(side_p) != declared_side:
                    sha_mismatches += 1
                    errors.append(f"sidecar_sha_mismatch:{sid}")
            else:
                errors.append(f"sidecar_sha_missing:{sid}")
        else:
            missing_sidecar += 1
            errors.append(f"missing_sidecar:{sid}")
        if len(errors) >= 200:
            errors.append("error_cap_reached")
            break

    binds = {
        "skel_phase_summary_rel": str(summary_path.relative_to(root)).replace("\\", "/"),
        "skel_phase_summary_sha256": sha256_file(summary_path),
        "skel_index_rel": str(index_path.relative_to(root)).replace("\\", "/"),
        "skel_index_sha256": sha256_file(index_path),
        "v14_halt_closeout_rel": D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL,
        "v14_halt_closeout_sha256": (
            sha256_file(closeout_path) if closeout_path.is_file() else None
        ),
        "required_n_records": CORPUS_N,
        "required_n_ok": CORPUS_N,
        "required_n_fail": 0,
    }

    return {
        "ok": not errors,
        "errors": errors[:200],
        "source_root_rel": sealed_binary_root_rel.replace("\\", "/"),
        "source_root_policy": "read_only",
        "mutated": False,
        "copied": False,
        "hardlinked": False,
        "summary_ok": bool(summary.get("ok")),
        "summary_n_ok": int(n_ok if n_ok is not None else 0),
        "summary_n_fail": int(n_fail if n_fail is not None else 0),
        "index_n_records": int(n_records if n_records is not None else 0),
        "sha_checked": sha_checked,
        "sha_mismatches": sha_mismatches,
        "missing_npz": missing_npz,
        "missing_sidecar": missing_sidecar,
        "verify_all_shas": verify_all_shas,
        "binds": binds,
        "forbid": [
            "copy_skeleton_tree",
            "hardlink_skeleton_tree",
            "reextract_skeleton_v3",
            "mutate_v14_manifests",
            "mutate_v14_failure_ledger",
        ],
    }


def validate_run_contract_v8(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v8":
        errors.append("bad_run_contract_v8_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v8_status")
    if obj.get("output_version") != OUTPUT_VERSION_V8:
        errors.append("run_contract_v8_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v8_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v8_autochain")
    if obj.get("approval_v8_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V8_REL):
        errors.append("run_contract_v8_approval_sha")
    if obj.get("execution_protocol_v8_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V8_REL):
        errors.append("run_contract_v8_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v8(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v8":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V8:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V8
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V8_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V8:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V8,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v8(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v8":
        errors.append("bad_approval_v8_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v8_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v8":
        errors.append("approval_v8_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v8_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v8_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V8:
        errors.append("approval_v8_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v8_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v8_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v8_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v8_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v8_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v8_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v8_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v8_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v8_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v8_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v8_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v8_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v8_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v8_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v8_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v8_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V8,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V8):
        errors.append("approval_v8_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V8_REL:
        errors.append("approval_v8_evidence_root")
    if obj.get("immutable_v7_approval_sha256") != EXPECTED_V7_APPROVAL_SHA256:
        errors.append("approval_v8_v7_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V7_REL) != EXPECTED_V7_APPROVAL_SHA256:
        errors.append("v7_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v7_sha256") != EXPECTED_V7_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v8_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V7_REL) != EXPECTED_V7_EXTERNAL_REAUDIT_SHA256:
        errors.append("v7_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v9(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v9":
        errors.append("bad_run_contract_v9_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v9_status")
    if obj.get("output_version") != OUTPUT_VERSION_V9:
        errors.append("run_contract_v9_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v9_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v9_autochain")
    if obj.get("approval_v9_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V9_REL):
        errors.append("run_contract_v9_approval_sha")
    if obj.get("execution_protocol_v9_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V9_REL):
        errors.append("run_contract_v9_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v10(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v10":
        errors.append("bad_run_contract_v10_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v10_status")
    if obj.get("output_version") != OUTPUT_VERSION_V10:
        errors.append("run_contract_v10_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v10_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v10_autochain")
    if obj.get("approval_v10_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V10_REL):
        errors.append("run_contract_v10_approval_sha")
    if obj.get("execution_protocol_v10_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V10_REL):
        errors.append("run_contract_v10_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v11(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v11":
        errors.append("bad_run_contract_v11_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v11_status")
    if obj.get("output_version") != OUTPUT_VERSION_V11:
        errors.append("run_contract_v11_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v11_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v11_autochain")
    if obj.get("approval_v11_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V11_REL):
        errors.append("run_contract_v11_approval_sha")
    if obj.get("execution_protocol_v11_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V11_REL):
        errors.append("run_contract_v11_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v12(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v12":
        errors.append("bad_run_contract_v12_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v12_status")
    if obj.get("output_version") != OUTPUT_VERSION_V12:
        errors.append("run_contract_v12_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v12_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v12_autochain")
    if obj.get("approval_v12_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V12_REL):
        errors.append("run_contract_v12_approval_sha")
    if obj.get("execution_protocol_v12_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V12_REL):
        errors.append("run_contract_v12_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v13(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v13":
        errors.append("bad_run_contract_v13_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v13_status")
    if obj.get("output_version") != OUTPUT_VERSION_V13:
        errors.append("run_contract_v13_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v13_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v13_autochain")
    if obj.get("approval_v13_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V13_REL):
        errors.append("run_contract_v13_approval_sha")
    if obj.get("execution_protocol_v13_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V13_REL):
        errors.append("run_contract_v13_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v14(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v14":
        errors.append("bad_run_contract_v14_schema")
    if obj.get("status") not in {
        "AUTHORIZED_NOT_STARTED",
        "RUNNING",
        "RESUMED",
        "CLOSED_PASS",
        "CLOSED_FAIL",
    }:
        errors.append("run_contract_v14_status")
    if obj.get("output_version") != OUTPUT_VERSION:
        errors.append("run_contract_v14_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v14_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v14_autochain")
    if obj.get("approval_v14_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_V14_REL):
        errors.append("run_contract_v14_approval_sha")
    if obj.get("execution_protocol_v14_sha256") != sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL):
        errors.append("run_contract_v14_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v9(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v9":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V9:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V9
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V9_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V9:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V9,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v10(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v10":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V10:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V10
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V10_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V10:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V10,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v11(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v11":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V11:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V11
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V11_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V11:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V11,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v12(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v12":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V12:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V12
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V12_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V12:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V12,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v13(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v13":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V13:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V13
    ):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V13_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V13:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V13,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v14(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v14":
        errors.append("bad_exec_protocol_schema")
    if obj.get("output_version") != OUTPUT_VERSION:
        errors.append("exec_protocol_output_version")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(obj.get("code_lock_git_revision"), root=root):
        errors.append("exec_protocol_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_REL:
        errors.append("exec_protocol_evidence_root")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS:
        errors.append("exec_protocol_executable_paths")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v9(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v9":
        errors.append("bad_approval_v9_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v9_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v9":
        errors.append("approval_v9_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v9_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v9_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V9:
        errors.append("approval_v9_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v9_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v9_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v9_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v9_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v9_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v9_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v9_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v9_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v9_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v9_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v9_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v9_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v9_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v9_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v9_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v9_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V9,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V9):
        errors.append("approval_v9_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V9_REL:
        errors.append("approval_v9_evidence_root")
    if obj.get("immutable_v8_approval_sha256") != EXPECTED_V8_APPROVAL_SHA256:
        errors.append("approval_v9_v8_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V8_REL) != EXPECTED_V8_APPROVAL_SHA256:
        errors.append("v8_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v8_sha256") != EXPECTED_V8_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v9_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V8_REL) != EXPECTED_V8_EXTERNAL_REAUDIT_SHA256:
        errors.append("v8_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v10(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v10":
        errors.append("bad_approval_v10_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v10_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v10":
        errors.append("approval_v10_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v10_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v10_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V10:
        errors.append("approval_v10_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v10_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v10_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v10_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v10_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v10_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v10_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v10_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v10_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v10_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v10_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v10_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v10_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v10_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v10_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v10_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v10_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V10,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V10):
        errors.append("approval_v10_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V10_REL:
        errors.append("approval_v10_evidence_root")
    if obj.get("immutable_v9_approval_sha256") != EXPECTED_V9_APPROVAL_SHA256:
        errors.append("approval_v10_v9_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V9_REL) != EXPECTED_V9_APPROVAL_SHA256:
        errors.append("v9_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v9_sha256") != EXPECTED_V9_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v10_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V9_REL) != EXPECTED_V9_EXTERNAL_REAUDIT_SHA256:
        errors.append("v9_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v11(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v11":
        errors.append("bad_approval_v11_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v11_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v11":
        errors.append("approval_v11_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v11_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v11_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V11:
        errors.append("approval_v11_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v11_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v11_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v11_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v11_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v11_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v11_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v11_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v11_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v11_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v11_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v11_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v11_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v11_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v11_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v11_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v11_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V11,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V11):
        errors.append("approval_v11_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V11_REL:
        errors.append("approval_v11_evidence_root")
    if obj.get("immutable_v10_approval_sha256") != EXPECTED_V10_APPROVAL_SHA256:
        errors.append("approval_v11_v10_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V10_REL) != EXPECTED_V10_APPROVAL_SHA256:
        errors.append("v10_approval_bytes_drift")
    if obj.get("parent_execution_interruption_v10_sha256") != EXPECTED_V10_INTERRUPTION_SHA256:
        errors.append("approval_v11_interruption_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_INTERRUPTION_V10_REL) != EXPECTED_V10_INTERRUPTION_SHA256:
        errors.append("v10_interruption_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v12(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v12":
        errors.append("bad_approval_v12_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v12_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v12":
        errors.append("approval_v12_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v12_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v12_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V12:
        errors.append("approval_v12_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v12_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v12_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v12_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v12_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v12_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v12_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v12_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v12_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v12_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v12_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v12_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v12_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v12_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v12_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v12_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v12_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V12,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V12):
        errors.append("approval_v12_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V12_REL:
        errors.append("approval_v12_evidence_root")
    if obj.get("immutable_v11_approval_sha256") != EXPECTED_V11_APPROVAL_SHA256:
        errors.append("approval_v12_v11_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V11_REL) != EXPECTED_V11_APPROVAL_SHA256:
        errors.append("v11_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v11_sha256") != EXPECTED_V11_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v12_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V11_REL) != EXPECTED_V11_EXTERNAL_REAUDIT_SHA256:
        errors.append("v11_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v13(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v13":
        errors.append("bad_approval_v13_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v13_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v13":
        errors.append("approval_v13_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v13_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v13_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION_V13:
        errors.append("approval_v13_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v13_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v13_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v13_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v13_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v13_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v13_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v13_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v13_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v13_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v13_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v13_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v13_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v13_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v13_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v13_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v13_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V13,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V13):
        errors.append("approval_v13_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V13_REL:
        errors.append("approval_v13_evidence_root")
    if obj.get("immutable_v12_approval_sha256") != EXPECTED_V12_APPROVAL_SHA256:
        errors.append("approval_v13_v12_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V12_REL) != EXPECTED_V12_APPROVAL_SHA256:
        errors.append("v12_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v12_sha256") != EXPECTED_V12_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v13_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V12_REL) != EXPECTED_V12_EXTERNAL_REAUDIT_SHA256:
        errors.append("v12_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v14(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v14":
        errors.append("bad_approval_v14_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v14_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only_v14":
        errors.append("approval_v14_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v14_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("one_shot") is not True:
        errors.append("approval_v14_one_shot")
    if obj.get("output_version") != OUTPUT_VERSION:
        errors.append("approval_v14_output_version")
    if obj.get("skeleton_arm") != "C3":
        errors.append("approval_v14_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("approval_v14_psutil_pin")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("approval_v14_h1_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_bytes_drift")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("approval_v14_eligible")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("approval_v14_metadata")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("approval_v14_split")
    if obj.get("full_protocol_sha256") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("approval_v14_protocol")
    if obj.get("backbone_sha256") != EXPECTED_SLOWFAST_CKPT_SHA256:
        errors.append("approval_v14_backbone")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("approval_v14_phases")
    if int(obj.get("max_resumes") or -1) != MAX_RESUMES:
        errors.append("approval_v14_max_resumes")
    resume = obj.get("resume_contract") or {}
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("approval_v14_resume_idempotent")
    if resume.get("require_manifest_sha_for_resume") is not True:
        errors.append("approval_v14_resume_manifest_sha")
    if resume.get("require_source_sha256") is not True:
        errors.append("approval_v14_require_source_sha")
    if resume.get("worker_registry_blocks_resume") is not True:
        errors.append("approval_v14_worker_registry")
    if resume.get("sample_transaction_heal") is not True:
        errors.append("approval_v14_sample_tx")
    if resume.get("atomic_manifest_records_required") is not True:
        errors.append("approval_v14_atomic_manifest")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root):
        errors.append("approval_v14_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_REL:
        errors.append("approval_v14_evidence_root")
    if obj.get("immutable_v13_approval_sha256") != EXPECTED_V13_APPROVAL_SHA256:
        errors.append("approval_v14_v13_bind")
    if sha256_file(root / D0_FULL_CORPUS_APPROVAL_V13_REL) != EXPECTED_V13_APPROVAL_SHA256:
        errors.append("v13_approval_bytes_drift")
    if obj.get("parent_external_reaudit_v13_sha256") != EXPECTED_V13_EXTERNAL_REAUDIT_SHA256:
        errors.append("approval_v14_reaudit_bind")
    if sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V13_REL) != EXPECTED_V13_EXTERNAL_REAUDIT_SHA256:
        errors.append("v13_reaudit_bytes_drift")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_approval_v15(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """SEALED approval only — rejects draft scaffolding."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v15":
        errors.append("bad_approval_v15_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("approval_v15_decision")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("approval_v15_allow_extract")
    for key in ("allow_m0", "allow_training", "allow_hpo", "allow_test", "allow_demo", "allow_auto_chain"):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("approval_v15_output_version")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("approval_v15_campaign_token")
    tree = obj.get("executable_tree_sha256")
    if tree is None or tree == "":
        errors.append("approval_v15_executable_tree_null")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V15,
            tree,
            root=root,
            require_worktree_match=False,
        )
    )
    if obj.get("binary_root_rel") != binary_root_rel(rev, root=root, version=OUTPUT_VERSION_V15):
        errors.append("approval_v15_binary_root")
    if obj.get("skeleton_source_root_rel") != V14_SEALED_SKELETON_BINARY_ROOT_REL:
        errors.append("approval_v15_skeleton_source_root")
    if obj.get("carry_forward_required") is not True:
        errors.append("approval_v15_carry_forward_required")
    halt_path = root / D0_FULL_CORPUS_V14_HALT_CLOSEOUT_REL
    if not halt_path.is_file():
        errors.append("approval_v15_halt_closeout_missing")
    elif obj.get("parent_v14_halt_closeout_sha256") != sha256_file(halt_path):
        errors.append("approval_v15_halt_closeout_sha_mismatch")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V15_REL:
        errors.append("approval_v15_evidence_root")
    carry_bind = obj.get("skeleton_carry_forward_audit_sha256")
    if carry_bind is not None:
        carry_path = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
        if not carry_path.is_file():
            errors.append("approval_v15_carry_forward_audit_missing")
        elif carry_bind != sha256_file(carry_path):
            errors.append("approval_v15_carry_forward_audit_sha_mismatch")
    smoke_bind = obj.get("rgb_f0_smoke_report_sha256")
    if smoke_bind is not None:
        smoke_path = root / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL
        if not smoke_path.is_file():
            errors.append("approval_v15_smoke_report_missing")
        elif smoke_bind != sha256_file(smoke_path):
            errors.append("approval_v15_smoke_report_sha_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_execution_protocol_v15(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_execution_protocol_v15":
        errors.append("bad_exec_protocol_v15_schema")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("exec_protocol_v15_output_version")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("exec_protocol_v15_campaign")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("exec_protocol_v15_phases")
    if obj.get("eligible_n") != CORPUS_N:
        errors.append("exec_protocol_v15_eligible_n")
    if obj.get("binary_root_rel") != binary_root_rel(
        obj.get("code_lock_git_revision"), root=root, version=OUTPUT_VERSION_V15
    ):
        errors.append("exec_protocol_v15_binary_root")
    if obj.get("evidence_root_rel") != EVIDENCE_ROOT_V15_REL:
        errors.append("exec_protocol_v15_evidence_root")
    if obj.get("skeleton_source_root_rel") != V14_SEALED_SKELETON_BINARY_ROOT_REL:
        errors.append("exec_protocol_v15_skeleton_source")
    if obj.get("carry_forward_required") is not True:
        errors.append("exec_protocol_v15_carry_forward_required")
    if obj.get("allow_m0") is not False or obj.get("allow_auto_chain") is not False:
        errors.append("exec_protocol_v15_forbid_m0_autochain")
    if obj.get("stop_after_pass") != "D0_FULL_CORPUS_SCAN_COMPLETE_PASS":
        errors.append("exec_protocol_v15_stop_pass")
    if obj.get("stop_after_fail") != "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL":
        errors.append("exec_protocol_v15_stop_fail")
    if obj.get("skeleton_arm") != "C3":
        errors.append("exec_protocol_v15_skeleton_arm")
    if obj.get("psutil_pin") != PSUTIL_PIN:
        errors.append("exec_protocol_v15_psutil_pin")
    if obj.get("worker_registry_required") is not True:
        errors.append("exec_protocol_v15_worker_registry")
    if obj.get("sample_transaction_required") is not True:
        errors.append("exec_protocol_v15_sample_tx")
    if obj.get("atomic_manifest_required") is not True:
        errors.append("exec_protocol_v15_atomic_manifest")
    paths = list(obj.get("executable_paths") or [])
    if paths != FULL_CORPUS_EXECUTABLE_PATHS_V15:
        errors.append("exec_protocol_v15_executable_paths")
    tree = obj.get("executable_tree_sha256")
    if tree is None or tree == "":
        errors.append("exec_protocol_v15_executable_tree_null")
    rev = obj.get("code_lock_git_revision")
    errors.extend(
        verify_code_lock_revision_binding(
            rev,
            FULL_CORPUS_EXECUTABLE_PATHS_V15,
            tree,
            root=root,
            require_worktree_match=False,
        )
    )
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_run_contract_v15(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v15":
        errors.append("bad_run_contract_v15_schema")
    if obj.get("status") not in {"APPROVED", "SEALED"}:
        errors.append("run_contract_v15_status")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("run_contract_v15_version")
    if obj.get("campaign") != CAMPAIGN_TOKEN_V15 and obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("run_contract_v15_campaign")
    if obj.get("phases") != list(PHASES_EXACT):
        errors.append("run_contract_v15_phases")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_v15_autochain")
    # full_corpus_authorized True is OK for sealed contract
    if obj.get("full_corpus_authorized") is not True:
        errors.append("run_contract_v15_full_corpus_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    if approval_path.is_file() and obj.get("approval_v15_sha256") != sha256_file(approval_path):
        errors.append("run_contract_v15_approval_sha")
    if protocol_path.is_file() and obj.get("execution_protocol_v15_sha256") != sha256_file(protocol_path):
        errors.append("run_contract_v15_protocol_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_external_reaudit_pass_v15(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != EXTERNAL_REAUDIT_PASS_SCHEMA_V15:
        errors.append("external_reaudit_v15_bad_schema")
    verdict = obj.get("audit_verdict")
    if verdict != "PASS":
        errors.append(f"external_reaudit_v15_not_pass:{verdict}")
    if obj.get("auditor") != "independent":
        errors.append("external_reaudit_v15_auditor")
    if obj.get("worktree_clean_before_audit_artifact") is not True:
        errors.append("external_reaudit_v15_worktree_not_clean")
    head = obj.get("head_at_audit")
    origin = obj.get("origin_main_at_audit")
    if not (isinstance(head, str) and _HEX40.match(head)):
        errors.append("external_reaudit_v15_head_at_audit_invalid")
    if not (isinstance(origin, str) and _HEX40.match(origin)):
        errors.append("external_reaudit_v15_origin_main_at_audit_invalid")
    elif head != origin:
        errors.append("external_reaudit_v15_head_origin_mismatch")
    elif isinstance(head, str) and _HEX40.match(head) and not git_commit_exists(head, root=root):
        errors.append("external_reaudit_v15_head_not_found")

    approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    run_contract_path = root / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL
    approval_audit_path = root / D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL
    carry_path = root / D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL
    smoke_path = root / D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL

    approval_code_lock: str | None = None
    if not approval_path.is_file():
        errors.append("approval_v15_missing")
    else:
        try:
            approval = load_json(approval_path)
            if approval.get("status") != "APPROVED" or approval.get("decision") != "APPROVED":
                errors.append("external_reaudit_v15_draft_approval_rejected")
            approval_code_lock = approval.get("code_lock_git_revision")
            if obj.get("code_lock_git_revision") != approval_code_lock:
                errors.append("external_reaudit_v15_code_lock_mismatch")
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v15_unreadable")

    head_rev = head if isinstance(head, str) and _HEX40.match(head) else None
    core_tracked = {
        "approval_v15_sha256",
        "execution_protocol_v15_sha256",
        "run_contract_v15_sha256",
        "approval_audit_v15_sha256",
    }
    for field, path, rel in (
        ("approval_v15_sha256", approval_path, D0_FULL_CORPUS_APPROVAL_V15_REL),
        ("execution_protocol_v15_sha256", protocol_path, D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        ("run_contract_v15_sha256", run_contract_path, D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        ("approval_audit_v15_sha256", approval_audit_path, D0_FULL_CORPUS_APPROVAL_AUDIT_V15_REL),
        (
            "skeleton_carry_forward_audit_sha256",
            carry_path,
            D0_FULL_CORPUS_V15_SKELETON_CARRY_FORWARD_AUDIT_REL,
        ),
        ("rgb_f0_smoke_report_sha256", smoke_path, D0_FULL_CORPUS_V15_RGB_SMOKE_REPORT_REL),
    ):
        expected = obj.get(field)
        if not path.is_file():
            errors.append(f"{field.replace('_sha256', '')}_missing")
            continue
        disk_sha = sha256_file(path)
        if expected != disk_sha:
            errors.append(f"{field}_mismatch")
            continue
        if head_rev and git_commit_exists(head_rev, root=root):
            try:
                git_sha = sha256_git_blob_at_revision(rel, head_rev, root=root)
            except FileNotFoundError:
                if field in core_tracked:
                    errors.append(f"external_reaudit_v15_git_blob_missing:{field}")
            else:
                if expected != git_sha:
                    errors.append(f"external_reaudit_v15_git_blob_mismatch:{field}")

    blocking = obj.get("blocking_findings")
    if not isinstance(blocking, list):
        errors.append("external_reaudit_v15_blocking_findings_not_list")
    elif blocking:
        errors.append("external_reaudit_v15_blocking_findings_nonempty")

    decision = obj.get("decision")
    if not isinstance(decision, dict):
        errors.append("external_reaudit_v15_decision_missing")
    else:
        for key, required in EXTERNAL_REAUDIT_PASS_REQUIRED_DECISION.items():
            if key not in decision:
                errors.append(f"external_reaudit_v15_decision_missing:{key}")
            elif decision[key] is not required:
                if required is True:
                    errors.append("external_reaudit_v15_post_audit_not_allowed")
                else:
                    errors.append(f"external_reaudit_v15_{key}_must_be_false")

    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("external_reaudit_v15_output_version_mismatch")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("external_reaudit_v15_campaign_token_mismatch")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v15(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v15":
        errors.append("bad_post_audit_auth_v15_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("post_audit_v15_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("post_audit_v15_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    for key in (
        "allow_m0",
        "allow_training",
        "allow_hpo",
        "allow_test",
        "allow_demo",
        "allow_auto_chain",
        "m0_authorized",
        "training_authorized",
        "hpo_authorized",
        "test_authorized",
        "demo_authorized",
        "auto_chain",
        "auto_chain_to_m0",
    ):
        if key in obj and obj.get(key) is not False:
            errors.append(f"post_audit_v15_{key}_must_be_false")

    approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V15_REL
    if not approval_path.is_file():
        errors.append("approval_v15_missing")
    elif obj.get("approval_v15_sha256") != sha256_file(approval_path):
        errors.append("approval_v15_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v15_missing")
    elif obj.get("execution_protocol_v15_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v15_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v15_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    else:
        try:
            validate_external_reaudit_pass_v15(load_json(reaudit_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))

    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_v15_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_v15_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_v15_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V15,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v15_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_scan_recovery_external_reaudit_pass_v15(
    obj: dict[str, Any], *, root: Path = _ROOT
) -> dict[str, Any]:
    """Dedicated scan-recovery audit PASS (does not authorize extraction/M0).

    Requires mandatory exact SHA binds for approval, protocol, run-contract,
    R1 remediation companion, performance smoke, and RGB region-validity diagnostic.
    Omitted fields are rejected.
    """
    errors: list[str] = []
    if obj.get("schema") != SCAN_RECOVERY_EXTERNAL_REAUDIT_SCHEMA_V15:
        errors.append("scan_recovery_external_reaudit_bad_schema")
    if obj.get("audit_verdict") != "PASS":
        errors.append(f"scan_recovery_external_reaudit_not_pass:{obj.get('audit_verdict')}")
    if obj.get("auditor") != "independent":
        errors.append("scan_recovery_external_reaudit_auditor")
    if obj.get("scope") != "scan_recovery_only":
        errors.append("scan_recovery_external_reaudit_scope")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("scan_recovery_external_reaudit_output_version")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("scan_recovery_external_reaudit_campaign_token")
    for key in (
        "allow_extraction",
        "allow_m0",
        "allow_training",
        "allow_hpo",
        "allow_test",
        "allow_demo",
        "allow_auto_chain",
    ):
        if obj.get(key) is not False:
            errors.append(f"scan_recovery_external_reaudit_{key}_must_be_false")
    blocking = obj.get("blocking_findings")
    if not isinstance(blocking, list):
        errors.append("scan_recovery_external_reaudit_blocking_findings_required")
    elif len(blocking) != 0:
        errors.append("scan_recovery_external_reaudit_blocking_findings_must_be_empty")

    approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL
    run_contract_path = root / D0_FULL_CORPUS_RUN_CONTRACT_V15_REL
    r1_path = root / D0_FULL_CORPUS_V15_SCAN_RECOVERY_R1_PERF_REMEDIATION_REL
    smoke_path = root / D0_FULL_CORPUS_V15_SCAN_RECOVERY_PERF_SMOKE_REL
    diagnostic_path = root / D0_FULL_CORPUS_V15_RGB_REGION_VALIDITY_DIAGNOSTIC_REL

    required_sha_binds = (
        ("approval_v15_sha256", approval_path, "approval_v15_missing"),
        ("execution_protocol_v15_sha256", protocol_path, "execution_protocol_v15_missing"),
        ("run_contract_v15_sha256", run_contract_path, "run_contract_v15_missing"),
        (
            "scan_recovery_r1_perf_remediation_sha256",
            r1_path,
            "scan_recovery_r1_perf_remediation_missing",
        ),
        (
            "scan_recovery_perf_smoke_sha256",
            smoke_path,
            "scan_recovery_perf_smoke_missing",
        ),
        (
            "rgb_region_validity_diagnostic_sha256",
            diagnostic_path,
            "rgb_region_validity_diagnostic_missing",
        ),
    )
    for field, path, missing_err in required_sha_binds:
        declared = obj.get(field)
        if not isinstance(declared, str) or not declared:
            errors.append(f"scan_recovery_external_reaudit_{field}_required")
            continue
        if not path.is_file():
            errors.append(missing_err)
            continue
        observed = sha256_file(path)
        if declared != observed:
            errors.append(f"scan_recovery_external_reaudit_{field}_mismatch")

    if not approval_path.is_file():
        errors.append("approval_v15_missing")
    else:
        approval = load_json(approval_path)
        if obj.get("code_lock_git_revision") != approval.get("code_lock_git_revision"):
            errors.append("scan_recovery_external_reaudit_code_lock_mismatch")
        if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
            errors.append("scan_recovery_external_reaudit_tree_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_scan_recovery_post_audit_auth_v15(
    obj: dict[str, Any], *, root: Path = _ROOT
) -> dict[str, Any]:
    """Authorize one read-only scan-recovery execute after dedicated external PASS."""
    errors: list[str] = []
    if obj.get("schema") != SCAN_RECOVERY_POST_AUDIT_AUTH_SCHEMA_V15:
        errors.append("scan_recovery_post_audit_auth_bad_schema")
    if obj.get("scan_recovery_execute_authorized") is not True:
        errors.append("scan_recovery_execute_not_authorized")
    if obj.get("output_version") != OUTPUT_VERSION_V15:
        errors.append("scan_recovery_post_audit_auth_output_version")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V15:
        errors.append("scan_recovery_post_audit_auth_campaign_token")
    for key in (
        "allow_extraction",
        "allow_m0",
        "allow_training",
        "allow_hpo",
        "allow_test",
        "allow_demo",
        "allow_auto_chain",
        "auto_chain_to_m0",
    ):
        if obj.get(key) is not False:
            errors.append(f"scan_recovery_post_audit_auth_{key}_must_be_false")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V15_REL
    reaudit_path = root / D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL
    if not approval_path.is_file():
        errors.append("approval_v15_missing")
    elif obj.get("approval_v15_sha256") != sha256_file(approval_path):
        errors.append("scan_recovery_post_audit_auth_approval_sha_mismatch")
    if not reaudit_path.is_file():
        errors.append("scan_recovery_external_reaudit_missing")
    elif obj.get("scan_recovery_external_reaudit_sha256") != sha256_file(reaudit_path):
        errors.append("scan_recovery_external_reaudit_sha_mismatch")
    else:
        try:
            validate_scan_recovery_external_reaudit_pass_v15(load_json(reaudit_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))
    if approval_path.is_file():
        approval = load_json(approval_path)
        if obj.get("code_lock_git_revision") != approval.get("code_lock_git_revision"):
            errors.append("scan_recovery_post_audit_auth_code_lock_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def evaluate_scan_recovery_execute_readiness_v15(*, root: Path = _ROOT) -> dict[str, Any]:
    """Dry-plan readiness for scan-recovery --execute (not extraction launch)."""
    errors: list[str] = []
    reaudit_path = root / D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL
    auth_path = root / D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL
    if not reaudit_path.is_file():
        errors.append("scan_recovery_external_reaudit_v15_missing")
    else:
        try:
            validate_scan_recovery_external_reaudit_pass_v15(load_json(reaudit_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))
    if not auth_path.is_file():
        errors.append("scan_recovery_post_audit_auth_v15_missing")
    else:
        try:
            validate_scan_recovery_post_audit_auth_v15(load_json(auth_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))
    return {
        "execution_ready": len(errors) == 0,
        "errors": errors,
        "scan_recovery_external_reaudit_rel": D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL,
        "scan_recovery_post_audit_auth_rel": D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL,
    }


_SCAN_RECOVERY_ATTEMPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SCAN_RECOVERY_ATTEMPTS_ROOT_REL = f"{EVIDENCE_ROOT_V15_REL}/scan_recovery/attempts"


def validate_scan_recovery_attempt_id(attempt_id: str) -> str:
    """Reject absolute paths, separators, and '..' traversal in attempt ids."""
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("attempt_id_empty")
    aid = attempt_id.strip()
    if aid.startswith("/") or aid.startswith("\\") or (len(aid) >= 2 and aid[1] == ":"):
        raise ValueError("attempt_id_absolute_forbidden")
    if "/" in aid or "\\" in aid or ".." in aid:
        raise ValueError("attempt_id_path_traversal_forbidden")
    if not _SCAN_RECOVERY_ATTEMPT_ID_RE.match(aid):
        raise ValueError("attempt_id_invalid_charset")
    return aid


def resolve_scan_recovery_attempt_dir(
    attempt_id: str, *, root: Path = _ROOT
) -> tuple[str, Path, str]:
    """Return validated attempt_id, resolved attempt dir, and repo-relative posix path.

    Guarantees resolved path is contained under the canonical attempts root.
    """
    aid = validate_scan_recovery_attempt_id(attempt_id)
    root_r = Path(root).resolve()
    attempts_root = (root_r / SCAN_RECOVERY_ATTEMPTS_ROOT_REL).resolve()
    attempt_rt = (attempts_root / aid).resolve()
    try:
        attempt_rt.relative_to(attempts_root)
    except ValueError as exc:
        raise ValueError("attempt_id_outside_attempts_root") from exc
    if attempt_rt == attempts_root:
        raise ValueError("attempt_id_resolves_to_attempts_root")
    rel = str(attempt_rt.relative_to(root_r)).replace("\\", "/")
    return aid, attempt_rt, rel


def build_scan_recovery_attempt_auth_binds(*, root: Path = _ROOT) -> dict[str, Any]:
    """Exact SHA/HEAD binds proving which authorization an attempt executed under.

    Artifact digests are Git blob SHAs at execution HEAD (not worktree bytes),
    so CRLF checkout cannot forge a divergent bind.
    """
    root_r = Path(root).resolve()
    execution_head = git_rev(root_r)
    if not git_commit_exists(execution_head, root=root_r):
        raise ValueError("scan_recovery_execution_head_missing")

    required = (
        ("approval_v15_sha256", D0_FULL_CORPUS_APPROVAL_V15_REL),
        ("execution_protocol_v15_sha256", D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        ("run_contract_v15_sha256", D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        (
            "scan_recovery_external_reaudit_sha256",
            D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL,
        ),
        (
            "scan_recovery_post_audit_auth_sha256",
            D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL,
        ),
    )
    binds: dict[str, Any] = {"execution_head": execution_head}
    for field, rel in required:
        try:
            binds[field] = _git_blob_sha256(execution_head, rel, root=root_r)
        except FileNotFoundError as exc:
            raise ValueError(f"scan_recovery_auth_bind_missing:{field}") from exc
    approval = json.loads(
        subprocess.check_output(
            ["git", "show", f"{execution_head}:{D0_FULL_CORPUS_APPROVAL_V15_REL}"],
            cwd=root_r,
        ).decode("utf-8")
    )
    binds["code_lock_git_revision"] = approval.get("code_lock_git_revision")
    binds["executable_tree_sha256"] = approval.get("executable_tree_sha256")
    return binds


_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA64_RE = re.compile(r"^[0-9a-f]{64}$")


def _git_blob_sha256(rev: str, rel: str, *, root: Path) -> str:
    try:
        blob = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=root)
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"git_blob_missing:{rel}@{rev}") from exc
    return hashlib.sha256(blob).hexdigest()


def _git_rev_parse(name: str, *, root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", name], cwd=root, text=True
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"git_rev_parse_failed:{name}") from exc


def preflight_scan_recovery_execute(*, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed provenance gate before creating any attempt directory.

    Locks the executable working tree to the sealed code-lock tree and requires
    HEAD == origin/main with an empty porcelain status.
    """
    root_r = Path(root).resolve()
    errors: list[str] = []

    try:
        head = git_rev(root_r)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"scan_recovery_preflight_head_unreadable:{exc}") from exc
    try:
        origin_main = _git_rev_parse("origin/main", root=root_r)
    except ValueError as exc:
        raise ValueError("scan_recovery_preflight_origin_main_missing") from exc
    if head != origin_main:
        errors.append(f"scan_recovery_preflight_head_ne_origin_main:{head}!={origin_main}")

    porcelain = git_porcelain(root_r).strip()
    if porcelain:
        errors.append("scan_recovery_preflight_dirty_worktree")

    approval_path = root_r / D0_FULL_CORPUS_APPROVAL_V15_REL
    if not approval_path.is_file():
        errors.append("scan_recovery_preflight_approval_missing")
        raise ValueError(";".join(errors))
    try:
        approval = load_json(approval_path)
        validate_approval_v15(approval, root=root_r)
    except ValueError as exc:
        errors.append(f"scan_recovery_preflight_approval_invalid:{exc}")
        raise ValueError(";".join(errors)) from exc

    code_lock = approval.get("code_lock_git_revision")
    sealed_tree = approval.get("executable_tree_sha256")
    if not isinstance(code_lock, str) or not _SHA40_RE.match(code_lock):
        errors.append("scan_recovery_preflight_code_lock_not_sha40")
    elif not git_commit_exists(code_lock, root=root_r):
        errors.append("scan_recovery_preflight_code_lock_missing")
    if not isinstance(sealed_tree, str) or not _SHA64_RE.match(sealed_tree):
        errors.append("scan_recovery_preflight_sealed_tree_not_sha64")

    if not errors:
        # Disk/HEAD executable bytes must match the sealed code-lock (CRLF-safe).
        for rel in FULL_CORPUS_EXECUTABLE_PATHS_V15:
            try:
                subprocess.check_call(
                    ["git", "diff", "--quiet", code_lock, "--", rel],
                    cwd=root_r,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                errors.append(f"scan_recovery_preflight_executable_drift:{rel}")
                break
        try:
            tree_at_lock = executable_tree_sha256_at_revision(
                FULL_CORPUS_EXECUTABLE_PATHS_V15, code_lock, root=root_r
            )
            tree_at_head = executable_tree_sha256_at_revision(
                FULL_CORPUS_EXECUTABLE_PATHS_V15, head, root=root_r
            )
        except FileNotFoundError as exc:
            errors.append(f"scan_recovery_preflight_tree_error:{exc}")
        else:
            if tree_at_lock != sealed_tree:
                errors.append("scan_recovery_preflight_code_lock_tree_mismatch")
            if tree_at_head != sealed_tree:
                errors.append("scan_recovery_preflight_head_tree_mismatch")

    # External PASS + post-auth against current sealed artifacts.
    readiness = evaluate_scan_recovery_execute_readiness_v15(root=root_r)
    if not readiness.get("execution_ready"):
        for err in readiness.get("errors") or []:
            errors.append(f"scan_recovery_preflight_auth:{err}")

    if errors:
        raise ValueError(";".join(errors))
    return {
        "ok": True,
        "execution_head": head,
        "origin_main": origin_main,
        "code_lock_git_revision": code_lock,
        "executable_tree_sha256": sealed_tree,
    }


def validate_scan_recovery_attempt_auth_binds(
    obj: dict[str, Any], *, root: Path = _ROOT
) -> dict[str, Any]:
    """Fail closed: recompute Git blob SHAs at execution_head; reject forgeries."""
    errors: list[str] = []
    root_r = Path(root).resolve()

    execution_head = obj.get("execution_head")
    if not isinstance(execution_head, str) or not _SHA40_RE.match(execution_head):
        errors.append("attempt_auth_bind_execution_head_not_sha40")
    elif not git_commit_exists(execution_head, root=root_r):
        errors.append("attempt_auth_bind_execution_head_missing")

    code_lock = obj.get("code_lock_git_revision")
    if not isinstance(code_lock, str) or not _SHA40_RE.match(code_lock):
        errors.append("attempt_auth_bind_code_lock_not_sha40")
    elif not git_commit_exists(code_lock, root=root_r):
        errors.append("attempt_auth_bind_code_lock_missing")

    tree = obj.get("executable_tree_sha256")
    if not isinstance(tree, str) or not _SHA64_RE.match(tree):
        errors.append("attempt_auth_bind_executable_tree_not_sha64")

    sha_fields = (
        ("approval_v15_sha256", D0_FULL_CORPUS_APPROVAL_V15_REL),
        ("execution_protocol_v15_sha256", D0_FULL_CORPUS_EXEC_PROTOCOL_V15_REL),
        ("run_contract_v15_sha256", D0_FULL_CORPUS_RUN_CONTRACT_V15_REL),
        (
            "scan_recovery_external_reaudit_sha256",
            D0_FULL_CORPUS_SCAN_RECOVERY_EXTERNAL_REAUDIT_V15_REL,
        ),
        (
            "scan_recovery_post_audit_auth_sha256",
            D0_FULL_CORPUS_SCAN_RECOVERY_POST_AUDIT_AUTH_V15_REL,
        ),
    )
    for field, _rel in sha_fields:
        val = obj.get(field)
        if not isinstance(val, str) or not _SHA64_RE.match(val):
            errors.append(f"attempt_auth_bind_{field}_not_sha64")

    if errors:
        raise ValueError(";".join(errors))

    for field, rel in sha_fields:
        try:
            observed = _git_blob_sha256(execution_head, rel, root=root_r)
        except FileNotFoundError:
            errors.append(f"attempt_auth_bind_blob_missing:{field}")
            continue
        if obj.get(field) != observed:
            errors.append(f"attempt_auth_bind_forged_or_stale:{field}")

    # Approval blob at execution_head must carry the declared code-lock/tree.
    try:
        approval_blob = subprocess.check_output(
            ["git", "show", f"{execution_head}:{D0_FULL_CORPUS_APPROVAL_V15_REL}"],
            cwd=root_r,
        ).decode("utf-8")
        approval = json.loads(approval_blob)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        errors.append("attempt_auth_bind_approval_unreadable_at_head")
        raise ValueError(";".join(errors))

    if approval.get("code_lock_git_revision") != code_lock:
        errors.append("attempt_auth_bind_code_lock_vs_approval_blob")
    if approval.get("executable_tree_sha256") != tree:
        errors.append("attempt_auth_bind_tree_vs_approval_blob")

    try:
        tree_at_head = executable_tree_sha256_at_revision(
            FULL_CORPUS_EXECUTABLE_PATHS_V15, execution_head, root=root_r
        )
        tree_at_lock = executable_tree_sha256_at_revision(
            FULL_CORPUS_EXECUTABLE_PATHS_V15, code_lock, root=root_r
        )
    except FileNotFoundError as exc:
        errors.append(f"attempt_auth_bind_tree_error:{exc}")
    else:
        if tree_at_lock != tree:
            errors.append("attempt_auth_bind_code_lock_tree_mismatch")
        if tree_at_head != tree:
            errors.append("attempt_auth_bind_execution_head_tree_mismatch")

    # Reject superseded/stale PASS relative to current worktree tip when porcelain
    # is clean and HEAD matches execution_head (execute-time invariant).
    try:
        current_head = git_rev(root_r)
    except Exception:  # noqa: BLE001
        current_head = None
    if current_head == execution_head:
        for field, rel in sha_fields:
            path = root_r / rel
            if not path.is_file():
                errors.append(f"attempt_auth_bind_worktree_missing:{field}")
                continue
            # Compare via git blob at HEAD (same as worktree index when clean).
            try:
                tip_sha = _git_blob_sha256(current_head, rel, root=root_r)
            except FileNotFoundError:
                errors.append(f"attempt_auth_bind_worktree_blob_missing:{field}")
                continue
            if obj.get(field) != tip_sha:
                errors.append(f"attempt_auth_bind_superseded:{field}")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_post_audit_execution_authorization_v8(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_post_audit_execution_authorization_v8":
        errors.append("bad_post_audit_auth_schema")
    if obj.get("campaign_token") != CAMPAIGN_TOKEN_V8:
        errors.append("post_audit_campaign_token")
    if obj.get("output_version") != OUTPUT_VERSION_V8:
        errors.append("post_audit_output_version")
    if obj.get("launch_authorized") is not True:
        errors.append("launch_not_authorized")
    approval_path = root / D0_FULL_CORPUS_APPROVAL_V8_REL
    protocol_path = root / D0_FULL_CORPUS_EXEC_PROTOCOL_V8_REL
    reaudit_path = root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_V8_REL
    if not approval_path.is_file():
        errors.append("approval_v8_missing")
    elif obj.get("approval_v8_sha256") != sha256_file(approval_path):
        errors.append("approval_v8_sha_mismatch")
    if not protocol_path.is_file():
        errors.append("protocol_v8_missing")
    elif obj.get("execution_protocol_v8_sha256") != sha256_file(protocol_path):
        errors.append("protocol_v8_sha_mismatch")
    expected_reaudit_sha = obj.get("external_reaudit_sha256")
    if not reaudit_path.is_file():
        errors.append("external_reaudit_v8_missing")
    elif expected_reaudit_sha != sha256_file(reaudit_path):
        errors.append("external_reaudit_sha_mismatch")
    else:
        try:
            validate_external_reaudit_pass(load_json(reaudit_path), root=root)
        except ValueError as exc:
            errors.append(str(exc))
    rev = obj.get("code_lock_git_revision")
    if not rev:
        errors.append("post_audit_code_lock_missing")
    else:
        try:
            approval = load_json(approval_path) if approval_path.is_file() else {}
            if approval.get("code_lock_git_revision") != rev:
                errors.append("post_audit_code_lock_vs_approval")
            if obj.get("executable_tree_sha256") != approval.get("executable_tree_sha256"):
                errors.append("post_audit_executable_tree_vs_approval")
            if approval.get("executable_tree_sha256"):
                errors.extend(
                    verify_code_lock_revision_binding(
                        rev,
                        FULL_CORPUS_EXECUTABLE_PATHS_V8,
                        obj.get("executable_tree_sha256"),
                        root=root,
                        require_worktree_match=False,
                    )
                )
        except (json.JSONDecodeError, OSError):
            errors.append("approval_v8_unreadable")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def assert_binaries_gitignored(*, root: Path = _ROOT) -> None:
    probe = binary_root_rel("deadbeefdead") + "/skeleton_v3/000000.npz"
    # git check-ignore
    try:
        subprocess.check_call(
            ["git", "check-ignore", "-q", probe],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"binary_root_not_gitignored:{probe}") from exc


def build_immutable_run_contract_payload(
    *,
    approval: dict[str, Any],
    protocol: dict[str, Any],
    resume: bool,
    root: Path = _ROOT,
) -> dict[str, Any]:
    assert_no_out_of_provenance_sitecustomize(root=root)
    rev = approval["code_lock_git_revision"]
    if not git_commit_exists(rev, root=root):
        raise RuntimeError("approval_code_lock_missing")
    # Tip may advance for evidence/seal commits; executable paths must still match locked rev.
    for rel in FULL_CORPUS_EXECUTABLE_PATHS:
        try:
            subprocess.check_call(
                ["git", "diff", "--quiet", rev, "--", rel],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"executable_path_drift_vs_code_lock:{rel}") from exc
    dirty = git_porcelain(root).strip()
    if dirty:
        # Allow only ignored/untracked under binary/evidence runtime? Fail-closed: require clean tracked tree.
        raise RuntimeError(f"dirty_worktree_forbidden:{dirty[:300]}")
    tree = executable_tree_sha256_at_revision(FULL_CORPUS_EXECUTABLE_PATHS, rev, root=root)
    if tree != approval.get("executable_tree_sha256"):
        raise RuntimeError("executable_tree_mismatch_vs_approval")
    env = env_provenance(root=root)
    return {
        "schema": "ms_sta_v5_d0_full_corpus_immutable_run_contract_v14",
        "campaign_token": CAMPAIGN_TOKEN,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume": bool(resume),
        "output_version": OUTPUT_VERSION,
        "phases": list(PHASES_EXACT),
        "skeleton_arm": "C3",
        "eligible_n": CORPUS_N,
        "split_counts": dict(SPLIT_COUNTS),
        "approval_v14_sha256": sha256_file(root / D0_FULL_CORPUS_APPROVAL_V14_REL),
        "execution_protocol_v14_sha256": sha256_file(root / D0_FULL_CORPUS_EXEC_PROTOCOL_V14_REL),
        "post_audit_execution_auth_v14_sha256": sha256_file(root / D0_FULL_CORPUS_POST_AUDIT_EXECUTION_AUTH_REL),
        "external_reaudit_v14_sha256": sha256_file(root / D0_FULL_CORPUS_EXECUTION_EXTERNAL_REAUDIT_REL),
        "immutable_v13_approval_sha256": EXPECTED_V13_APPROVAL_SHA256,
        "parent_external_reaudit_v13_sha256": EXPECTED_V13_EXTERNAL_REAUDIT_SHA256,
        "r7c_h1_independent_reaudit_sha256": EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
        "full_protocol_sha256": EXPECTED_R4_FULL_PROTOCOL_SHA256,
        "r7c_amendment_sha256": sha256_file(root / FULL_PROTOCOL_R7C_AMENDMENT_REL),
        "backbone_rel": EXPECTED_SLOWFAST_CKPT_REL,
        "backbone_sha256": EXPECTED_SLOWFAST_CKPT_SHA256,
        "code_lock_git_revision": rev,
        "head_at_bind": git_rev(root),
        "executable_tree_sha256": tree,
        "executable_paths": list(FULL_CORPUS_EXECUTABLE_PATHS),
        "binary_root_rel": binary_root_rel(rev, root=root),
        "evidence_root_rel": EVIDENCE_ROOT_REL,
        "env_provenance": env,
        "psutil_pin": PSUTIL_PIN,
        "worker_registry_required": True,
        "sample_transaction_required": True,
        "atomic_manifest_required": True,
        "ambiguity_roi_lock": dict(R7C_AMBIGUITY_ROI_LOCK),
        "enable_ambiguity_aware_prior": True,
        "max_resumes": MAX_RESUMES,
        "quality_thresholds": dict(approval.get("quality_thresholds") or default_quality_thresholds()),
        "auto_chain_to_m0": False,
        "stop_after_pass": "D0_FULL_CORPUS_SCAN_COMPLETE_PASS",
        "stop_after_fail": "D0_FULL_CORPUS_SCAN_COMPLETE_FAIL",
    }
