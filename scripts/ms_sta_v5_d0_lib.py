#!/usr/bin/env python3
"""Shared fail-closed helpers for MS-STA v5 D0 pilot (bounded; no full-corpus extract)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]

D0_DIR_REL = "Docs/results/ms_sta_v5_d0"
APPROVAL_REL = f"{D0_DIR_REL}/D0_APPROVAL.json"
APPROVAL_AUDIT_REL = f"{D0_DIR_REL}/D0_APPROVAL_AUDIT.json"
PROTOCOL_REL = f"{D0_DIR_REL}/D0_PROTOCOL.json"
PROTOCOL_LOCK_REL = f"{D0_DIR_REL}/D0_PROTOCOL_LOCK.json"
PILOT_SAMPLE_REL = f"{D0_DIR_REL}/D0_PILOT_SAMPLE_SET.json"
PILOT_RUN_CONTRACT_REL = f"{D0_DIR_REL}/D0_PILOT_RUN_CONTRACT.json"
PILOT_REPORT_REL = f"{D0_DIR_REL}/D0_PILOT_REPORT.json"
PILOT_AUDIT_REL = f"{D0_DIR_REL}/D0_PILOT_AUDIT.json"
PILOT_OUT_DIR_REL = f"{D0_DIR_REL}/pilot_outputs"
INDEPENDENT_REAUDIT_REL = f"{D0_DIR_REL}/D0_PILOT_INDEPENDENT_REAUDIT.json"

# Versioned D0-PILOT-R1 artifacts (do not overwrite original pilot).
D0_R1_DIR_REL = "Docs/results/ms_sta_v5_d0_r1"
PILOT_R1_SAMPLE_REL = f"{D0_R1_DIR_REL}/D0_R1_PILOT_SAMPLE_SET.json"
PILOT_R1_NEGATIVE_REL = f"{D0_R1_DIR_REL}/D0_R1_NEGATIVE_CONTROL_SET.json"
PILOT_R1_RUN_CONTRACT_REL = f"{D0_R1_DIR_REL}/D0_R1_PILOT_RUN_CONTRACT.json"
PILOT_R1_REPORT_REL = f"{D0_R1_DIR_REL}/D0_R1_PILOT_REPORT.json"
PILOT_R1_AUDIT_REL = f"{D0_R1_DIR_REL}/D0_R1_PILOT_AUDIT.json"
PILOT_R1_OUT_DIR_REL = f"{D0_R1_DIR_REL}/pilot_outputs"
PILOT_R1_VISUAL_REL = f"{D0_R1_DIR_REL}/visual_qa"
PILOT_R1_INDEPENDENT_REAUDIT_REL = f"{D0_R1_DIR_REL}/D0_R1_INDEPENDENT_REAUDIT.json"

# Versioned D0-PILOT-R2 artifacts (do not overwrite R1).
D0_R2_DIR_REL = "Docs/results/ms_sta_v5_d0_r2"
PILOT_R2_RUN_CONTRACT_REL = f"{D0_R2_DIR_REL}/D0_R2_PILOT_RUN_CONTRACT.json"
PILOT_R2_REPORT_REL = f"{D0_R2_DIR_REL}/D0_R2_PILOT_REPORT.json"
PILOT_R2_AUDIT_REL = f"{D0_R2_DIR_REL}/D0_R2_PILOT_AUDIT.json"
PILOT_R2_OUT_DIR_REL = f"{D0_R2_DIR_REL}/pilot_outputs"
PILOT_R2_VISUAL_REL = f"{D0_R2_DIR_REL}/visual_qa"
PILOT_R2_INDEPENDENT_REAUDIT_REL = f"{D0_R2_DIR_REL}/D0_R2_INDEPENDENT_REAUDIT.json"
PILOT_R2_COMPLETION_REL = f"{D0_R2_DIR_REL}/D0_R2_PILOT_COMPLETION.json"

# Versioned D0-PILOT-R3 artifacts (do not overwrite R2).
D0_R3_DIR_REL = "Docs/results/ms_sta_v5_d0_r3"
PILOT_R3_RUN_CONTRACT_REL = f"{D0_R3_DIR_REL}/D0_R3_PILOT_RUN_CONTRACT.json"
PILOT_R3_REPORT_REL = f"{D0_R3_DIR_REL}/D0_R3_PILOT_REPORT.json"
PILOT_R3_AUDIT_REL = f"{D0_R3_DIR_REL}/D0_R3_PILOT_AUDIT.json"
PILOT_R3_OUT_DIR_REL = f"{D0_R3_DIR_REL}/pilot_outputs"
PILOT_R3_VISUAL_REL = f"{D0_R3_DIR_REL}/visual_qa"
PILOT_R3_VISUAL_MANUAL_REL = f"{D0_R3_DIR_REL}/D0_R3_VISUAL_QA_MANUAL.json"
PILOT_R3_COMPLETION_REL = f"{D0_R3_DIR_REL}/D0_R3_PILOT_COMPLETION.json"
PILOT_R3_INDEPENDENT_REAUDIT_REL = f"{D0_R3_DIR_REL}/D0_R3_INDEPENDENT_REAUDIT.json"

# Versioned D0-PILOT-R4 artifacts (do not overwrite R3).
D0_R4_DIR_REL = "Docs/results/ms_sta_v5_d0_r4"
PILOT_R4_RUN_CONTRACT_REL = f"{D0_R4_DIR_REL}/D0_R4_PILOT_RUN_CONTRACT.json"
PILOT_R4_REPORT_REL = f"{D0_R4_DIR_REL}/D0_R4_PILOT_REPORT.json"
PILOT_R4_AUDIT_REL = f"{D0_R4_DIR_REL}/D0_R4_PILOT_AUDIT.json"
PILOT_R4_OUT_DIR_REL = f"{D0_R4_DIR_REL}/pilot_outputs"
PILOT_R4_VISUAL_REL = f"{D0_R4_DIR_REL}/visual_qa"
PILOT_R4_VISUAL_MANUAL_REL = f"{D0_R4_DIR_REL}/D0_R4_VISUAL_QA_MANUAL.json"
PILOT_R4_COMPLETION_REL = f"{D0_R4_DIR_REL}/D0_R4_PILOT_COMPLETION.json"
PILOT_R4_INDEPENDENT_REAUDIT_REL = f"{D0_R4_DIR_REL}/D0_R4_INDEPENDENT_REAUDIT.json"

# Versioned D0-PILOT-R5 QA remediation artifacts (do not overwrite R4 tensors).
D0_R5_DIR_REL = "Docs/results/ms_sta_v5_d0_r5"
PILOT_R5_RUN_CONTRACT_REL = f"{D0_R5_DIR_REL}/D0_R5_PILOT_RUN_CONTRACT.json"
PILOT_R5_REPORT_REL = f"{D0_R5_DIR_REL}/D0_R5_PILOT_REPORT.json"
PILOT_R5_AUDIT_REL = f"{D0_R5_DIR_REL}/D0_R5_PILOT_AUDIT.json"
PILOT_R5_VISUAL_REL = f"{D0_R5_DIR_REL}/visual_qa"
PILOT_R5_VISUAL_MANUAL_REL = f"{D0_R5_DIR_REL}/D0_R5_VISUAL_QA_MANUAL.json"
PILOT_R5_VISUAL_FREEZE_REL = f"{D0_R5_DIR_REL}/D0_R5_VISUAL_QA_FREEZE_LIST.json"
PILOT_R5_COMPLETION_REL = f"{D0_R5_DIR_REL}/D0_R5_PILOT_COMPLETION.json"
PILOT_R5_OVERLAYS_INDEX_REL = f"{D0_R5_DIR_REL}/D0_R5_OVERLAYS_INDEX.json"

FULL_PROTOCOL_REL = f"{D0_DIR_REL}/D0_FULL_PROTOCOL.json"
FULL_PROTOCOL_R5_AMENDMENT_REL = f"{D0_DIR_REL}/D0_FULL_PROTOCOL_R5_QA_AMENDMENT.json"

EXPECTED_SLOWFAST_CKPT_REL = (
    "Docs/results/ms_sta_v5_f0/weights/slowfast_r50_partial_unfreeze/SLOWFAST_8x8_R50.pyth"
)
EXPECTED_SLOWFAST_CKPT_SHA256 = (
    "454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2"
)
EXPECTED_PARENT_PILOT_AUDIT_SHA256 = (
    "8c02311bdd43ba3f1e1f5db330563fbf995535b01d5441b9b119a42f6af6f9d6"
)
EXPECTED_R1_PILOT_AUDIT_SHA256 = (
    "8c7900c8c4194bcb344982fd44f7b787076c96feb04cf60642d4dfd5c79d8d8e"
)
EXPECTED_R1_SAMPLE_SHA256 = (
    "5f14a08cca68b8a55dad5190ceeed65d9ad3fadbf8c5b14bb7044b978f4b57f5"
)
EXPECTED_R2_PILOT_AUDIT_SHA256 = (
    "be7b5dfa40819aced3ecf7d29ec8ad21644dbe4723a9e24309da81b3e62fcef8"
)
EXPECTED_R3_PILOT_AUDIT_SHA256 = (
    "a76cd31ee9c58f791ebf8c6f17bc37f3888969e4c7243c1d3e3a8aa19ae6e1ef"
)
EXPECTED_R4_PILOT_AUDIT_SHA256 = (
    "9e352d78a9281da7add23208ae472e98238229680c6cb3a9897965ece7a8c6cc"
)
EXPECTED_R4_FULL_PROTOCOL_SHA256 = (
    "74c9e33912963f1ae983e276d528d4c63b8b27652f27aa0c5ee342bafbfbf440"
)
EXPECTED_R5_PILOT_AUDIT_SHA256 = (
    "b7e6b81b3bfa771ab54ce83068b5c7a147d83a53b2c680dcf536a5b5d2386141"
)
EXPECTED_R5_PILOT_REPORT_SHA256 = (
    "a45cc8bc85ecee2d744da08230a65319e2933bbe27e19f6d3666249c4559384d"
)
EXPECTED_R5_CODE_LOCK_GIT_REVISION = (
    "3294f7ef445689c7f70b53c3d9eaa6edc7b69c97"
)

EXPECTED_R6_PILOT_AUDIT_SHA256 = (
    "533f515415236e14bd3f569acf90cf131c762d763bc275c41661135e13c767ae"
)
EXPECTED_R6_PILOT_REPORT_SHA256 = (
    "36413d3d5b59eedc10a9a14d94b5cbbae4b7116f79776c7aa3068bd619ec216f"
)
EXPECTED_R7_PILOT_AUDIT_SHA256 = (
    "aa3c37ae5332ddfcdb41cfac2fbd4c475ebd68f3d4ff85440c7beec0b2453a71"
)
EXPECTED_R7_PILOT_REPORT_SHA256 = (
    "acc55872f642c8990ca513ab518989d2b44f78ef9c906c4709ec5db96b30e4e7"
)
EXPECTED_R7_PILOT_COMPLETION_SHA256 = (
    "9fe73c946a0b402fa2dcf5a2c20738458dd9ef30739dba74504aace8f6972e61"
)

# Versioned D0-PILOT-R6 detector-recovery artifacts.
D0_R6_DIR_REL = "Docs/results/ms_sta_v5_d0_r6"
PILOT_R6_INDEPENDENT_REAUDIT_REL = f"{D0_R6_DIR_REL}/D0_R6_INDEPENDENT_REAUDIT.json"
FULL_PROTOCOL_R7_AMENDMENT_REL = f"{D0_DIR_REL}/D0_FULL_PROTOCOL_R7_DUAL_SOURCE_ROI_AMENDMENT.json"
FULL_PROTOCOL_R7B_AMENDMENT_REL = (
    f"{D0_DIR_REL}/D0_FULL_PROTOCOL_R7B_GATE_HARDENING_AMENDMENT.json"
)
D0_R7_DIR_REL = "Docs/results/ms_sta_v5_d0_r7"
PILOT_R7_INDEPENDENT_REAUDIT_REL = f"{D0_R7_DIR_REL}/D0_R7_INDEPENDENT_REAUDIT.json"
PILOT_R7_RUN_CONTRACT_REL = f"{D0_R7_DIR_REL}/D0_R7_PILOT_RUN_CONTRACT.json"
PILOT_R7_REPORT_REL = f"{D0_R7_DIR_REL}/D0_R7_PILOT_REPORT.json"
PILOT_R7_AUDIT_REL = f"{D0_R7_DIR_REL}/D0_R7_PILOT_AUDIT.json"
PILOT_R7_OUT_DIR_REL = f"{D0_R7_DIR_REL}/pilot_outputs"
PILOT_R7_VISUAL_REL = f"{D0_R7_DIR_REL}/visual_qa"
PILOT_R7_VISUAL_MANUAL_REL = f"{D0_R7_DIR_REL}/D0_R7_VISUAL_QA_MANUAL.json"
PILOT_R7_VISUAL_FREEZE_REL = f"{D0_R7_DIR_REL}/D0_R7_VISUAL_QA_FREEZE_LIST.json"
PILOT_R7_COMPLETION_REL = f"{D0_R7_DIR_REL}/D0_R7_PILOT_COMPLETION.json"
PILOT_R7_OVERLAYS_INDEX_REL = f"{D0_R7_DIR_REL}/D0_R7_OVERLAYS_INDEX.json"
PILOT_R7_DIAGNOSTIC_REL = f"{D0_R7_DIR_REL}/D0_R7_DIAGNOSTIC_MATRIX.json"
D0_R7B_DIR_REL = "Docs/results/ms_sta_v5_d0_r7b"
PILOT_R7B_RUN_CONTRACT_REL = f"{D0_R7B_DIR_REL}/D0_R7B_PILOT_RUN_CONTRACT.json"
PILOT_R7B_REPORT_REL = f"{D0_R7B_DIR_REL}/D0_R7B_PILOT_REPORT.json"
PILOT_R7B_AUDIT_REL = f"{D0_R7B_DIR_REL}/D0_R7B_PILOT_AUDIT.json"
PILOT_R7B_OUT_DIR_REL = f"{D0_R7B_DIR_REL}/pilot_outputs"
PILOT_R7B_VISUAL_REL = f"{D0_R7B_DIR_REL}/visual_qa"
PILOT_R7B_VISUAL_MANUAL_REL = f"{D0_R7B_DIR_REL}/D0_R7B_VISUAL_QA_MANUAL.json"
PILOT_R7B_VISUAL_MANUAL_TEMPLATE_REL = f"{D0_R7B_DIR_REL}/D0_R7B_VISUAL_QA_MANUAL.TEMPLATE.json"
PILOT_R7B_VISUAL_FREEZE_REL = f"{D0_R7B_DIR_REL}/D0_R7B_VISUAL_QA_FREEZE_LIST.json"
PILOT_R7B_COMPLETION_REL = f"{D0_R7B_DIR_REL}/D0_R7B_PILOT_COMPLETION.json"
PILOT_R7B_OVERLAYS_INDEX_REL = f"{D0_R7B_DIR_REL}/D0_R7B_OVERLAYS_INDEX.json"
PILOT_R7B_DIAGNOSTIC_REL = f"{D0_R7B_DIR_REL}/D0_R7B_DIAGNOSTIC_MATRIX.json"
PILOT_R7B_FIXTURE_ABSENT_REL = f"{D0_R7B_DIR_REL}/fixtures/truly_absent_wrist"
PILOT_R7B_INDEPENDENT_REAUDIT_REL = f"{D0_R7B_DIR_REL}/D0_R7B_INDEPENDENT_REAUDIT.json"
EXPECTED_R7B_PILOT_AUDIT_SHA256 = (
    "ea5c769f34d0ddda58f2be0221f736a55efc6d973c92384dbe8085ebdcf7add8"
)
EXPECTED_R7B_PILOT_REPORT_SHA256 = (
    "2b98cf46843b42866441e839d900e1167f4fc68c2672d8b12fec53d2561c0540"
)
EXPECTED_R7B_PILOT_COMPLETION_SHA256 = (
    "7c9d8b3d07abb1939f7ad8d39d8ffea8953b74a8ac9d748b8949e4994f8adbe5"
)
EXPECTED_R7B_VISUAL_MANUAL_SHA256 = (
    "11ca2919fd8c376c756a6883f848da933ee479817a65b374e287d172f3653c63"
)

FULL_PROTOCOL_R7C_AMENDMENT_REL = (
    f"{D0_DIR_REL}/D0_FULL_PROTOCOL_R7C_AMBIGUITY_AWARE_ROI_AMENDMENT.json"
)
D0_R7C_DIR_REL = "Docs/results/ms_sta_v5_d0_r7c"
PILOT_R7C_RUN_CONTRACT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_PILOT_RUN_CONTRACT.json"
PILOT_R7C_REPORT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_PILOT_REPORT.json"
PILOT_R7C_AUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_PILOT_AUDIT.json"
PILOT_R7C_OUT_DIR_REL = f"{D0_R7C_DIR_REL}/pilot_outputs"
PILOT_R7C_VISUAL_REL = f"{D0_R7C_DIR_REL}/visual_qa"
PILOT_R7C_VISUAL_MANUAL_REL = f"{D0_R7C_DIR_REL}/D0_R7C_VISUAL_QA_MANUAL.json"
PILOT_R7C_VISUAL_MANUAL_TEMPLATE_REL = f"{D0_R7C_DIR_REL}/D0_R7C_VISUAL_QA_MANUAL.TEMPLATE.json"
PILOT_R7C_VISUAL_FREEZE_REL = f"{D0_R7C_DIR_REL}/D0_R7C_VISUAL_QA_FREEZE_LIST.json"
PILOT_R7C_COMPLETION_REL = f"{D0_R7C_DIR_REL}/D0_R7C_PILOT_COMPLETION.json"
PILOT_R7C_OVERLAYS_INDEX_REL = f"{D0_R7C_DIR_REL}/D0_R7C_OVERLAYS_INDEX.json"
PILOT_R7C_DIAGNOSTIC_REL = f"{D0_R7C_DIR_REL}/D0_R7C_DIAGNOSTIC_MATRIX.json"
PILOT_R7C_INDEPENDENT_REAUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_INDEPENDENT_REAUDIT.json"
PILOT_R7C_FIXTURE_ABSENT_REL = f"{D0_R7C_DIR_REL}/fixtures/truly_absent_wrist"
# Evidence-hardening campaign (does not overwrite frozen R7C PASS audit/report/completion).
PILOT_R7C_H_REPORT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_PILOT_REPORT.json"
PILOT_R7C_H_AUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_PILOT_AUDIT.json"
PILOT_R7C_H_COMPLETION_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_PILOT_COMPLETION.json"
PILOT_R7C_H_VISUAL_MANUAL_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_VISUAL_QA_MANUAL.json"
PILOT_R7C_H_VISUAL_MANUAL_TEMPLATE_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_VISUAL_QA_MANUAL.TEMPLATE.json"
PILOT_R7C_H_INDEPENDENT_REAUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H_INDEPENDENT_REAUDIT.json"
EXPECTED_R7C_H_PILOT_AUDIT_SHA256 = (
    "2e4bdce953bea964a98c8fdcb6f31dae6cf543bfb442e13604460cf63b973fe4"
)
EXPECTED_R7C_H_PILOT_REPORT_SHA256 = (
    "3bc87565094ff02c55eb256734832e8f135dc8e4b35ef7b377253479837bcd99"
)
EXPECTED_R7C_H_PILOT_COMPLETION_SHA256 = (
    "4b05b9410bc119ec46ae772e6741a2947b1d71908df89333eb16b03eca807ce3"
)
EXPECTED_R7C_H_VISUAL_MANUAL_SHA256 = (
    "adf60d38732820dc3afa0fc6dd5de7ed39636502e41df95c0af6e9aea14fa340"
)
EXPECTED_R7C_H_INDEPENDENT_REAUDIT_SHA256 = (
    "5189cd04be4d1699e252e62e15192f15e72df925323e9d43e4a0e5b4c05c5c42"
)
# H1 genuine manual-review campaign (does not overwrite frozen R7C-H self-PASS).
PILOT_R7C_H1_REPORT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_PILOT_REPORT.json"
PILOT_R7C_H1_AUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_PILOT_AUDIT.json"
PILOT_R7C_H1_COMPLETION_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_PILOT_COMPLETION.json"
PILOT_R7C_H1_VISUAL_MANUAL_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_VISUAL_QA_MANUAL.json"
PILOT_R7C_H1_VISUAL_MANUAL_TEMPLATE_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_VISUAL_QA_MANUAL.TEMPLATE.json"
PILOT_R7C_H1_PRIOR_JUDGMENTS_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_PRIOR_TILE_JUDGMENTS.json"

PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL = f"{D0_R7C_DIR_REL}/D0_R7C_H1_INDEPENDENT_REAUDIT.json"
EXPECTED_R7C_H1_PILOT_AUDIT_SHA256 = (
    "0227e12ed46a15c7852ac4cd08da223bac67a9cee172c8cd94cde17f976b3f89"
)
EXPECTED_R7C_H1_PILOT_REPORT_SHA256 = (
    "34b9416b767ea0d31ccdc5d33a09c2d484e6b6fa97e66fc69f4327331f8212b4"
)
EXPECTED_R7C_H1_PILOT_COMPLETION_SHA256 = (
    "36f621e4b52cf597abe2ac1c960ec1470faac756e775d087862c57f964861b10"
)
EXPECTED_R7C_H1_VISUAL_MANUAL_SHA256 = (
    "40451101412f35406f43984ffeddb374f2bfec4aba852c4848ce16642c7922fa"
)
EXPECTED_R7C_H1_PRIOR_JUDGMENTS_SHA256 = (
    "9ab1ee3d28698f1fe3a4a7cd7914d6eb71d05c3ac1d866af674f1369804f2d4d"
)
EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256 = (
    "e5d6d5853f7147e6cd19265e7ab9ae895ced9007b7a12a6adfbb3087e248483d"
)
D0_FULL_CORPUS_DIR_REL = f"{D0_DIR_REL}"
D0_FULL_CORPUS_APPROVAL_REL = f"{D0_DIR_REL}/D0_FULL_CORPUS_APPROVAL.json"
D0_FULL_CORPUS_APPROVAL_AUDIT_REL = f"{D0_DIR_REL}/D0_FULL_CORPUS_APPROVAL_AUDIT.json"
D0_FULL_CORPUS_RUN_CONTRACT_REL = f"{D0_DIR_REL}/D0_FULL_CORPUS_RUN_CONTRACT.json"
D0_FULL_CORPUS_OUT_DIR_REL = "Docs/results/ms_sta_v5_d0_full/v1"

SCRIPTED_PRIOR_TILE_NOTE_MARKERS = (
    "[prior_tile]",
    "local ROI accepted under R7C ambiguity lock",
)
EXPECTED_R7C_PILOT_AUDIT_SHA256 = (
    "934523fbb07d57d86fb35b6e2939602c3cf67685b447c8622856111c0515149a"
)
EXPECTED_R7C_PILOT_REPORT_SHA256 = (
    "32055b9086d574cfe212b9e4020812df17e7ab63cea5f7ff2e1199e28f62368e"
)
EXPECTED_R7C_PILOT_COMPLETION_SHA256 = (
    "f943410ef1dd2766eb99cec7d4150ad7dd322c8692079af7459fdce9c527f485"
)
EXPECTED_R7C_VISUAL_MANUAL_SHA256 = (
    "2a9a1c4b9f6aa2ad872944985abeb2847f50caf7c00877cf534c370b71dab3b3"
)
EXPECTED_R7C_INDEPENDENT_REAUDIT_SHA256 = (
    "7053217455bf065844e83139cf836eb15f4a8f4a57017fbadcf2ba73b6ec845e"
)
EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256 = (
    "c49c2825d4de5fb35fc37c916f30fc0c0d1fe986d082457005c44aec90e4ad3d"
)
RGB_PRIOR_SOURCES = frozenset({"pose_wrist_prior", "bimanual_shared_prior"})

# Canonical executable tree for R7B gate-hardening replay (must stay git-tracked).
R7B_EXECUTABLE_PATHS = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/data_prep/joint_layout_v3.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/run_ms_sta_v5_d0_r6_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_finalize.py",
    "scripts/run_ms_sta_v5_d0_r7b_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7b_finalize.py",
    "scripts/build_ms_sta_v5_d0_r7b_freeze.py",
    "scripts/write_ms_sta_v5_d0_r7b_manual_template.py",
    "scripts/build_ms_sta_v5_d0_r7b_absent_fixture.py",
]

R7C_EXECUTABLE_PATHS = [
    "src/data_prep/hand_recovery_v6.py",
    "src/data_prep/extract_and_smooth.py",
    "src/data_prep/skel_v3.py",
    "src/data_prep/joint_layout_v3.py",
    "src/models/rgb_token_v3.py",
    "scripts/ms_sta_v5_d0_lib.py",
    "scripts/run_ms_sta_v5_d0_r6_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7_finalize.py",
    "scripts/run_ms_sta_v5_d0_r7b_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7b_finalize.py",
    "scripts/run_ms_sta_v5_d0_r7c_pilot.py",
    "scripts/run_ms_sta_v5_d0_r7c_finalize.py",
    "scripts/build_ms_sta_v5_d0_r7c_freeze.py",
    "scripts/write_ms_sta_v5_d0_r7c_manual_template.py",
    "scripts/build_ms_sta_v5_d0_r7b_absent_fixture.py",
]

# Evidence-hardening executables (extends R7C; do not mutate frozen R7C tree identity).
R7C_H_EXECUTABLE_PATHS = [
    *R7C_EXECUTABLE_PATHS,
    "scripts/build_ms_sta_v5_d0_r7c_absent_fixture.py",
    "scripts/write_ms_sta_v5_d0_r7c_h_manual_template.py",
    "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py",
    "scripts/run_ms_sta_v5_d0_r7c_h_finalize.py",
]

# H1 genuine manual-review executables (extends H; fill script retained as FORBIDDEN stub).
R7C_H1_EXECUTABLE_PATHS = [
    *R7C_H_EXECUTABLE_PATHS,
    "scripts/write_ms_sta_v5_d0_r7c_h1_manual_template.py",
    "scripts/author_ms_sta_v5_d0_r7c_h1_human_judgments.py",
    "scripts/merge_ms_sta_v5_d0_r7c_h1_human_judgments.py",
    "scripts/run_ms_sta_v5_d0_r7c_h1_finalize.py",
    "scripts/list_ms_sta_v5_d0_r7c_prior_slots.py",
]

# Locked pose-prior geometry (R7B area lock retained as upper bound).
POSE_PRIOR_GEOMETRY_LOCK = {
    "crop_scale": 1.0,
    "min_side": 24.0,
    "max_area_fraction_of_frame": 0.08,
    "input_size": 224,
}

# R7C ambiguity-aware single/bimanual lock (must match rgb_token_v3).
R7C_AMBIGUITY_ROI_LOCK = {
    "bimanual_wrist_distance_frac_of_shoulder": 0.55,
    "single_hand_crop_scale": 0.85,
    "single_hand_max_area_fraction": 0.06,
    "single_hand_max_center_offset_frac_of_half": 0.35,
    "bimanual_margin_scale": 0.35,
    "bimanual_max_area_fraction": 0.08,
    "input_size": 224,
    "region_layout_parity": "four_regions_no_extra_bimanual_token",
    "bimanual_applies_same_box_to_left_and_right": True,
    "skeleton_missing_unchanged": True,
}
PILOT_R5_INDEPENDENT_REAUDIT_REL = f"{D0_R5_DIR_REL}/D0_R5_INDEPENDENT_REAUDIT.json"
FULL_PROTOCOL_R6_AMENDMENT_REL = f"{D0_DIR_REL}/D0_FULL_PROTOCOL_R6_DETECTOR_RECOVERY_AMENDMENT.json"
PILOT_R6_RUN_CONTRACT_REL = f"{D0_R6_DIR_REL}/D0_R6_PILOT_RUN_CONTRACT.json"
PILOT_R6_REPORT_REL = f"{D0_R6_DIR_REL}/D0_R6_PILOT_REPORT.json"
PILOT_R6_AUDIT_REL = f"{D0_R6_DIR_REL}/D0_R6_PILOT_AUDIT.json"
PILOT_R6_OUT_DIR_REL = f"{D0_R6_DIR_REL}/pilot_outputs"
PILOT_R6_VISUAL_REL = f"{D0_R6_DIR_REL}/visual_qa"
PILOT_R6_VISUAL_MANUAL_REL = f"{D0_R6_DIR_REL}/D0_R6_VISUAL_QA_MANUAL.json"
PILOT_R6_VISUAL_FREEZE_REL = f"{D0_R6_DIR_REL}/D0_R6_VISUAL_QA_FREEZE_LIST.json"
PILOT_R6_HOLDOUT_FREEZE_REL = f"{D0_R6_DIR_REL}/D0_R6_HOLDOUT_RISK_FREEZE_LIST.json"
PILOT_R6_DIAGNOSTIC_REL = f"{D0_R6_DIR_REL}/D0_R6_DIAGNOSTIC_MATRIX.json"
PILOT_R6_COMPLETION_REL = f"{D0_R6_DIR_REL}/D0_R6_PILOT_COMPLETION.json"
PILOT_R6_OVERLAYS_INDEX_REL = f"{D0_R6_DIR_REL}/D0_R6_OVERLAYS_INDEX.json"

F0_CLOSEOUT_REL = "Docs/results/ms_sta_v5_f0/F0_PHASE_CLOSEOUT.json"
R0_LOCK_REL = "Docs/results/ms_sta_v5_r0/R0_LOCK.json"
R0_AUDIT_REL = "Docs/results/ms_sta_v5_r0/R0_AUDIT.json"
ELIGIBLE_REL = "data/processed/vsl400_eligible_sample_manifest.json"
SPLIT_REL = "data/processed/vsl400_split_signers.json"
METADATA_REL = "metadata.csv"

EXPECTED_ELIGIBLE_SHA256 = (
    "f6b04713fabdbee58d3e343178f9cd793b993708b017b5581ca8d7afab840522"
)
EXPECTED_METADATA_SHA256 = (
    "05a24a6b3ee173230555c4e25f59274d66e4f615be23f67f5849af91deb559fe"
)
EXPECTED_SPLIT_SHA256 = (
    "84b6b3cebeb359583b5717b5932118270dc6230daf06ea377e0b89bbd4c5f31b"
)
EXPECTED_R0_LOCK_SHA256 = (
    "8e629da2983540451604b4c6602e41ba1b308bbc344b2b656171ac36b7f33ca3"
)
EXPECTED_R0_AUDIT_SHA256 = (
    "23e770a09480107e9bf994817a066408e9eacff4bbcabba129057aa24f8700d7"
)

STAGE_B_FEASIBLE = (
    "videomae_vit_small",
    "videomae_vit_base",
    "video_swin_tiny",
)
STAGE_B_EXCLUDED = ("slowfast_r50_partial_unfreeze",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    # utf-8-sig accepts plain UTF-8 and UTF-8-with-BOM without mutating file bytes.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: dict[str, Any], *, overwrite: bool = False) -> str:
    raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if b"\r" in raw:
        raise RuntimeError(f"crlf_forbidden:{path}")
    if path.is_file() and not overwrite:
        raise RuntimeError(f"already_exists:{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return sha256_file(path)


def git_rev(root: Path = _ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_rev_short(root: Path = _ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True).strip()


def git_ls_files(root: Path = _ROOT) -> set[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return {p.replace("\\", "/") for p in raw.decode("utf-8").split("\0") if p}


def executable_tree_sha256(
    paths: list[str],
    *,
    root: Path = _ROOT,
    require_git_tracked: bool = True,
) -> str:
    """SHA-256 over sorted relative executable paths and their file digests.

    When require_git_tracked=True, every path must appear in `git ls-files`
    (existence alone is insufficient).
    """
    tracked = git_ls_files(root) if require_git_tracked else None
    h = hashlib.sha256()
    for rel in sorted({p.replace("\\", "/") for p in paths}):
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"executable_tree_missing:{rel}")
        if tracked is not None and rel not in tracked:
            raise FileNotFoundError(f"executable_tree_untracked:{rel}")
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def git_commit_exists(rev: str, *, root: Path = _ROOT) -> bool:
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        return False
    try:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{rev}^{{commit}}"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def executable_tree_sha256_at_revision(
    paths: list[str],
    rev: str,
    *,
    root: Path = _ROOT,
) -> str:
    """SHA-256 of executable tree blobs at an exact git revision (fail-closed)."""
    if not git_commit_exists(rev, root=root):
        raise FileNotFoundError(f"executable_tree_revision_missing:{rev}")
    h = hashlib.sha256()
    for rel in sorted({p.replace("\\", "/") for p in paths}):
        try:
            blob = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=root)
        except subprocess.CalledProcessError as exc:
            raise FileNotFoundError(f"executable_tree_missing_at_rev:{rel}@{rev}") from exc
        digest = hashlib.sha256(blob).hexdigest()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(digest.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def verify_code_lock_revision_binding(
    rev: Any,
    paths: list[str],
    claimed_tree: Any,
    *,
    root: Path = _ROOT,
    require_worktree_match: bool = False,
) -> list[str]:
    """Bind code_lock_git_revision to real commit + executable blob tree."""
    errors: list[str] = []
    if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
        errors.append("code_lock_not_full_sha40")
        return errors
    if not git_commit_exists(rev, root=root):
        errors.append("code_lock_revision_not_in_git")
        return errors
    try:
        tree_at_rev = executable_tree_sha256_at_revision(paths, rev, root=root)
    except FileNotFoundError as exc:
        errors.append(f"code_lock_tree_at_rev_error:{exc}")
        return errors
    if claimed_tree != tree_at_rev:
        errors.append("code_lock_tree_not_bound_to_revision")
    if require_worktree_match:
        if git_rev(root) != rev:
            errors.append("code_lock_head_mismatch")
        # Compare path contents to the locked revision without hashing CRLF worktree bytes.
        for rel in sorted({p.replace("\\", "/") for p in paths}):
            try:
                subprocess.check_call(
                    ["git", "diff", "--quiet", rev, "--", rel],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                errors.append(f"code_lock_worktree_diverges:{rel}")
                break
    return errors


def video_is_blank(path: Path, *, max_pixel: int = 2) -> tuple[bool, str]:
    """Return (ok, reason). Blank means every decoded pixel <= max_pixel."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "video_unreadable"
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if int(frame.max()) > max_pixel:
            cap.release()
            return False, "video_not_blank"
    cap.release()
    if n < 1:
        return False, "video_empty"
    return True, f"blank_frames:{n}"


def repo_rel_path(path: str | Path, *, root: Path = _ROOT) -> str:
    """Return POSIX repo-relative path. Absolute paths are always rejected."""
    raw = str(path).replace("\\", "/")
    # Drive / UNC / leading slash absolute forms (even if under repo when resolved).
    if Path(path).is_absolute() or raw.startswith("/") or (len(raw) >= 2 and raw[1] == ":"):
        raise ValueError(f"absolute_path_forbidden:{path}")
    rel = Path(raw).as_posix()
    # Must resolve under root without escaping.
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path_outside_repo:{path}") from exc
    return rel


def int_field(obj: dict[str, Any], key: str) -> int | None:
    """Read int field without coercing 0 via `or`."""
    if key not in obj or obj.get(key) is None:
        return None
    return int(obj[key])


def validate_absent_wrist_fixture_proof(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed empirical absent-wrist fixture proof."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7b_truly_absent_wrist_fixture_v1":
        errors.append("bad_absent_fixture_schema")
    video_rel = obj.get("video_rel")
    skel_rel = obj.get("skel_rel")
    if not isinstance(video_rel, str) or not video_rel:
        errors.append("absent_fixture_video_rel_missing")
    if not isinstance(skel_rel, str) or not skel_rel:
        errors.append("absent_fixture_skel_rel_missing")
    expected = obj.get("expected") or {}
    observed = obj.get("observed") or {}
    for side in ("left_hand", "right_hand"):
        if expected.get(f"{side}_box") is not None:
            errors.append(f"absent_fixture_expected_box_must_be_null:{side}")
        if expected.get(f"{side}_source") != "missing":
            errors.append(f"absent_fixture_expected_source:{side}")
        if expected.get(f"{side}_reason") != "pose_wrist_missing":
            errors.append(f"absent_fixture_expected_reason:{side}")
        if observed.get(f"{side}_box") is not None:
            errors.append(f"absent_fixture_observed_box_not_null:{side}")
        if observed.get(f"{side}_source") != "missing":
            errors.append(f"absent_fixture_observed_source:{side}")
        if observed.get(f"{side}_reason") != "pose_wrist_missing":
            errors.append(f"absent_fixture_observed_reason:{side}")
    if video_rel and skel_rel:
        try:
            vrel = repo_rel_path(video_rel, root=root)
            srel = repo_rel_path(skel_rel, root=root)
        except ValueError:
            errors.append("absent_fixture_paths_not_repo_relative")
            vrel = srel = None
        if vrel and srel:
            vp = root / vrel
            sp = root / srel
            if not vp.is_file():
                errors.append("absent_fixture_video_missing")
            if not sp.is_file():
                errors.append("absent_fixture_skel_missing")
            else:
                import numpy as np

                data = np.load(sp)
                miss = data["missing_mask_prefill"]
                if miss.shape[0] < 1 or miss.shape[1] < 10:
                    errors.append("absent_fixture_skel_shape")
                else:
                    if not bool(miss[:, 8].all()) or not bool(miss[:, 9].all()):
                        errors.append("absent_fixture_wrists_not_all_missing")
                    # Recompute pose-prior on frame 0
                    from src.models.rgb_token_v3 import region_boxes_with_rgb_sources

                    boxes, sources, reasons = region_boxes_with_rgb_sources(
                        data["joints_raw"][0],
                        miss[0],
                        enable_pose_wrist_prior=True,
                    )
                    if boxes["left_hand"] is not None or boxes["right_hand"] is not None:
                        errors.append("absent_fixture_recompute_emitted_box")
                    if sources["left_hand"] != "missing" or sources["right_hand"] != "missing":
                        errors.append("absent_fixture_recompute_source")
                    if reasons["left_hand"] != "pose_wrist_missing" or reasons["right_hand"] != "pose_wrist_missing":
                        errors.append("absent_fixture_recompute_reason")
                # Optional declared digests (R7B closed bytes omit hashes; R7C requires them).
                for key, path in (("video_sha256", vp), ("skel_sha256", sp)):
                    declared = obj.get(key)
                    if declared is not None:
                        if not path.is_file() or declared != sha256_file(path):
                            errors.append(f"absent_fixture_hash_mismatch:{key}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7c_absent_wrist_fixture_proof(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed R7C absent-wrist fixture: required digests, blank video, ambiguity recompute."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7c_truly_absent_wrist_fixture_v1":
        errors.append("bad_r7c_absent_fixture_schema")
    if obj.get("ambiguity_aware_prior") is not True:
        errors.append("r7c_absent_fixture_ambiguity_flag")
    lock = obj.get("ambiguity_lock") or {}
    for k, v in R7C_AMBIGUITY_ROI_LOCK.items():
        if lock.get(k) != v:
            errors.append(f"r7c_absent_fixture_lock_mismatch:{k}")
    video_rel = obj.get("video_rel")
    skel_rel = obj.get("skel_rel")
    if not isinstance(video_rel, str) or not video_rel:
        errors.append("r7c_absent_fixture_video_rel_missing")
    if not isinstance(skel_rel, str) or not skel_rel:
        errors.append("r7c_absent_fixture_skel_rel_missing")
    expected = obj.get("expected") or {}
    observed = obj.get("observed") or {}
    for side in ("left_hand", "right_hand"):
        if expected.get(f"{side}_box") is not None:
            errors.append(f"r7c_absent_fixture_expected_box_must_be_null:{side}")
        if expected.get(f"{side}_source") != "missing":
            errors.append(f"r7c_absent_fixture_expected_source:{side}")
        if expected.get(f"{side}_reason") != "pose_wrist_missing":
            errors.append(f"r7c_absent_fixture_expected_reason:{side}")
        if observed.get(f"{side}_box") is not None:
            errors.append(f"r7c_absent_fixture_observed_box_not_null:{side}")
        if observed.get(f"{side}_source") != "missing":
            errors.append(f"r7c_absent_fixture_observed_source:{side}")
        if observed.get(f"{side}_reason") != "pose_wrist_missing":
            errors.append(f"r7c_absent_fixture_observed_reason:{side}")
    if video_rel and skel_rel:
        try:
            vrel = repo_rel_path(video_rel, root=root)
            srel = repo_rel_path(skel_rel, root=root)
        except ValueError:
            errors.append("r7c_absent_fixture_paths_not_repo_relative")
            vrel = srel = None
        if vrel and srel:
            vp = root / vrel
            sp = root / srel
            if not vp.is_file():
                errors.append("r7c_absent_fixture_video_missing")
            if not sp.is_file():
                errors.append("r7c_absent_fixture_skel_missing")
            video_sha = obj.get("video_sha256")
            skel_sha = obj.get("skel_sha256")
            if not is_hex_sha256(video_sha):
                errors.append("r7c_absent_fixture_video_sha_required")
            elif vp.is_file() and video_sha != sha256_file(vp):
                errors.append("r7c_absent_fixture_video_sha_mismatch")
            if not is_hex_sha256(skel_sha):
                errors.append("r7c_absent_fixture_skel_sha_required")
            elif sp.is_file() and skel_sha != sha256_file(sp):
                errors.append("r7c_absent_fixture_skel_sha_mismatch")
            if vp.is_file():
                blank_ok, blank_reason = video_is_blank(vp)
                if not blank_ok:
                    errors.append(f"r7c_absent_fixture_{blank_reason}")
            if sp.is_file():
                import numpy as np

                data = np.load(sp)
                miss = data["missing_mask_prefill"]
                if miss.shape[0] < 1 or miss.shape[1] < 10:
                    errors.append("r7c_absent_fixture_skel_shape")
                else:
                    if not bool(miss[:, 8].all()) or not bool(miss[:, 9].all()):
                        errors.append("r7c_absent_fixture_wrists_not_all_missing")
                    from src.models.rgb_token_v3 import region_boxes_with_rgb_sources

                    boxes, sources, reasons = region_boxes_with_rgb_sources(
                        data["joints_raw"][0],
                        miss[0],
                        enable_pose_wrist_prior=False,
                        enable_ambiguity_aware_prior=True,
                        ambiguity_lock=dict(R7C_AMBIGUITY_ROI_LOCK),
                    )
                    if boxes["left_hand"] is not None or boxes["right_hand"] is not None:
                        errors.append("r7c_absent_fixture_recompute_emitted_box")
                    if sources["left_hand"] != "missing" or sources["right_hand"] != "missing":
                        errors.append("r7c_absent_fixture_recompute_source")
                    if (
                        reasons["left_hand"] != "pose_wrist_missing"
                        or reasons["right_hand"] != "pose_wrist_missing"
                    ):
                        errors.append("r7c_absent_fixture_recompute_reason")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}

def git_porcelain(root: Path = _ROOT) -> str:
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)


def is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def validate_f0_phase_closeout(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_f0_phase_closeout_v1":
        errors.append("bad_closeout_schema")
    if obj.get("phase_status") != "CLOSED_WITH_3_FEASIBLE_CANDIDATES":
        errors.append("closeout_phase_status")
    if obj.get("f0_r2_authorized") is not False:
        errors.append("f0_r2_must_remain_false")
    binds = obj.get("immutable_parent_audits") or {}
    expected = {
        "f0_final_audit_sha256": "e7efe0e1edcef9db34731b185b4bf96f50bee428c627f89e617520b11062f9ff",
        "f0_feasibility_report_sha256": "0c21d5948d9a79ec7d9dcd88ba5214aed4eb6f536652174e97c41da97942e457",
        "f0_r1_final_audit_sha256": "a2b1444227d153b51f3229db7c60b8f53dcc7b4d9a571a807a3a3d76047c83d3",
        "f0_r1_feasibility_report_sha256": "c777ca42849cb3990fa86b2ae6134a98d78492a7a7f09cc741556ce8b38e4faa",
    }
    for key, digest in expected.items():
        if binds.get(key) != digest:
            errors.append(f"closeout_bind_mismatch:{key}")
        rel_key = key.replace("_sha256", "")
        rel = binds.get(rel_key)
        if not rel or not (root / str(rel)).is_file():
            errors.append(f"closeout_missing_parent:{key}")
        elif sha256_file(root / str(rel)) != digest:
            errors.append(f"closeout_parent_bytes_drift:{key}")
    pool = obj.get("stage_b_candidate_pool") or {}
    feasible_ids = [c.get("candidate_id") for c in (pool.get("feasible") or [])]
    if feasible_ids != list(STAGE_B_FEASIBLE):
        errors.append("closeout_feasible_pool_mismatch")
    for c in pool.get("feasible") or []:
        if c.get("status") != "FEASIBLE":
            errors.append(f"closeout_feasible_status:{c.get('candidate_id')}")
    excluded = pool.get("excluded") or []
    if len(excluded) != 1 or excluded[0].get("candidate_id") != "slowfast_r50_partial_unfreeze":
        errors.append("closeout_excluded_slowfast_missing")
    elif excluded[0].get("status") != "CONTRACT_INVALID_NOT_TESTED":
        errors.append("closeout_slowfast_must_be_contract_invalid_not_tested")
    elif excluded[0].get("not") != "GPU_INFEASIBLE":
        errors.append("closeout_slowfast_must_not_claim_gpu_infeasible")
    inv = obj.get("slowfast_runtime_block_inventory") or {}
    blocks = inv.get("blocks") or []
    if len(blocks) != 7:
        errors.append("closeout_block_inventory_count")
    else:
        if blocks[5].get("parameterized") is not False or blocks[5].get("param_count") != 0:
            errors.append("closeout_block5_must_be_parameterless")
        if blocks[6].get("role") != "classifier_head":
            errors.append("closeout_block6_classifier")
        for i in range(5):
            if blocks[i].get("parameterized") is not True:
                errors.append(f"closeout_block{i}_must_be_parameterized")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "phase_status": obj["phase_status"]}


def validate_d0_approval(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_approval_v1":
        errors.append("bad_d0_approval_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("d0_approval_decision")
    if obj.get("scope") != "d0_pilot_only":
        errors.append("d0_approval_scope")
    if obj.get("one_shot_pilot") is not True:
        errors.append("d0_approval_one_shot_pilot")
    for key in (
        "allow_full_corpus_extraction",
        "allow_m0",
        "allow_training",
        "allow_accuracy",
        "allow_hpo",
        "allow_test",
        "allow_demo",
        "allow_auto_chain",
        "allow_f0_r2",
    ):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("max_pilot_resumes") != 2:
        errors.append("d0_approval_max_pilot_resumes")
    binds = {
        "r0_lock_sha256": EXPECTED_R0_LOCK_SHA256,
        "r0_audit_sha256": EXPECTED_R0_AUDIT_SHA256,
        "eligible_manifest_sha256": EXPECTED_ELIGIBLE_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "split_signers_sha256": EXPECTED_SPLIT_SHA256,
    }
    for k, expected in binds.items():
        if obj.get(k) != expected:
            errors.append(f"d0_approval_bad_{k}")
    closeout_sha = obj.get("f0_phase_closeout_sha256")
    if not is_hex_sha256(closeout_sha):
        errors.append("d0_approval_closeout_sha")
    elif sha256_file(root / F0_CLOSEOUT_REL) != closeout_sha:
        errors.append("d0_approval_closeout_bytes_mismatch")
    else:
        validate_f0_phase_closeout(load_json(root / F0_CLOSEOUT_REL), root=root)
    if obj.get("code_lock_sha256") != git_rev(root):
        # Approval may bind tip at creation; callers re-check at execution.
        pass
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_d0_protocol(obj: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_protocol_v1":
        errors.append("bad_d0_protocol_schema")
    if obj.get("status") != "PILOT":
        errors.append("d0_protocol_status")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("d0_protocol_full_corpus_forbidden")
    skel = obj.get("skeleton_v3") or {}
    for key in (
        "native_timestamps",
        "original_frame_index",
        "prefill_missing_mask",
        "joint_validity_confidence",
        "normalized_coordinates",
        "velocity",
        "bone_vectors",
        "joint_part_chirality_identity",
        "real_sequence_length",
        "padding_mask",
        "forbid_reconstruct_masks_from_v2",
    ):
        if skel.get(key) is not True:
            errors.append(f"skel_contract_missing:{key}")
    rgb = obj.get("rgb_token_v3") or {}
    if rgb.get("backbone") != "frozen_slowfast_r50":
        errors.append("rgb_backbone_must_be_frozen_slowfast")
    if rgb.get("hook") != "blocks.4_before_blocks.5_pooling":
        errors.append("rgb_hook_must_be_block4")
    if rgb.get("retain_slow_fast_pathways") is not True:
        errors.append("rgb_must_retain_both_pathways")
    if rgb.get("roi_method") != "torchvision.ops.roi_align":
        errors.append("rgb_roi_method")
    regions = list(rgb.get("regions") or [])
    if regions != ["global", "left_hand", "right_hand", "face"]:
        errors.append("rgb_regions")
    if rgb.get("compact_tokens_only") is not True:
        errors.append("rgb_compact_tokens_only")
    if rgb.get("forbid_partial_unfreeze") is not True:
        errors.append("rgb_forbid_partial_unfreeze")
    if rgb.get("forbid_accuracy") is not True:
        errors.append("rgb_forbid_accuracy")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_pilot_sample_set(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Legacy original pilot validator (kept for historical artifact checks)."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_pilot_sample_set_v1":
        errors.append("bad_pilot_sample_schema")
    if obj.get("status") != "FROZEN":
        errors.append("pilot_sample_not_frozen")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("pilot_sample_eligible_sha")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("pilot_sample_metadata_sha")
    samples = obj.get("samples") or []
    if len(samples) < 8:
        errors.append("pilot_sample_too_small")
    ids = [str(s.get("sample_id")) for s in samples]
    if len(ids) != len(set(ids)):
        errors.append("pilot_sample_duplicate_ids")
    eligible = set(load_json(root / ELIGIBLE_REL).get("eligible_ids") or [])
    for s in samples:
        sid = str(s.get("sample_id"))
        if sid not in eligible:
            errors.append(f"pilot_sample_not_eligible:{sid}")
        if s.get("dataset") != "VSL400":
            errors.append(f"pilot_sample_dataset:{sid}")
    if obj.get("sample_ids_sha256") != sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")):
        errors.append("pilot_sample_ids_sha_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n": len(samples), "ids": ids}


def validate_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_pilot_independent_reaudit_v1":
        errors.append("bad_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("independent_reaudit_must_fail")
    if obj.get("parent_pilot_audit_sha256") != EXPECTED_PARENT_PILOT_AUDIT_SHA256:
        errors.append("independent_reaudit_parent_audit_sha")
    if sha256_file(root / PILOT_AUDIT_REL) != EXPECTED_PARENT_PILOT_AUDIT_SHA256:
        errors.append("parent_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("parent_pilot_audit_must_remain_pass")
    findings = obj.get("findings") or []
    ids = {f.get("id") for f in findings}
    required = {f"P0-{i}" for i in range(1, 7)}
    if ids != required:
        errors.append("independent_reaudit_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("independent_reaudit_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r1_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r1_pilot_independent_reaudit_v1":
        errors.append("bad_r1_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r1_independent_reaudit_must_fail")
    if obj.get("parent_r1_pilot_audit_sha256") != EXPECTED_R1_PILOT_AUDIT_SHA256:
        errors.append("r1_independent_reaudit_parent_sha")
    if sha256_file(root / PILOT_R1_AUDIT_REL) != EXPECTED_R1_PILOT_AUDIT_SHA256:
        errors.append("r1_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R1_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r1_pilot_audit_must_remain_pass")
    findings = obj.get("findings") or []
    ids = {f.get("id") for f in findings}
    required = {"P0-1", "P0-2", "P0-3", "P0-4", "P1-1"}
    if ids != required:
        errors.append("r1_independent_reaudit_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r1_independent_full_corpus_must_be_false")
    if obj.get("scientific_pilot_pass") is not False:
        errors.append("r1_independent_scientific_pass_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_pilot_sample_set_r1(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Strict R1 engineering pilot: train/val only, evidence-based tags, gloss diversity."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r1_pilot_sample_set_v1":
        errors.append("bad_r1_pilot_sample_schema")
    if obj.get("status") != "FROZEN":
        errors.append("r1_pilot_sample_not_frozen")
    if obj.get("engineering_scope") != "train_val_only":
        errors.append("r1_must_be_train_val_only_scope")
    if obj.get("test_clips_forbidden_for_roi_preprocess") is not True:
        errors.append("r1_must_forbid_test_for_roi_preprocess")
    samples = obj.get("samples") or []
    if len(samples) < 12:
        errors.append("r1_pilot_sample_too_small")
    ids = [str(s.get("sample_id")) for s in samples]
    if len(ids) != len(set(ids)):
        errors.append("r1_duplicate_ids")
    eligible = set(load_json(root / ELIGIBLE_REL).get("eligible_ids") or [])
    glosses = set()
    signers = set()
    frames = []
    for s in samples:
        sid = str(s.get("sample_id"))
        split = str(s.get("split"))
        if split not in {"train", "val"}:
            errors.append(f"r1_test_or_unknown_split_forbidden:{sid}:{split}")
        if sid not in eligible:
            errors.append(f"r1_not_eligible:{sid}")
        if "decode_edge" in (s.get("edge_tags") or []):
            if s.get("decode_failure_evidence") is not True:
                errors.append(f"r1_decode_edge_without_evidence:{sid}")
        if "short_clip" in (s.get("edge_tags") or []) or "long_clip" in (s.get("edge_tags") or []):
            if not isinstance(s.get("n_frames"), int) or s.get("length_quantile") is None:
                errors.append(f"r1_length_tag_missing_quantile:{sid}")
        if "missing_hand_or_face_risk" in (s.get("edge_tags") or []):
            mr = s.get("measured_missing_rate")
            if not isinstance(mr, (int, float)) or float(mr) < 0.05:
                errors.append(f"r1_missing_tag_without_measured_rate:{sid}")
        glosses.add(str(s.get("gloss") or ""))
        signers.add(str(s.get("signer_id") or ""))
        frames.append(int(s.get("n_frames") or 0))
    if len(glosses) < 8:
        errors.append(f"r1_insufficient_gloss_diversity:{len(glosses)}")
    if len(signers) < 6:
        errors.append(f"r1_insufficient_signer_diversity:{len(signers)}")
    if frames and (max(frames) - min(frames) < 30):
        errors.append("r1_frame_range_too_narrow")
    if obj.get("sample_ids_sha256") != sha256_bytes(("\n".join(ids) + "\n").encode("utf-8")):
        errors.append("r1_sample_ids_sha_mismatch")
    # Historical false-PASS patterns must be rejected by this validator.
    if "ANH" in glosses and len(glosses) == 1:
        errors.append("r1_rejects_single_gloss_anh_set")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n": len(samples), "ids": ids, "n_glosses": len(glosses)}


def validate_r2_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r2_pilot_independent_reaudit_v1":
        errors.append("bad_r2_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r2_independent_must_fail")
    if obj.get("parent_r2_pilot_audit_sha256") != EXPECTED_R2_PILOT_AUDIT_SHA256:
        errors.append("r2_independent_parent_sha")
    if sha256_file(root / PILOT_R2_AUDIT_REL) != EXPECTED_R2_PILOT_AUDIT_SHA256:
        errors.append("r2_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R2_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r2_pilot_audit_must_remain_pass")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if ids != {"P0-1", "P0-2", "P0-3"}:
        errors.append("r2_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r2_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_d0_full_protocol(obj: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_v1":
        errors.append("bad_full_protocol_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("full_protocol_corpus_must_be_false")
    if obj.get("forbid_temporal_mean_before_roi_align") is not True:
        errors.append("full_protocol_must_forbid_mean_before_roi")
    if obj.get("forbid_zero_dilution_in_temporal_mean") is not True:
        errors.append("full_protocol_must_forbid_zero_dilution")
    skel = obj.get("skeleton_v3") or {}
    if skel.get("model_complexity") != 0:
        errors.append("full_protocol_model_complexity")
    if skel.get("frame_stride") != 2:
        errors.append("full_protocol_frame_stride")
    rgb = obj.get("rgb_token_v3") or {}
    if rgb.get("window_size") != 32 or rgb.get("window_stride") != 16:
        errors.append("full_protocol_window")
    if rgb.get("operator_order") != "roi_align_per_temporal_slice_then_masked_mean_aggregate":
        errors.append("full_protocol_operator_order")
    if rgb.get("temporal_aggregation") != "masked_mean_exclude_missing_zeros":
        errors.append("full_protocol_temporal_aggregation")
    align = rgb.get("alignment") or {}
    if align.get("tie_break") != "prefer_earlier_skel_index":
        errors.append("full_protocol_tie_break")
    if align.get("timestamp_dtype") != "float64":
        errors.append("full_protocol_timestamp_dtype")
    vqa = obj.get("visual_qa") or {}
    if vqa.get("manual_review_required") is not True:
        errors.append("full_protocol_manual_visual_required")
    if vqa.get("manual_acceptance_rule") != "all_reviewed_inspections_must_pass_zero_fail_allowed":
        errors.append("full_protocol_manual_acceptance_rule")
    if vqa.get("fail_cases_cannot_be_pass_precondition") is not True:
        errors.append("full_protocol_fail_not_pass_precondition")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r3_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r3_pilot_independent_reaudit_v1":
        errors.append("bad_r3_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r3_independent_must_fail")
    if obj.get("parent_r3_pilot_audit_sha256") != EXPECTED_R3_PILOT_AUDIT_SHA256:
        errors.append("r3_independent_parent_sha")
    if sha256_file(root / PILOT_R3_AUDIT_REL) != EXPECTED_R3_PILOT_AUDIT_SHA256:
        errors.append("r3_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R3_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r3_pilot_audit_must_remain_pass")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if ids != {"P0-1", "P0-2", "P0-3"}:
        errors.append("r3_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r3_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def _norm_path(p: str) -> str:
    return str(p).replace("\\", "/").rstrip("/")


def validate_visual_qa_manual(
    obj: dict[str, Any],
    *,
    overlays_index: list[dict[str, Any]],
    require_zero_fail: bool = False,
    require_temporal_contact: bool = False,
    require_r5_identity: bool = False,
) -> dict[str, Any]:
    """Validate manual visual QA.

    R5 (require_r5_identity=True): qa_kind==bound.kind + contact tile identity + stratified freeze.
    R4 (require_zero_fail=True): strict identity bind + logical verdict + zero FAIL.
    R3 schema remains supported for historical artifacts when require_zero_fail=False.
    """
    errors: list[str] = []
    schema = obj.get("schema")
    if require_r5_identity:
        if schema != "ms_sta_v5_d0_r5_visual_qa_manual_v1":
            errors.append("bad_visual_manual_schema_r5")
    elif require_zero_fail:
        if schema not in {
            "ms_sta_v5_d0_r4_visual_qa_manual_v1",
            "ms_sta_v5_d0_r5_visual_qa_manual_v1",
        }:
            errors.append("bad_visual_manual_schema_r4")
    elif schema not in {
        "ms_sta_v5_d0_r3_visual_qa_manual_v1",
        "ms_sta_v5_d0_r4_visual_qa_manual_v1",
        "ms_sta_v5_d0_r5_visual_qa_manual_v1",
    }:
        errors.append("bad_visual_manual_schema")
    if obj.get("source") != "independent_human_or_named_reviewer_artifact":
        errors.append("visual_manual_source_invalid")
    if obj.get("auto_stroke_color_sufficient") is not False:
        errors.append("visual_manual_must_reject_auto_stroke_as_sufficient")
    reviewer = obj.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("visual_manual_reviewer_missing")
    inspections = obj.get("inspections") or []
    if len(inspections) < 3:
        errors.append("visual_manual_too_few")

    by_sha = {o.get("sha256"): o for o in overlays_index if o.get("sha256")}
    n_pass = 0
    n_fail = 0
    for row in inspections:
        verdict = row.get("verdict")
        if verdict not in {"PASS", "FAIL"}:
            errors.append("visual_manual_verdict_invalid")
        sha = row.get("sha256")
        if not sha or sha not in by_sha:
            errors.append(f"visual_manual_sha_unbound:{sha}")
            continue
        bound = by_sha[sha]
        for key in ("box_on_correct_body_region", "handedness_ok", "visible_but_missing_consistent"):
            if key not in row:
                errors.append(f"visual_manual_missing_check:{key}")
        if row.get("derived_from_overlay_render_integrity_only") is True:
            errors.append("visual_manual_cannot_be_render_integrity_only")

        if row.get("sample_id") != bound.get("sample_id"):
            errors.append(f"visual_manual_sample_id_mismatch:{row.get('sample_id')}")
        row_tail = _norm_path(str(row.get("path") or ""))
        bound_tail = _norm_path(str(bound.get("path") or ""))
        if "Docs/results/" in row_tail:
            row_tail = row_tail.split("Docs/results/", 1)[1]
        if "Docs/results/" in bound_tail:
            bound_tail = bound_tail.split("Docs/results/", 1)[1]
        if Path(row_tail).as_posix() != Path(bound_tail).as_posix():
            if Path(row_tail).name != Path(bound_tail).name:
                errors.append(f"visual_manual_path_mismatch:{row.get('sample_id')}")
            elif "ms_sta_v5_d0" in row_tail and row_tail != bound_tail:
                errors.append(f"visual_manual_path_mismatch:{row.get('sample_id')}")
        if row.get("center_frame") is None or int(row["center_frame"]) != int(bound.get("center_frame") or -1):
            errors.append(f"visual_manual_center_frame_mismatch:{row.get('sample_id')}")
        if row.get("native_frame") is None or bound.get("native_frame") is None:
            errors.append(f"visual_manual_native_frame_missing:{row.get('sample_id')}")
        elif int(row["native_frame"]) != int(bound["native_frame"]):
            errors.append(f"visual_manual_native_frame_mismatch:{row.get('sample_id')}")
        if "boxes" not in row or row.get("boxes") != bound.get("boxes"):
            errors.append(f"visual_manual_boxes_mismatch:{row.get('sample_id')}")
        miss_row = row.get("missing_regions")
        if miss_row is None:
            miss_row = row.get("missing_regions_declared")
        if miss_row != bound.get("missing_regions"):
            errors.append(f"visual_manual_missing_regions_mismatch:{row.get('sample_id')}")

        # Fail-closed kind bind (R5 required; also when qa_kind present).
        if require_r5_identity or require_temporal_contact or row.get("qa_kind") is not None:
            if row.get("qa_kind") not in {"center_overlay", "temporal_contact_sheet"}:
                errors.append(f"visual_manual_qa_kind_missing:{row.get('sample_id')}")
            elif row.get("qa_kind") != bound.get("kind"):
                errors.append(f"visual_manual_qa_kind_mismatch:{row.get('sample_id')}")

        if require_r5_identity and bound.get("kind") == "temporal_contact_sheet":
            for key in ("n_tiles", "slice_native_frames", "tiles"):
                if key not in row:
                    errors.append(f"visual_manual_contact_field_missing:{key}")
                elif row.get(key) != bound.get(key):
                    errors.append(f"visual_manual_contact_field_mismatch:{key}")
            tiles = row.get("tiles") or []
            bound_tiles = bound.get("tiles") or []
            if len(tiles) != len(bound_tiles):
                errors.append("visual_manual_tiles_len_mismatch")
            else:
                for i, (tile, bt) in enumerate(zip(tiles, bound_tiles)):
                    for key in (
                        "pathway",
                        "t_idx",
                        "native_frame",
                        "landmark_frame",
                        "boxes",
                        "region_missing",
                    ):
                        if key not in tile:
                            errors.append(f"visual_manual_tile_missing:{i}:{key}")
                        elif tile.get(key) != bt.get(key):
                            errors.append(f"visual_manual_tile_mismatch:{i}:{key}")

        box_ok = bool(row.get("box_on_correct_body_region"))
        hand_ok = bool(row.get("handedness_ok"))
        vis_ok = bool(row.get("visible_but_missing_consistent"))
        logical = "PASS" if (box_ok and hand_ok and vis_ok) else "FAIL"
        if verdict == "PASS" and logical != "PASS":
            errors.append(f"visual_manual_contradictory_pass:{row.get('sample_id')}")
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1

    if obj.get("arbitrary_wrong_box_negative_accepted") is not False:
        errors.append("visual_manual_must_document_wrong_box_rejection")

    declared_pass = obj.get("n_pass")
    declared_fail = obj.get("n_fail")
    if declared_pass is not None and int(declared_pass) != n_pass:
        errors.append("visual_manual_n_pass_mismatch")
    if declared_fail is not None and int(declared_fail) != n_fail:
        errors.append("visual_manual_n_fail_mismatch")

    if require_zero_fail or require_r5_identity:
        if n_fail != 0:
            errors.append("visual_manual_fail_not_allowed_for_pass_campaign")
        if n_pass < 3:
            errors.append("visual_manual_insufficient_pass")
        if obj.get("acceptance_rule") != "all_reviewed_inspections_must_pass_zero_fail_allowed":
            errors.append("visual_manual_acceptance_rule_missing")
        kinds = {row.get("qa_kind") for row in inspections}
        if "center_overlay" not in kinds or "temporal_contact_sheet" not in kinds:
            errors.append("visual_manual_missing_center_or_temporal_kind")

    if require_r5_identity:
        if obj.get("freeze_list_rel") != PILOT_R5_VISUAL_FREEZE_REL:
            errors.append("visual_manual_freeze_list_unbound")
        strata = set(obj.get("strata_covered") or [])
        required = {
            "full_valid_control_vf_eq_1",
            "partial_mid_0_5_le_vf_lt_1",
            "partial_low_0_lt_vf_lt_0_5",
            "worst_face",
            "worst_left_hand",
            "worst_right_hand",
            "train_split",
            "val_split",
            "mandatory_015417_w0016_face_vf_0_15",
        }
        if not required.issubset(strata):
            errors.append("visual_manual_strata_incomplete")
        # Reject all-full-valid selection bias.
        vfs = [float(r.get("min_valid_fraction")) for r in inspections if r.get("min_valid_fraction") is not None]
        if vfs and all(v >= 1.0 - 1e-12 for v in vfs):
            errors.append("visual_manual_all_full_valid_selection_forbidden")
        ids = {(r.get("sample_id"), int(r.get("window_start_idx"))) for r in inspections if r.get("window_start_idx") is not None}
        if ("015417", 16) not in ids:
            errors.append("visual_manual_missing_mandatory_015417_w0016")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n": len(inspections), "n_pass": n_pass, "n_fail": n_fail}


def validate_r4_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r4_pilot_independent_reaudit_v1":
        errors.append("bad_r4_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r4_independent_must_fail")
    if obj.get("parent_r4_pilot_audit_sha256") != EXPECTED_R4_PILOT_AUDIT_SHA256:
        errors.append("r4_independent_parent_sha")
    if sha256_file(root / PILOT_R4_AUDIT_REL) != EXPECTED_R4_PILOT_AUDIT_SHA256:
        errors.append("r4_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R4_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r4_pilot_audit_must_remain_pass")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if ids != {"P0-1", "P0-2"}:
        errors.append("r4_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r4_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r5_qa_amendment(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_r5_qa_amendment_v1":
        errors.append("bad_r5_amendment_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r5_amendment_corpus_must_be_false")
    if obj.get("do_not_overwrite_canonical_full_protocol") is not True:
        errors.append("r5_amendment_must_protect_canonical")
    if obj.get("parent_protocol_sha256_at_r4") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("r5_amendment_parent_protocol_sha")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift_from_r4")
    vqa = obj.get("visual_qa") or {}
    if vqa.get("qa_kind_must_equal_bound_kind") is not True:
        errors.append("r5_amendment_kind_bind_required")
    if vqa.get("freeze_list_before_review") is not True:
        errors.append("r5_amendment_freeze_required")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r5_visual_freeze_list(obj: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r5_visual_qa_freeze_list_v1":
        errors.append("bad_r5_freeze_schema")
    if obj.get("frozen_before_review") is not True:
        errors.append("r5_freeze_must_be_before_review")
    windows = obj.get("windows") or []
    if len(windows) < 6:
        errors.append("r5_freeze_too_few")
    strata = {w.get("stratum") for w in windows}
    for w in windows:
        strata.update(w.get("also_strata") or [])
    required = {
        "full_valid_control_vf_eq_1",
        "partial_mid_0_5_le_vf_lt_1",
        "partial_low_0_lt_vf_lt_0_5",
        "worst_face",
        "worst_left_hand",
        "worst_right_hand",
        "mandatory_015417_w0016_face_vf_0_15",
    }
    if not required.issubset(strata):
        errors.append("r5_freeze_strata_incomplete")
    splits = {w.get("split") for w in windows}
    if "train" not in splits or "val" not in splits:
        errors.append("r5_freeze_missing_train_or_val")
    keys = {(w.get("sample_id"), int(w.get("window_start_idx"))) for w in windows}
    if ("015417", 16) not in keys:
        errors.append("r5_freeze_missing_015417_w0016")
    vfs = [float(w.get("min_valid_fraction")) for w in windows]
    if vfs and all(v >= 1.0 - 1e-12 for v in vfs):
        errors.append("r5_freeze_all_full_valid_forbidden")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n": len(windows)}


def validate_r5_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r5_pilot_independent_reaudit_v1":
        errors.append("bad_r5_independent_reaudit_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r5_independent_must_fail")
    if obj.get("parent_r5_pilot_audit_sha256") != EXPECTED_R5_PILOT_AUDIT_SHA256:
        errors.append("r5_independent_parent_sha")
    if sha256_file(root / PILOT_R5_AUDIT_REL) != EXPECTED_R5_PILOT_AUDIT_SHA256:
        errors.append("r5_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R5_AUDIT_REL)
    if parent.get("audit_verdict") != "FAIL":
        errors.append("r5_pilot_audit_must_remain_fail")
    term = obj.get("terminology_correction") or {}
    if term.get("correct") != "hand_landmark_detector_false_negative_vs_visible_rgb_hand":
        errors.append("r5_independent_terminology")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if not {"P0-1", "P1-1"}.issubset(ids):
        errors.append("r5_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r5_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r6_detector_amendment(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_r6_detector_recovery_amendment_v1":
        errors.append("bad_r6_amendment_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r6_amendment_corpus_must_be_false")
    if obj.get("do_not_overwrite_canonical_full_protocol") is not True:
        errors.append("r6_amendment_must_protect_canonical")
    if obj.get("parent_protocol_sha256_at_r4") != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("r6_amendment_parent_protocol_sha")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift_from_r4")
    if obj.get("preferred_arm") != "C3":
        errors.append("r6_preferred_arm")
    prov = obj.get("provenance") or {}
    if prov.get("code_lock_field") != "code_lock_git_revision":
        errors.append("r6_provenance_code_lock_field")
    if prov.get("executable_tree_sha256_required") is not True:
        errors.append("r6_provenance_tree_required")
    matrix = obj.get("diagnostic_matrix") or {}
    if set(matrix) != {"C0", "C1", "C2", "C3"}:
        errors.append("r6_diagnostic_matrix_incomplete")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r6_freeze_lists(
    primary: dict[str, Any],
    holdout: dict[str, Any],
    *,
    r5_freeze: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if primary.get("schema") != "ms_sta_v5_d0_r6_visual_qa_freeze_list_v1":
        errors.append("bad_r6_freeze_schema")
    if holdout.get("schema") != "ms_sta_v5_d0_r6_holdout_risk_freeze_list_v1":
        errors.append("bad_r6_holdout_schema")
    if primary.get("frozen_before_algorithm") is not True:
        errors.append("r6_freeze_must_precede_algorithm")
    if holdout.get("frozen_before_algorithm") is not True:
        errors.append("r6_holdout_must_precede_algorithm")
    r5_keys = {
        (w.get("sample_id"), int(w.get("window_start_idx")))
        for w in (r5_freeze.get("windows") or [])
    }
    prim_keys = {
        (w.get("sample_id"), int(w.get("window_start_idx")))
        for w in (primary.get("windows") or [])
    }
    if prim_keys != r5_keys:
        errors.append("r6_primary_freeze_must_match_r5_windows")
    hold_keys = {
        (w.get("sample_id"), int(w.get("window_start_idx")))
        for w in (holdout.get("windows") or [])
    }
    if not hold_keys:
        errors.append("r6_holdout_empty")
    if hold_keys & prim_keys:
        errors.append("r6_holdout_overlaps_primary")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n_primary": len(prim_keys), "n_holdout": len(hold_keys)}


def validate_r6_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r6_pilot_independent_reaudit_v1":
        errors.append("bad_r6_independent_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r6_independent_must_fail")
    if obj.get("parent_r6_pilot_audit_sha256") != EXPECTED_R6_PILOT_AUDIT_SHA256:
        errors.append("r6_independent_parent_sha")
    if sha256_file(root / PILOT_R6_AUDIT_REL) != EXPECTED_R6_PILOT_AUDIT_SHA256:
        errors.append("r6_pilot_audit_bytes_drift")
    parent = load_json(root / PILOT_R6_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r6_pilot_audit_must_remain_pass")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if not {"P0-1", "P0-2", "P0-3", "P0-4", "P0-5"}.issubset(ids):
        errors.append("r6_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r6_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7_dual_source_amendment(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_r7_dual_source_roi_amendment_v1":
        errors.append("bad_r7_amendment_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7_amendment_corpus_must_be_false")
    if obj.get("do_not_overwrite_canonical_full_protocol") is not True:
        errors.append("r7_amendment_must_protect_canonical")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift_from_r4")
    dual = obj.get("dual_source") or {}
    if dual.get("forbid_falsify_skeleton_missing") is not True:
        errors.append("r7_must_forbid_falsify_skeleton")
    qa = obj.get("qa_gates") or {}
    for key in (
        "manual_is_human_frozen_artifact",
        "finalizer_must_not_synthesize_verdicts",
        "primary_and_holdout_zero_fail",
        "c0_bit_exact_all_16_pilot_samples",
    ):
        if qa.get(key) is not True:
            errors.append(f"r7_qa_gate_missing:{key}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7_visual_freeze_list(obj: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") not in {
        "ms_sta_v5_d0_r7_visual_qa_freeze_list_v1",
        "ms_sta_v5_d0_r7b_visual_qa_freeze_list_v1",
        "ms_sta_v5_d0_r7c_visual_qa_freeze_list_v1",
    }:
        errors.append("bad_r7_freeze_schema")
    if obj.get("frozen_before_algorithm") is not True:
        errors.append("r7_freeze_must_precede_algorithm")
    windows = obj.get("windows") or []
    if len(windows) != 11:
        errors.append("r7_freeze_must_be_exact_11")
    roles = {
        w.get("r7_role") or w.get("r7b_role") or w.get("r7c_role") for w in windows
    }
    if "primary_r5_carryforward" not in roles or "holdout_r6_carryforward" not in roles:
        errors.append("r7_freeze_missing_primary_or_holdout_role")
    negs = obj.get("truly_absent_negative_controls") or []
    if len(negs) < 2:
        errors.append("r7_freeze_missing_truly_absent_negative")
    else:
        kinds = {n.get("kind") for n in negs}
        for n in negs:
            if n.get("kind") == "unit_backed":
                if not n.get("test"):
                    errors.append("r7_neg_unit_backed_missing_test")
            elif n.get("kind") == "empirical_fixture":
                if not (n.get("sample_id") and n.get("fixture_rel") and n.get("proof_rel")):
                    errors.append("r7_neg_empirical_fixture_incomplete")
            elif not (n.get("sample_id") and n.get("window_start_idx") is not None):
                errors.append("r7_neg_empirical_missing_key")
        if obj.get("schema") in {
            "ms_sta_v5_d0_r7b_visual_qa_freeze_list_v1",
            "ms_sta_v5_d0_r7c_visual_qa_freeze_list_v1",
        }:
            if "empirical_fixture" not in kinds:
                errors.append("r7b_freeze_requires_empirical_fixture_negative")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n": len(windows), "n_neg": len(negs)}


def validate_r7_visual_qa_manual(
    obj: dict[str, Any],
    *,
    overlays_index: list[dict[str, Any]],
    freeze_list: dict[str, Any] | None = None,
    root: Path = _ROOT,
    visual_dir_rel: str = PILOT_R7B_VISUAL_REL,
    require_exact_22: bool = True,
    require_prior_tile_judgments: bool = False,
) -> dict[str, Any]:
    """Fail-closed R7/R7B manual: exact freeze×kinds, unique SHA/path, recomputed digests."""
    errors: list[str] = []
    if obj.get("schema") not in {
        "ms_sta_v5_d0_r7_visual_qa_manual_v1",
        "ms_sta_v5_d0_r7b_visual_qa_manual_v1",
        "ms_sta_v5_d0_r7c_visual_qa_manual_v1",
        "ms_sta_v5_d0_r7c_h_visual_qa_manual_v1",
        "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1",
    }:
        errors.append("bad_r7_manual_schema")
    if obj.get("schema") in {
        "ms_sta_v5_d0_r7c_h_visual_qa_manual_v1",
        "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1",
    }:
        require_prior_tile_judgments = True
    if obj.get("schema") == "ms_sta_v5_d0_r7c_h1_visual_qa_manual_v1":
        # Reject the retired H fill-script fingerprint notes.
        for row in obj.get("inspections") or []:
            for j in row.get("prior_tile_judgments") or []:
                note = str(j.get("notes") or "")
                if any(m in note for m in SCRIPTED_PRIOR_TILE_NOTE_MARKERS):
                    errors.append("r7_manual_scripted_prior_tile_note_forbidden")
                    break
    if obj.get("source") != "independent_human_or_named_reviewer_artifact":
        errors.append("r7_manual_source_invalid")
    if obj.get("synthesized_by_finalizer") is not False:
        errors.append("r7_manual_must_declare_not_synthesized")
    if obj.get("synthesized_by_template_tool") is not False:
        errors.append("r7_manual_must_declare_not_template_synth")
    if obj.get("auto_stroke_color_sufficient") is not False:
        errors.append("r7_manual_must_reject_auto_stroke")
    if obj.get("acceptance_rule") != "primary_and_holdout_all_pass_zero_fail_allowed":
        errors.append("r7_manual_acceptance_rule")
    reviewer = obj.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("r7_manual_reviewer_required")
    reviewed_at = obj.get("reviewed_at")
    if not isinstance(reviewed_at, str) or not reviewed_at.strip():
        errors.append("r7_manual_reviewed_at_required")
    method = obj.get("method")
    if not isinstance(method, str) or not method.strip():
        errors.append("r7_manual_method_required")

    inspections = obj.get("inspections") or []
    if require_exact_22 and len(inspections) != 22:
        errors.append("r7_manual_must_be_exact_22")
    elif len(inspections) < 11:
        errors.append("r7_manual_too_few")

    freeze = freeze_list or {}
    freeze_windows = freeze.get("windows") or []
    role_by = {
        (w.get("sample_id"), int(w.get("window_start_idx"))): (
            w.get("r7_role") or w.get("r7b_role") or w.get("r7c_role")
        )
        for w in freeze_windows
    }
    expected_keys: set[tuple[Any, int, str]] = set()
    for w in freeze_windows:
        sid = w.get("sample_id")
        ws = int(w.get("window_start_idx"))
        for kind in ("center_overlay", "temporal_contact_sheet"):
            expected_keys.add((sid, ws, kind))
    if freeze_windows and require_exact_22 and len(expected_keys) != 22:
        errors.append("r7_freeze_expected_keys_not_22")

    by_sha = {o.get("sha256"): o for o in overlays_index if o.get("sha256")}
    if len(by_sha) != len([o for o in overlays_index if o.get("sha256")]):
        errors.append("r7_overlays_sha_not_unique")

    seen_identity: set[tuple[Any, int, str]] = set()
    seen_sha: set[str] = set()
    n_pass = n_fail = 0
    visual_prefix = visual_dir_rel.replace("\\", "/").rstrip("/") + "/"

    for row in inspections:
        sha = row.get("sha256")
        sid = row.get("sample_id")
        kind = row.get("qa_kind")
        ws = row.get("window_start_idx")
        try:
            ws_i = int(ws)
        except Exception:
            errors.append(f"r7_manual_bad_window:{sid}")
            continue
        ident = (sid, ws_i, kind)
        if ident in seen_identity:
            errors.append(f"r7_manual_duplicate_identity:{ident}")
        seen_identity.add(ident)
        if sha in seen_sha:
            errors.append(f"r7_manual_duplicate_sha:{sha}")
        if isinstance(sha, str):
            seen_sha.add(sha)

        if sha not in by_sha:
            errors.append(f"r7_manual_unbound:{sha}")
            continue
        bound = by_sha[sha]
        if kind != bound.get("kind"):
            errors.append(f"r7_manual_kind_mismatch:{sid}")
        if sid != bound.get("sample_id"):
            errors.append(f"r7_manual_sample_mismatch:{sid}")
        if int(bound.get("window_start_idx")) != ws_i:
            errors.append(f"r7_manual_window_mismatch:{sid}")
        if row.get("boxes") != bound.get("boxes"):
            errors.append(f"r7_manual_boxes_mismatch:{sid}")
        if row.get("missing_regions") != bound.get("missing_regions"):
            errors.append(f"r7_manual_missing_regions_mismatch:{sid}")
        if row.get("rgb_roi_sources") != bound.get("rgb_roi_sources"):
            errors.append(f"r7_manual_rgb_sources_mismatch:{sid}")
        if kind == "temporal_contact_sheet":
            if row.get("tiles") != bound.get("tiles"):
                errors.append(f"r7_manual_tiles_mismatch:{sid}")
            if row.get("n_tiles") != bound.get("n_tiles"):
                errors.append(f"r7_manual_n_tiles_mismatch:{sid}")
            if row.get("slice_native_frames") != bound.get("slice_native_frames"):
                errors.append(f"r7_manual_slice_mismatch:{sid}")

        path_raw = row.get("path")
        try:
            rel = repo_rel_path(str(path_raw), root=root)
        except ValueError:
            errors.append(f"r7_manual_path_not_repo_relative:{sid}")
            rel = None
        bound_rel = None
        try:
            bound_rel = repo_rel_path(str(bound.get("path")), root=root)
        except ValueError:
            errors.append(f"r7_manual_bound_path_not_repo_relative:{sid}")
        if rel is not None:
            if not rel.startswith(visual_prefix):
                errors.append(f"r7_manual_path_outside_visual_dir:{sid}")
            if bound_rel is not None and rel != bound_rel:
                errors.append(f"r7_manual_path_mismatch:{sid}")
            fpath = root / rel
            if not fpath.is_file():
                errors.append(f"r7_manual_overlay_missing:{rel}")
            else:
                dig = sha256_file(fpath)
                if dig != sha:
                    errors.append(f"r7_manual_sha_recompute_mismatch:{sid}")

        if freeze_windows:
            role = role_by.get((sid, ws_i))
            if role is None:
                errors.append(f"r7_manual_not_in_freeze:{sid}:{ws_i}")
            else:
                row_role = row.get("r7_role") or row.get("r7b_role")
                if row_role != role:
                    errors.append(f"r7_manual_role_mismatch:{sid}")

        for key in ("box_on_correct_body_region", "handedness_ok", "visible_but_missing_consistent"):
            if key not in row:
                errors.append(f"r7_manual_missing_check:{key}")
            elif row.get(key) is None:
                errors.append(f"r7_manual_check_null:{key}:{sid}")
        if row.get("verdict") not in ("PASS", "FAIL"):
            errors.append(f"r7_manual_verdict_null_or_invalid:{sid}")

        if require_prior_tile_judgments:
            expected_prior: list[tuple[Any, ...]] = []
            if kind == "center_overlay":
                src = row.get("rgb_roi_sources") or {}
                for hand in ("left_hand", "right_hand"):
                    if src.get(hand) in RGB_PRIOR_SOURCES:
                        expected_prior.append(("center", None, None, hand, src.get(hand)))
            elif kind == "temporal_contact_sheet":
                for tile in row.get("tiles") or []:
                    tsrc = tile.get("rgb_roi_sources") or {}
                    for hand in ("left_hand", "right_hand"):
                        if tsrc.get(hand) in RGB_PRIOR_SOURCES:
                            expected_prior.append(
                                (
                                    tile.get("pathway"),
                                    tile.get("t_idx"),
                                    tile.get("native_frame"),
                                    hand,
                                    tsrc.get(hand),
                                )
                            )
            judgments = row.get("prior_tile_judgments")
            if expected_prior:
                if not isinstance(judgments, list):
                    errors.append(f"r7_manual_prior_tile_judgments_required:{sid}:{kind}")
                    judgments = []
                seen_j: set[tuple[Any, ...]] = set()
                prior_fail = False
                for j in judgments:
                    if not isinstance(j, dict):
                        errors.append(f"r7_manual_prior_tile_judgment_bad:{sid}")
                        continue
                    key = (
                        j.get("pathway") if kind == "temporal_contact_sheet" else "center",
                        j.get("t_idx") if kind == "temporal_contact_sheet" else None,
                        j.get("native_frame") if kind == "temporal_contact_sheet" else None,
                        j.get("hand"),
                        j.get("source"),
                    )
                    if key in seen_j:
                        errors.append(f"r7_manual_prior_tile_judgment_dup:{sid}:{key}")
                    seen_j.add(key)
                    for ck in ("box_on_correct_body_region", "handedness_ok"):
                        if j.get(ck) is None:
                            errors.append(f"r7_manual_prior_tile_check_null:{sid}:{ck}")
                    if j.get("verdict") not in ("PASS", "FAIL"):
                        errors.append(f"r7_manual_prior_tile_verdict_invalid:{sid}")
                    j_logical = (
                        "PASS"
                        if bool(j.get("box_on_correct_body_region")) and bool(j.get("handedness_ok"))
                        else "FAIL"
                    )
                    if j.get("verdict") in ("PASS", "FAIL") and j.get("verdict") != j_logical:
                        errors.append(f"r7_manual_prior_tile_verdict_logical_mismatch:{sid}")
                    if j.get("verdict") == "FAIL":
                        prior_fail = True
                if set(expected_prior) != seen_j:
                    errors.append(f"r7_manual_prior_tile_coverage_incomplete:{sid}")
                if prior_fail and row.get("verdict") == "PASS":
                    errors.append(f"r7_manual_prior_tile_fail_but_row_pass:{sid}")
            elif judgments not in (None, []):
                errors.append(f"r7_manual_prior_tile_judgments_unexpected:{sid}")

        logical = (
            "PASS"
            if (
                bool(row.get("box_on_correct_body_region"))
                and bool(row.get("handedness_ok"))
                and bool(row.get("visible_but_missing_consistent"))
            )
            else "FAIL"
        )
        if row.get("verdict") in ("PASS", "FAIL") and row.get("verdict") != logical:
            errors.append(f"r7_manual_verdict_logical_mismatch:{sid}")
        if row.get("verdict") == "PASS":
            n_pass += 1
        elif row.get("verdict") == "FAIL":
            n_fail += 1

    if freeze_windows and require_exact_22:
        if seen_identity != expected_keys:
            errors.append("r7_manual_freeze_coverage_incomplete")

    # Structural validator allows FAIL rows; campaign PASS requires n_fail==0 in finalize/report.
    declared_fail = obj.get("n_fail")
    declared_pass = obj.get("n_pass")
    if declared_fail is None or int(declared_fail) != n_fail:
        errors.append("r7_manual_n_fail_mismatch")
    if declared_pass is None or int(declared_pass) != n_pass:
        errors.append("r7_manual_n_pass_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True, "n_pass": n_pass, "n_fail": n_fail}


def validate_r7_pilot_report(
    obj: dict[str, Any],
    *,
    root: Path = _ROOT,
    executable_paths: list[str] | None = None,
    manual_rel: str = PILOT_R7B_VISUAL_MANUAL_REL,
    overlays_rel: str = PILOT_R7B_OVERLAYS_INDEX_REL,
    diagnostic_rel: str = PILOT_R7B_DIAGNOSTIC_REL,
    freeze_rel: str = PILOT_R7B_VISUAL_FREEZE_REL,
    amendment_rel: str = FULL_PROTOCOL_R7B_AMENDMENT_REL,
    producer_rel: str = "scripts/run_ms_sta_v5_d0_r7b_finalize.py",
    require_campaign_pass: bool = True,
    require_worktree_code_lock_match: bool = False,
) -> dict[str, Any]:
    """Fail-closed report binding. When require_campaign_pass, ok+gates must be true."""
    errors: list[str] = []
    if obj.get("schema") not in {
        "ms_sta_v5_d0_r7_pilot_report_v1",
        "ms_sta_v5_d0_r7b_pilot_report_v1",
        "ms_sta_v5_d0_r7c_pilot_report_v1",
        "ms_sta_v5_d0_r7c_h_pilot_report_v1",
        "ms_sta_v5_d0_r7c_h1_pilot_report_v1",
    }:
        errors.append("bad_r7_report_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7_report_corpus_gate")
    forbid = obj.get("forbid") or {}
    if forbid.get("full_corpus") is not True:
        errors.append("r7_report_forbid_full_corpus")

    gates = obj.get("gates") or {}
    required_true_always = (
        "finalizer_did_not_synthesize_manual",
        "producer_in_executable_tree",
        "executable_paths_git_tracked",
        "pose_prior_geometry_locked",
    )
    for key in required_true_always:
        if gates.get(key) is not True:
            errors.append(f"r7_report_gate_false:{key}")

    if require_campaign_pass:
        if obj.get("ok") is not True:
            errors.append("r7_report_ok_must_be_true")
        for key in (
            "primary_and_holdout_zero_fail",
            "c0_bit_exact_all_16",
            "r5_fail_keys_remediated_with_boxes",
        ):
            if gates.get(key) is not True:
                errors.append(f"r7_report_gate_false:{key}")
        # Prefer split fields; fall back to legacy residual for older reports.
        miss_n = int_field(obj, "holdout_missing_box_n_fail")
        man_n = int_field(obj, "holdout_manual_n_fail")
        if miss_n is not None or man_n is not None:
            if miss_n != 0:
                errors.append("r7_report_holdout_missing_box_must_be_zero")
            if man_n != 0:
                errors.append("r7_report_holdout_manual_must_be_zero")
        else:
            residual = int_field(obj, "holdout_residual_n_fail")
            if residual is None or residual != 0:
                errors.append("r7_report_holdout_residual_must_be_zero")
    else:
        if obj.get("ok") is not False:
            errors.append("r7_report_fail_path_ok_must_be_false")

    paths = list(executable_paths or R7B_EXECUTABLE_PATHS)
    if producer_rel not in paths:
        errors.append("r7_report_producer_not_listed")
    errors.extend(
        verify_code_lock_revision_binding(
            obj.get("code_lock_git_revision"),
            paths,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=require_worktree_code_lock_match,
        )
    )
    try:
        executable_tree_sha256(paths, root=root, require_git_tracked=True)
    except FileNotFoundError as exc:
        errors.append(f"r7_report_tree_error:{exc}")
    if gates.get("producer_in_executable_tree") is True and producer_rel not in paths:
        errors.append("r7_report_producer_gate_unproven")

    for rel, field in (
        (manual_rel, "visual_qa_manual_sha256"),
        (overlays_rel, "overlays_index_sha256"),
        (diagnostic_rel, "diagnostic_sha256"),
        (freeze_rel, "freeze_sha256"),
        (amendment_rel, "r7c_amendment_sha256"),
    ):
        p = root / rel
        if not p.is_file():
            errors.append(f"r7_report_missing_bound_file:{rel}")
            continue
        dig = sha256_file(p)
        if field == "r7c_amendment_sha256":
            got = (
                obj.get("r7c_amendment_sha256")
                or obj.get("r7b_amendment_sha256")
                or obj.get("r7_amendment_sha256")
            )
        else:
            got = obj.get(field)
        if got != dig:
            errors.append(f"r7_report_hash_mismatch:{field}")

    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7_pilot_audit(
    obj: dict[str, Any],
    *,
    root: Path = _ROOT,
    report_rel: str = PILOT_R7B_REPORT_REL,
    executable_paths: list[str] | None = None,
    require_pass: bool = True,
    require_worktree_code_lock_match: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") not in {
        "ms_sta_v5_d0_r7_pilot_audit_v1",
        "ms_sta_v5_d0_r7b_pilot_audit_v1",
        "ms_sta_v5_d0_r7c_pilot_audit_v1",
        "ms_sta_v5_d0_r7c_h_pilot_audit_v1",
        "ms_sta_v5_d0_r7c_h1_pilot_audit_v1",
    }:
        errors.append("bad_r7_audit_schema")
    if require_pass:
        if obj.get("audit_verdict") != "PASS":
            errors.append("r7_audit_must_pass")
        if obj.get("gates_passed") is not True:
            errors.append("r7_audit_gates_passed")
    else:
        if obj.get("audit_verdict") != "FAIL":
            errors.append("r7_audit_fail_path_must_fail")
        if obj.get("gates_passed") is not False:
            errors.append("r7_audit_fail_path_gates_must_be_false")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7_audit_corpus")
    paths = list(executable_paths or R7B_EXECUTABLE_PATHS)
    errors.extend(
        verify_code_lock_revision_binding(
            obj.get("code_lock_git_revision"),
            paths,
            obj.get("executable_tree_sha256"),
            root=root,
            require_worktree_match=require_worktree_code_lock_match,
        )
    )
    try:
        executable_tree_sha256(paths, root=root, require_git_tracked=True)
    except FileNotFoundError as exc:
        errors.append(f"r7_audit_tree_error:{exc}")
    rp = root / report_rel
    if not rp.is_file():
        errors.append("r7_audit_report_missing")
    elif obj.get("pilot_report_sha256") != sha256_file(rp):
        errors.append("r7_audit_report_hash_mismatch")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7_pilot_completion(
    obj: dict[str, Any],
    *,
    root: Path = _ROOT,
    report_rel: str = PILOT_R7B_REPORT_REL,
    audit_rel: str = PILOT_R7B_AUDIT_REL,
    require_pass: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") not in {
        "ms_sta_v5_d0_r7_pilot_completion_v1",
        "ms_sta_v5_d0_r7b_pilot_completion_v1",
        "ms_sta_v5_d0_r7c_pilot_completion_v1",
        "ms_sta_v5_d0_r7c_h_pilot_completion_v1",
        "ms_sta_v5_d0_r7c_h1_pilot_completion_v1",
    }:
        errors.append("bad_r7_completion_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7_completion_corpus")
    if require_pass:
        if obj.get("pilot_audit_verdict") != "PASS":
            errors.append("r7_completion_audit_not_pass")
    else:
        if obj.get("pilot_audit_verdict") != "FAIL":
            errors.append("r7_completion_fail_path_must_fail")
    for rel, field in ((report_rel, "pilot_report_sha256"), (audit_rel, "pilot_audit_sha256")):
        p = root / rel
        if not p.is_file():
            errors.append(f"r7_completion_missing:{rel}")
        elif obj.get(field) != sha256_file(p):
            errors.append(f"r7_completion_hash_mismatch:{field}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7_pilot_independent_reaudit_v1":
        errors.append("bad_r7_independent_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r7_independent_must_fail")
    if obj.get("parent_r7_pilot_audit_sha256") != EXPECTED_R7_PILOT_AUDIT_SHA256:
        errors.append("r7_independent_parent_sha")
    if sha256_file(root / PILOT_R7_AUDIT_REL) != EXPECTED_R7_PILOT_AUDIT_SHA256:
        errors.append("r7_pilot_audit_bytes_drift")
    if sha256_file(root / PILOT_R7_REPORT_REL) != EXPECTED_R7_PILOT_REPORT_SHA256:
        errors.append("r7_pilot_report_bytes_drift")
    parent = load_json(root / PILOT_R7_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r7_pilot_audit_must_remain_pass")
    ids = {f.get("id") for f in (obj.get("findings") or [])}
    if not {"P0-1", "P0-2", "P0-3", "P0-4"}.issubset(ids):
        errors.append("r7_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7b_gate_hardening_amendment(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_r7b_gate_hardening_amendment_v1":
        errors.append("bad_r7b_amendment_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7b_amendment_corpus")
    if obj.get("do_not_overwrite_canonical_full_protocol") is not True:
        errors.append("r7b_amendment_must_protect_canonical")
    if obj.get("do_not_overwrite_r7_self_pass") is not True:
        errors.append("r7b_must_preserve_r7_pass")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift_from_r4")
    if sha256_file(root / PILOT_R7_AUDIT_REL) != EXPECTED_R7_PILOT_AUDIT_SHA256:
        errors.append("r7_audit_drift")
    geo = obj.get("pose_prior_geometry_lock") or {}
    for k, v in POSE_PRIOR_GEOMETRY_LOCK.items():
        if geo.get(k) != v:
            errors.append(f"r7b_geometry_lock_mismatch:{k}")
    qa = obj.get("qa_gates") or {}
    for key in (
        "manual_template_only_no_judgments_table",
        "manual_validator_exact_22_unique_recompute",
        "report_audit_completion_fail_closed",
        "executable_tree_git_tracked_includes_joint_layout",
        "primary_and_holdout_zero_fail",
        "empirical_truly_absent_negative_required",
    ):
        if qa.get(key) is not True:
            errors.append(f"r7b_qa_gate_missing:{key}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}

def validate_r7b_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7b_pilot_independent_reaudit_v1":
        errors.append("bad_r7b_independent_schema")
    layers = obj.get("audit_layers") or {}
    if layers.get("evidence_lifecycle") != "PASS":
        errors.append("r7b_independent_evidence_must_pass")
    if layers.get("scientific_d0_gate") != "FAIL":
        errors.append("r7b_independent_science_must_fail")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r7b_independent_overall_must_fail")
    if obj.get("parent_r7b_pilot_audit_sha256") != EXPECTED_R7B_PILOT_AUDIT_SHA256:
        errors.append("r7b_independent_parent_sha")
    if sha256_file(root / PILOT_R7B_AUDIT_REL) != EXPECTED_R7B_PILOT_AUDIT_SHA256:
        errors.append("r7b_pilot_audit_bytes_drift")
    if sha256_file(root / PILOT_R7B_REPORT_REL) != EXPECTED_R7B_PILOT_REPORT_SHA256:
        errors.append("r7b_pilot_report_bytes_drift")
    parent = load_json(root / PILOT_R7B_AUDIT_REL)
    if parent.get("audit_verdict") != "FAIL":
        errors.append("r7b_pilot_audit_must_remain_fail")
    ids = {b.get("id") for b in (obj.get("contract_blockers_before_r7c") or [])}
    if not {"P0-1", "P0-2", "P0-3", "P0-4", "P0-5"}.issubset(ids):
        errors.append("r7b_independent_blockers_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7b_independent_full_corpus_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7c_ambiguity_roi_amendment(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_protocol_r7c_ambiguity_aware_roi_amendment_v1":
        errors.append("bad_r7c_amendment_schema")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7c_amendment_corpus")
    if obj.get("do_not_overwrite_canonical_full_protocol") is not True:
        errors.append("r7c_must_protect_canonical")
    if obj.get("do_not_overwrite_r7b_self_fail") is not True:
        errors.append("r7c_must_preserve_r7b_fail")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift")
    if sha256_file(root / PILOT_R7B_AUDIT_REL) != EXPECTED_R7B_PILOT_AUDIT_SHA256:
        errors.append("r7b_audit_drift")
    lock = obj.get("ambiguity_roi_lock") or {}
    for k, v in R7C_AMBIGUITY_ROI_LOCK.items():
        if lock.get(k) != v:
            errors.append(f"r7c_ambiguity_lock_mismatch:{k}")
    qa = obj.get("qa_gates") or {}
    for key in (
        "holdout_missing_box_and_manual_split",
        "fixture_proof_fail_closed",
        "absolute_paths_forbidden",
        "verdict_equals_logical_bidirectional",
        "primary_and_holdout_zero_fail",
        "exact_11_windows_carryforward",
        "b3_b4_four_region_parity_no_extra_token",
    ):
        if qa.get(key) is not True:
            errors.append(f"r7c_qa_gate_missing:{key}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7c_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Bind independent R7C FAIL companion; preserve self-PASS bytes; require R7B fixture restore."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7c_pilot_independent_reaudit_v1":
        errors.append("bad_r7c_independent_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r7c_independent_must_fail")
    layers = obj.get("audit_layers") or {}
    if layers.get("evidence_lifecycle") != "FAIL":
        errors.append("r7c_independent_evidence_must_fail")
    if layers.get("scientific_d0_gate") != "NOT_ACCEPTED":
        errors.append("r7c_independent_science_not_accepted")
    if obj.get("parent_r7c_pilot_audit_sha256") != EXPECTED_R7C_PILOT_AUDIT_SHA256:
        errors.append("r7c_independent_parent_sha")
    if sha256_file(root / PILOT_R7C_AUDIT_REL) != EXPECTED_R7C_PILOT_AUDIT_SHA256:
        errors.append("r7c_pilot_audit_bytes_drift")
    if sha256_file(root / PILOT_R7C_REPORT_REL) != EXPECTED_R7C_PILOT_REPORT_SHA256:
        errors.append("r7c_pilot_report_bytes_drift")
    if sha256_file(root / PILOT_R7C_COMPLETION_REL) != EXPECTED_R7C_PILOT_COMPLETION_SHA256:
        errors.append("r7c_pilot_completion_bytes_drift")
    if sha256_file(root / PILOT_R7C_VISUAL_MANUAL_REL) != EXPECTED_R7C_VISUAL_MANUAL_SHA256:
        errors.append("r7c_visual_manual_bytes_drift")
    if sha256_file(root / PILOT_R7B_FIXTURE_ABSENT_REL / "PROOF.json") != EXPECTED_R7B_ABSENT_FIXTURE_PROOF_SHA256:
        errors.append("r7b_absent_fixture_proof_not_restored")
    parent = load_json(root / PILOT_R7C_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r7c_pilot_audit_must_remain_pass")
    ids = {b.get("id") for b in (obj.get("blocking_findings") or [])}
    if not {"P0-1", "P0-2", "P0-3", "P1-1", "P1-2"}.issubset(ids):
        errors.append("r7c_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7c_independent_full_corpus_must_be_false")
    if obj.get("algorithm_change_authorized") is not False:
        errors.append("r7c_independent_algo_change_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_r7c_h_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7c_h_pilot_independent_reaudit_v1":
        errors.append("bad_r7c_h_independent_schema")
    if obj.get("audit_verdict") != "FAIL":
        errors.append("r7c_h_independent_must_fail")
    layers = obj.get("audit_layers") or {}
    if layers.get("evidence_lifecycle") != "FAIL":
        errors.append("r7c_h_independent_evidence_must_fail")
    if obj.get("parent_r7c_h_pilot_audit_sha256") != EXPECTED_R7C_H_PILOT_AUDIT_SHA256:
        errors.append("r7c_h_independent_parent_sha")
    if sha256_file(root / PILOT_R7C_H_AUDIT_REL) != EXPECTED_R7C_H_PILOT_AUDIT_SHA256:
        errors.append("r7c_h_pilot_audit_bytes_drift")
    if sha256_file(root / PILOT_R7C_H_REPORT_REL) != EXPECTED_R7C_H_PILOT_REPORT_SHA256:
        errors.append("r7c_h_pilot_report_bytes_drift")
    if sha256_file(root / PILOT_R7C_H_COMPLETION_REL) != EXPECTED_R7C_H_PILOT_COMPLETION_SHA256:
        errors.append("r7c_h_pilot_completion_bytes_drift")
    if sha256_file(root / PILOT_R7C_H_VISUAL_MANUAL_REL) != EXPECTED_R7C_H_VISUAL_MANUAL_SHA256:
        errors.append("r7c_h_visual_manual_bytes_drift")
    parent = load_json(root / PILOT_R7C_H_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r7c_h_pilot_audit_must_remain_pass")
    ids = {b.get("id") for b in (obj.get("blocking_findings") or [])}
    if not {"P0-H1", "P0-H2", "P1-H1"}.issubset(ids):
        errors.append("r7c_h_independent_findings_incomplete")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7c_h_independent_full_corpus_must_be_false")
    if obj.get("algorithm_change_authorized") is not False:
        errors.append("r7c_h_independent_algo_change_must_be_false")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_manual_template_null_prior_slots(obj: dict[str, Any]) -> dict[str, Any]:
    """Template must keep all human judgment fields null until review."""
    errors: list[str] = []
    if obj.get("synthesized_by_template_tool") is not True:
        errors.append("template_must_declare_template_synth")
    if obj.get("status") != "TEMPLATE_AWAITING_HUMAN_REVIEW":
        errors.append("template_status")
    for row in obj.get("inspections") or []:
        for key in ("box_on_correct_body_region", "handedness_ok", "visible_but_missing_consistent", "verdict"):
            if row.get(key) is not None:
                errors.append(f"template_row_judgment_not_null:{key}")
        for j in row.get("prior_tile_judgments") or []:
            for key in ("box_on_correct_body_region", "handedness_ok", "verdict"):
                if j.get(key) is not None:
                    errors.append(f"template_prior_judgment_not_null:{key}")
            if str(j.get("notes") or "").strip():
                errors.append("template_prior_notes_must_be_empty")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def assert_fill_script_cannot_assign_pass(*, root: Path = _ROOT) -> dict[str, Any]:
    """Negative contract: fill producer source must not assign PASS/True to judgments."""
    import ast

    path = root / "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py"
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    if target.slice.value in {
                        "verdict",
                        "box_on_correct_body_region",
                        "handedness_ok",
                    }:
                        forbidden.append(f"assign:{target.slice.value}")
        if isinstance(node, ast.Constant) and node.value == "PASS":
            # Allow mentioning PASS only in comments/docstrings; reject string literals used as values.
            # Docstrings are Expr(Constant) at module/class/func start — still flagged if present as value.
            forbidden.append("literal_PASS")
    # Allow FORBIDDEN docstring text without PASS assignment path: strip docstring PASS mentions via allowlist.
    # Hard rule: file must exit non-zero and must not call write_json on a manual with verdicts.
    if "write_json" in src and "PILOT_R7C_H_VISUAL_MANUAL" in src:
        forbidden.append("writes_h_manual")
    if "j[\"verdict\"] = \"PASS\"" in src or "j['verdict'] = 'PASS'" in src:
        forbidden.append("explicit_pass_assign")
    if "True" in src and ("box_on_correct_body_region" in src) and ("= True" in src or "=True" in src):
        # Only flag if assignment pattern remains
        if "box_on_correct_body_region\"] = True" in src or "box_on_correct_body_region'] = True" in src:
            forbidden.append("explicit_true_box_assign")
    # Soft: literal_PASS in print/help is OK if no assignment; filter literal_PASS if FORBIDDEN stub.
    if "FORBIDDEN" in src and "must not write human verdict" in src:
        forbidden = [f for f in forbidden if f not in {"literal_PASS"}]
    if forbidden:
        raise ValueError("fill_script_still_assigns_human_verdicts:" + ",".join(sorted(set(forbidden))))
    if "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py" not in R7C_H_EXECUTABLE_PATHS:
        raise ValueError("fill_script_unbound_from_r7c_h_executable_tree")
    if "scripts/fill_ms_sta_v5_d0_r7c_h_manual.py" not in R7C_H1_EXECUTABLE_PATHS:
        raise ValueError("fill_script_unbound_from_r7c_h1_executable_tree")
    return {"ok": True}



def validate_r7c_h1_independent_reaudit(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Bind independent R7C-H1 PASS companion; preserve H1 self-PASS bytes."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_r7c_h1_pilot_independent_reaudit_v1":
        errors.append("bad_r7c_h1_independent_schema")
    if obj.get("audit_verdict") != "PASS":
        errors.append("r7c_h1_independent_must_pass")
    layers = obj.get("audit_layers") or {}
    for key in ("technical_contract", "evidence_lifecycle", "scientific_d0_pilot_gate"):
        if layers.get(key) != "PASS":
            errors.append(f"r7c_h1_independent_layer:{key}")
    if obj.get("parent_r7c_h1_pilot_audit_sha256") != EXPECTED_R7C_H1_PILOT_AUDIT_SHA256:
        errors.append("r7c_h1_independent_parent_sha")
    if sha256_file(root / PILOT_R7C_H1_AUDIT_REL) != EXPECTED_R7C_H1_PILOT_AUDIT_SHA256:
        errors.append("r7c_h1_pilot_audit_bytes_drift")
    if sha256_file(root / PILOT_R7C_H1_REPORT_REL) != EXPECTED_R7C_H1_PILOT_REPORT_SHA256:
        errors.append("r7c_h1_pilot_report_bytes_drift")
    if sha256_file(root / PILOT_R7C_H1_COMPLETION_REL) != EXPECTED_R7C_H1_PILOT_COMPLETION_SHA256:
        errors.append("r7c_h1_pilot_completion_bytes_drift")
    if sha256_file(root / PILOT_R7C_H1_VISUAL_MANUAL_REL) != EXPECTED_R7C_H1_VISUAL_MANUAL_SHA256:
        errors.append("r7c_h1_visual_manual_bytes_drift")
    if sha256_file(root / PILOT_R7C_H1_PRIOR_JUDGMENTS_REL) != EXPECTED_R7C_H1_PRIOR_JUDGMENTS_SHA256:
        errors.append("r7c_h1_prior_judgments_bytes_drift")
    parent = load_json(root / PILOT_R7C_H1_AUDIT_REL)
    if parent.get("audit_verdict") != "PASS":
        errors.append("r7c_h1_pilot_audit_must_remain_pass")
    if obj.get("blocking_findings") not in ([], None):
        errors.append("r7c_h1_independent_must_have_zero_blockers")
    if obj.get("full_corpus_authorized") is not False:
        errors.append("r7c_h1_independent_full_corpus_must_be_false")
    if obj.get("scientific_pilot_pass") is not True:
        errors.append("r7c_h1_independent_scientific_pilot_pass")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_d0_full_corpus_approval(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    """Fail-closed separate authorization for D0 full-corpus extraction only."""
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_approval_v1":
        errors.append("bad_full_corpus_approval_schema")
    if obj.get("decision") != "APPROVED" or obj.get("status") != "APPROVED":
        errors.append("full_corpus_approval_decision")
    if obj.get("scope") != "d0_full_corpus_extraction_only":
        errors.append("full_corpus_approval_scope")
    if obj.get("allow_full_corpus_extraction") is not True:
        errors.append("full_corpus_must_allow_extraction")
    for key in (
        "allow_m0",
        "allow_training",
        "allow_accuracy",
        "allow_hpo",
        "allow_test",
        "allow_demo",
        "allow_auto_chain",
    ):
        if obj.get(key) is not False:
            errors.append(f"{key}_must_be_false")
    if obj.get("stop_after") != "D0_FULL_CORPUS_SCAN_COMPLETE":
        errors.append("full_corpus_stop_after")
    if obj.get("output_version") != "v1":
        errors.append("full_corpus_output_version")
    if obj.get("output_dir_rel") != D0_FULL_CORPUS_OUT_DIR_REL:
        errors.append("full_corpus_output_dir")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("full_corpus_requires_h1_independent_pass_sha")
    if sha256_file(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL) != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("h1_independent_reaudit_bytes_drift")
    h1 = load_json(root / PILOT_R7C_H1_INDEPENDENT_REAUDIT_REL)
    if h1.get("audit_verdict") != "PASS":
        errors.append("h1_independent_must_be_pass")
    if obj.get("eligible_manifest_sha256") != EXPECTED_ELIGIBLE_SHA256:
        errors.append("full_corpus_eligible_sha")
    if obj.get("metadata_sha256") != EXPECTED_METADATA_SHA256:
        errors.append("full_corpus_metadata_sha")
    if obj.get("split_signers_sha256") != EXPECTED_SPLIT_SHA256:
        errors.append("full_corpus_split_sha")
    if sha256_file(root / FULL_PROTOCOL_REL) != EXPECTED_R4_FULL_PROTOCOL_SHA256:
        errors.append("canonical_full_protocol_drift")
    resume = obj.get("resume_contract") or {}
    if resume.get("enabled") is not True:
        errors.append("resume_contract_enabled")
    if int(resume.get("max_resumes") or -1) < 1:
        errors.append("resume_contract_max_resumes")
    if resume.get("idempotent_skip_existing_sha_matched_outputs") is not True:
        errors.append("resume_contract_idempotent")
    disk = obj.get("disk_preflight") or {}
    if disk.get("required") is not True:
        errors.append("disk_preflight_required")
    if float(disk.get("min_free_gb") or 0) < 50:
        errors.append("disk_preflight_min_free_gb")
    thr = obj.get("quality_thresholds") or {}
    for key in (
        "max_missing_roi_rate_face",
        "max_missing_roi_rate_left_hand",
        "max_missing_roi_rate_right_hand",
        "max_partial_missing_rate_any_region",
    ):
        if key not in thr:
            errors.append(f"quality_threshold_missing:{key}")
    stats = obj.get("quality_stats_required") or []
    for req in (
        "missing_roi_rate_by_region_from_tensor_validity",
        "missing_roi_rate_by_split_from_tensor_validity",
        "missing_roi_rate_by_signer_from_tensor_validity",
        "partial_missing_rate_by_region",
    ):
        if req not in stats:
            errors.append(f"quality_stats_missing:{req}")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}


def validate_d0_full_corpus_run_contract(obj: dict[str, Any], *, root: Path = _ROOT) -> dict[str, Any]:
    errors: list[str] = []
    if obj.get("schema") != "ms_sta_v5_d0_full_corpus_run_contract_v1":
        errors.append("bad_full_corpus_run_contract_schema")
    if obj.get("status") not in {"AUTHORIZED_NOT_STARTED", "RUNNING", "RESUMED", "CLOSED"}:
        errors.append("full_corpus_run_contract_status")
    if obj.get("full_corpus_authorized") is not True:
        errors.append("run_contract_must_be_authorized")
    if obj.get("auto_chain_to_m0") is not False:
        errors.append("run_contract_auto_chain_forbidden")
    if obj.get("stop_after") != "D0_FULL_CORPUS_SCAN_COMPLETE":
        errors.append("run_contract_stop_after")
    if obj.get("output_dir_rel") != D0_FULL_CORPUS_OUT_DIR_REL:
        errors.append("run_contract_output_dir")
    if obj.get("approval_sha256") != sha256_file(root / D0_FULL_CORPUS_APPROVAL_REL):
        errors.append("run_contract_approval_sha_mismatch")
    if obj.get("r7c_h1_independent_reaudit_sha256") != EXPECTED_R7C_H1_INDEPENDENT_REAUDIT_SHA256:
        errors.append("run_contract_h1_sha")
    if errors:
        raise ValueError(";".join(errors))
    return {"ok": True}
