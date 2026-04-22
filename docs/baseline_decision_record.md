# Baseline Decision Record (C0.5)

## Context

Sau C0 lineage reconciliation:

- Phase B đã hoàn tất.
- Frozen baseline source-of-truth claim trong governance là **83.33%**.
- Kết quả tái dựng lineage có thể audit được hiện tại cho cả legacy protocol và hardened same-protocol đều là **94.00%**.
- Per-image diff giữa legacy protocol tái dựng và hardened same-protocol: **0 ảnh đổi nhãn**.

## Decision

### 1) Trạng thái 83.33%

- **83.33% được giữ là historical baseline claim**, **không phải** auditable technical baseline ở thời điểm hiện tại.
- Lý do: chưa thu hồi được artifact đóng băng (script output / report / run metadata immutable) tái lập đúng 83.33% từ lineage có thể kiểm chứng.

### 2) Reproducible technical anchor

- **94.00% được chấp nhận làm reproducible technical comparison anchor tạm thời** cho các phase kỹ thuật kế tiếp.
- Anchor này dùng để so sánh kỹ thuật nhất quán (protocol, dataset path, threshold, weights hash) và **không rewrite lịch sử**.
- Historical claim 83.33% vẫn được giữ nguyên vai trò lịch sử trong tài liệu governance.

### 3) Hardening impact statement

- Theo C0 per-image diff: hardening **không đổi nhãn dự đoán** khi chạy cùng protocol và cùng dataset (prediction changed count = 0).
- Vì vậy drift 83.33 -> 94.00 **không có bằng chứng** là do hardening classifier behavior trong run so sánh này.

## Frozen assumptions for this decision

- Dataset path (actual used): `D:\Codes\face\ai_deepfake\dataset_final\test`
- Default operating threshold: **`0.65`** (C1 policy; C0 lineage tables in `docs/benchmark_lineage_report.md` used `0.5` where labeled.)
- Lineage commit (AI feature baseline lineage): `8905873`
- Script lineage:
  - `ai_deepfake/test_model.py` (from commit `8905873`)
  - `ai_deepfake/detect.py` (from commit `8905873`)
- Current hardening commits:
  - PR2: `b48c017`
  - PR3: `cb81991`
  - PR4: `198ce98`
  - C0 artifacts: `3cbb20e`

## Model files used (local-only, not committed)

- `best_model.pth`
  - size: `16339622` bytes
  - sha256: `1ae8c63ab7ea5d4629d771c0dba664f1bbfcafa2df713452c523a36d76736bec`
- `best_model_v2.pth`
  - size: `296785927` bytes
  - sha256: `c2d114932ff3586c56c93b0bff3e73883ddafc9b702d247cdc72dc4f6906b01c`

## Governance output links

- `docs/baseline_frozen_reproduced.json`
- `docs/baseline_hardened_same_protocol.json`
- `docs/per_image_diff.csv`
- `docs/benchmark_lineage_report.md`
- `docs/baseline_protocol_frozen.md`
- `docs/baseline_reproducible_anchor.json`

## Can C1 start?

- **Yes, conditionally safe to start C1** (calibration + threshold analysis), using the reproducible technical anchor frozen in C0.5.
- Constraint remains: không model replacement, không training mới, không rewrite baseline history.
