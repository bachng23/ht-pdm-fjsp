# Canonical experiment workflow

Tài liệu này là workflow mặc định cho mọi code experiment trong repository.
Máy chạy full experiment là Ubuntu lab tại `bachng@100.111.83.52`; full run trên
lab ưu tiên CPU, trừ khi một experiment ghi rõ yêu cầu khác.

## Quy tắc bắt buộc

1. Khóa protocol trước khi xem kết quả: hypothesis, primary/secondary metrics,
   train/development/test seeds, stopping rule và artifact schema.
2. Implement trên một branch `codex/<experiment-name>` riêng.
3. Chạy unit/integration tests và một smoke experiment nhỏ trên Mac trước.
4. Chỉ commit và push lên GitHub sau khi smoke gate đạt.
5. Sau khi push, assistant chỉ hiển thị một compound command để người dùng tự
   chạy trực tiếp trong terminal máy lab. Assistant không được SSH vào lab,
   không nhập lệnh vào terminal và không tự khởi chạy full experiment.
6. Mỗi run dùng một thư mục artifact có UTC timestamp mới. Không ghi đè âm thầm
   lên full run cũ.
7. Training và multi-seed evaluation phải hiển thị tiến độ bằng `tqdm`.
8. Không dùng `tmux` trên lab. Giữ terminal mở cho tới khi lệnh hoàn tất.
9. Không dùng `set -euo pipefail`. Được dùng riêng `set -o pipefail` để lỗi của
   experiment không bị `tee` che mất.
10. Sau khi full run hoàn tất, assistant hiển thị một lệnh
    `rsync -avhP --partial` để người dùng tự chạy trên Mac. Assistant không tự
    khởi chạy transfer.

## 1. Khóa experiment protocol

Mỗi plan trong `docs/` phải ghi tối thiểu:

- research question và falsifiable hypothesis;
- môi trường, cấu hình và các điều kiện so sánh;
- primary metric và secondary/mechanism metrics;
- train seeds, development/evaluation seeds và sealed test seeds;
- fixed training budget hoặc stopping rule được khóa trước;
- tiêu chí pass/fail của smoke và full run;
- artifact schema;
- giới hạn diễn giải, đặc biệt phân biệt diagnostic association với causal
  effect.

Không thay stopping rule hoặc seed panel sau khi đã nhìn kết quả, trừ khi tạo
một protocol amendment mới và ghi rõ đây là exploratory follow-up.

## 2. Implement và kiểm tra trên Mac

Quy trình tối thiểu:

```bash
git switch -c codex/<experiment-name>
uv sync --frozen
uv run pytest -q
RUN_ID="<experiment-name>_smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "artifacts/$RUN_ID"
set -o pipefail
uv run <experiment-command> \
  --profile smoke \
  --device cpu \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/$RUN_ID.log"
```

Smoke gate phải kiểm tra được:

- process kết thúc thành công;
- mọi episode cần thiết đã hoàn tất;
- model/checkpoint có thể save và load;
- CSV/JSON/manifest đúng schema;
- feasibility và coordination audits đạt;
- progress bar xuất hiện trong training và evaluation;
- sealed test panel chưa bị mở.

Smoke artifacts là engineering evidence, không phải final scientific result.

## 3. Commit và push

Chỉ push sau khi tests và smoke đều đạt:

```bash
git status --short
git diff --check
git add <verified-files>
git commit -m "<experiment commit message>"
git push -u origin codex/<experiment-name>
```

Không commit secrets, local virtual environments, downloaded datasets hoặc
large run artifacts nếu chúng đang được `.gitignore` loại trừ.

## 4. Handoff một lệnh để người dùng chạy trong terminal Ubuntu lab

Ngay sau khi code đã push, assistant phải dừng execution và hiển thị cho người
dùng **một compound command duy nhất**. Người dùng tự chạy lệnh đó sau khi đã mở
terminal trên lab. Assistant không được thử SSH, không điều khiển terminal lab,
không chạy command thay người dùng. Command handoff không thêm `ssh`, không dùng
`tmux`, và không dùng `set -euo pipefail`.

Template chuẩn:

```bash
cd ~/ht-pdm-fjsp && \
export PATH="$HOME/.local/bin:$PATH" && \
if ! command -v uv >/dev/null 2>&1; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi && \
git fetch origin && \
(git switch codex/<experiment-name> || git switch --track origin/codex/<experiment-name>) && \
git pull --ff-only && \
uv sync --frozen && \
set -o pipefail && \
RUN_ID="<experiment-name>_full_cpu_$(date -u +%Y%m%dT%H%M%SZ)" && \
mkdir -p "artifacts/$RUN_ID" && \
uv run <experiment-command> \
  --profile full \
  --device cpu \
  --output-dir "artifacts/$RUN_ID" \
  2>&1 | tee "artifacts/$RUN_ID.log"
```

Nếu experiment phụ thuộc checkpoint có sẵn, chèn các lệnh
`test -f <checkpoint>` sau `uv sync --frozen` và trước khi tạo `RUN_ID`. Nếu
bất kỳ checkpoint nào thiếu, full run phải dừng trước khi tiêu tốn compute.

Lab workflow mặc định dùng `--device cpu`. Chỉ chuyển sang `cuda` sau khi code
đã có explicit GPU support và preflight sau đạt:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

Vì không dùng `tmux`, không đóng terminal hoặc ngắt kết nối trong lúc chạy. Khi
run bị gián đoạn, chỉ dùng `--resume` trên đúng artifact directory nếu runner có
resume contract đã được test; nếu không, tạo một timestamped run mới.

## 5. Artifact contract

Mỗi full run nên chứa tối thiểu:

```text
artifacts/<timestamped-run-id>/
├── manifest.json                 # protocol, commit, runtime, status, seeds
├── benchmark_config.json         # source configuration snapshot
├── resolved_config.json          # effective/scaled configuration
├── training_progress.csv
├── training_episodes.csv
├── episodes.partial.csv          # refreshed during a long run
├── episodes.csv
├── decisions.csv                 # when step-level diagnostics are required
├── coordination.csv              # feasibility/conflict audit
├── summary.json
└── <algorithm>/
    ├── checkpoints/
    └── model.pt
```

Tên file cụ thể có thể mang experiment prefix, nhưng manifest phải ghi đầy đủ
output files, Git commit, resolved device, settings, seed panels, completion
status và audit gate. Log console được lưu cạnh thư mục run dưới tên
`artifacts/<timestamped-run-id>.log`.

## 6. Handoff một lệnh để người dùng kéo toàn bộ kết quả về Mac

Assistant hiển thị lệnh sau để người dùng tự chạy trong terminal Mac sau khi
full run đã hoàn tất. Assistant không tự chạy `rsync`. Đây là lệnh canonical
cho lab `100.111.83.52`:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

Trailing slash ở `artifacts/` là có chủ ý: toàn bộ nội dung của thư mục remote
được đồng bộ vào `lab_results/`. Chạy lại cùng lệnh sẽ tiếp tục partial transfer
và chỉ cập nhật phần còn thiếu.

## 7. Phân tích kết quả

Chỉ phân tích bản đã kéo về Mac:

1. xác nhận manifest có `status=COMPLETED`;
2. xác nhận episode/checkpoint counts đúng protocol;
3. kiểm tra feasibility và coordination audit trước khi đọc performance;
4. kiểm tra seed overlap và sealed test status;
5. phân tích paired differences trên cùng evaluation seeds;
6. báo cáo effect size/uncertainty cùng giá trị trung bình;
7. tách rõ confirmatory result, exploratory result và failed/incomplete run;
8. không xóa hoặc chỉnh raw artifacts sau khi đã phân tích.

## 8. Mẫu handoff bắt buộc

Sau khi code đã push, assistant kết thúc tại checkpoint handoff. Câu trả lời
phải cung cấp đúng hai lệnh chính để người dùng tự chạy:

1. một lệnh compound để pull branch và chạy full experiment trong terminal lab;
2. một lệnh `rsync` để người dùng kéo toàn bộ artifacts từ
   `bachng@100.111.83.52` về Mac.

Kèm theo branch, commit, smoke result và ước lượng phạm vi artifact; không yêu
cầu người dùng tự ghép nhiều đoạn lệnh rời rạc. Assistant không được thực thi
hai lệnh handoff này, kể cả khi có thể truy cập host.
