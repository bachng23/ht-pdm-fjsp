# Experiment 1 — Leave-one-component-out ablation của RA-QMIX

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-27
- Verification Status: PLAN ONLY — chưa implement hoặc chạy thí nghiệm này.
- Version Label: ra_qmix_leave_one_out_v1
- Trạng thái: protocol đề xuất; khóa cấu hình và commit trước full run.

## 1. Mục tiêu và phạm vi

Đáp ứng ưu tiên 1 trong email của thầy: xác định đóng góp của technician-edge
utilities, queue-conditioned mixing và counterfactual-consistency loss trong
mô hình RA-QMIX đầy đủ.

Research question: khi giữ nguyên môi trường và training protocol, bỏ riêng
từng thành phần có làm tăng objective cost so với Full RA-QMIX không?

Đây là vòng đánh giá development có kiểm soát. Giữ môi trường 3 máy / 2 kỹ thuật
viên. Không mở rộng quy mô, không quét hyperparameter và không mở sealed test
panel trong experiment này. Parameter-matched QMIX thuộc experiment 2.

## 2. Giả thuyết và cách diễn giải

Với mỗi thành phần k, định nghĩa:

`delta_k = mean_cost(RA-QMIX bỏ k) - mean_cost(Full RA-QMIX)`.

- H-edge: bỏ edge utilities làm tăng objective cost ở nominal tại 50k episodes.
- H-queue: bỏ queue-conditioned mixing làm tăng objective cost ở cùng điều kiện.
- H-CF: bỏ CF loss làm tăng objective cost ở cùng điều kiện.
- So sánh tham chiếu: Full RA-QMIX có cost thấp hơn standard QMIX.

Delta dương ủng hộ đóng góp của thành phần trong cấu hình đang thử. Delta gần 0,
không ổn định giữa seeds, hoặc độ bất định lớn nghĩa là chưa đủ bằng chứng.
Delta âm cho thấy cần xem lại lợi ích của thành phần trong điều kiện này.
Không đặt điều kiện experiment chỉ thành công khi Full RA-QMIX thắng.

Leave-one-out đo đóng góp có điều kiện khi các thành phần còn lại hiện diện;
không xác định mọi tương tác giữa các thành phần và không chứng minh từng
thành phần đều cần thiết trong mọi môi trường.

## 3. Năm biến thể

| Reporting name | Algorithm ID đề xuất | Edge utilities | Queue mixer | CF loss |
|---|---|---|---|---|
| Standard QMIX | `qmix` | Không | Không | Không |
| RA-QMIX − edge | `tqmix_no_edge` | Không | Có | Có |
| RA-QMIX − queue | `tqmix_no_queue` | Có | Không | Có |
| RA-QMIX − CF | `tqmix_no_cf` | Có | Có | Không |
| Full RA-QMIX | `tqmix` | Có | Có | Có |

Trong repo, `tqmix` là ID của mô hình đầy đủ; báo cáo sử dụng tên RA-QMIX và
manifest lưu ánh xạ này. Ba ID `tqmix_no_*` cần được implement.

Quy tắc can thiệp:

- Bỏ edge: thay technician-edge Q head bằng local Q head chuẩn; giữ queue mixer
  và CF loss hoạt động.
- Bỏ queue: dùng mixer chuẩn của QMIX; giữ edge head và CF loss. Không xóa thông
  tin queue khỏi observation chung, vì sẽ tạo thêm một can thiệp về thông tin.
- Bỏ CF: đặt trọng số CF bằng 0; giữ nguyên kiến trúc edge và queue.
- Full và các biến thể có CF dùng `lambda_cf = 0.05`, tương ứng hệ số hiện tại.
- Cùng action space, action mask, observation, reward, FIFO semantics và
  evaluation policy; chỉ khác thành phần được chỉ định.
- Lưu số tham số trainable của từng mô hình, tách agent và mixer; không cộng
  target-network copies vào số tham số trainable. Chưa khớp số tham số ở vòng này.

Bộ hiện tại trong `passive_technician_ablation.py` dùng các biến thể chỉ có một
thành phần. Giữ bộ cũ để truy xuất; tạo runner mới cho leave-one-out.

## 4. Môi trường và điều kiện kiểm soát

Train tất cả mô hình trên `in_distribution`. Đánh giá cùng checkpoint trên
bốn scenario hiện có trong `passive_technician_long_comparison.py`:

| Scenario | Cấu hình |
|---|---|
| `in_distribution` | Snapshot đầy đủ của `stress_config()` |
| `early_failure` | Base + horizon 18, failure age 4, failure probability 0.60 |
| `slow_service` | Base + horizon 18, service times `((3,5),(5,3),(4,4))` |
| `combined_pressure` | Base + cả hai thay đổi trên, horizon 18 |

Giữ recovery-at-maintenance-start để cô lập thay đổi thuật toán so với setup
hiện tại. Trước chạy, phải xác minh bằng test thời điểm reset age/failure,
machine availability trong service và thời điểm tính downtime; lưu semantics
đã xác minh vào manifest. Nếu implementation không khớp mô tả, sửa protocol
công khai trước full run, không âm thầm đổi simulator.

Không đổi recovery semantics giữa các biến thể. Thí nghiệm recovery-at-completion
sẽ được thực hiện riêng; kết luận vòng này chỉ áp dụng cho semantics đã khóa.

Các scenario có horizon khác nhau: so sánh mô hình trong từng scenario, không
gộp raw total cost của cả bốn thành một bảng xếp hạng chung. Có thể bổ sung
cost/timestep như metric phụ và ghi rõ mẫu số.

## 5. Seeds, budget và stopping rule

| Hạng mục | Full development run | Smoke trên Mac |
|---|---|---|
| Training seeds | 11, 12, 13 | 11 |
| Evaluation seeds | 101–200 | 101–103 |
| Sealed test seeds dự kiến | 201–300, không chạy | Không chạy |
| Checkpoints | 20,000 và 50,000 episodes | 8 và 16 episodes |
| Training scenario | Nominal | Nominal |
| Evaluation scenarios | Cả 4 | Cả 4 |
| Variants | Cả 5 | Cả 5 |

- Kiểm tra lịch sử sử dụng seeds trước khi gọi panel 201–300 là sealed; nếu đã
  từng được xem, phải chọn và ghi một panel mới trước full run.
- Training seed điều khiển khởi tạo mạng, exploration, replay và luồng episode
  training. Evaluation dùng RNG riêng, không tác động RNG hoặc trạng thái train.
- Train liên tục đến 50k; checkpoint 20k thuộc cùng training trajectory, không
  phải một run độc lập với exploration schedule khác.
- Dùng cùng `ValueTrainSettings` cho mọi biến thể, lưu toàn bộ resolved settings
  thay vì chỉ dựa vào defaults của code. Không tune riêng từng biến thể.
- 50k nominal là endpoint chính. 20k và stress scenarios là phân tích phụ.
- Không early-stop theo performance; không chọn checkpoint tốt nhất sau khi
  xem evaluation. Chỉ dừng vì lỗi, NaN/Inf hoặc vi phạm engineering audit.
- Run lỗi được ghi `FAILED`/`INCOMPLETE`; không loại seed xấu để cải thiện mean.
  Nếu cần chạy lại sau sửa lỗi, tạo run mới, giữ artifacts cũ và giải thích lý do.

Quy mô dự kiến: 15 training trajectories, 750,000 training episodes, 30
checkpoints và 12,000 evaluation episodes. Smoke: 5 trajectories, 80 training
episodes, 10 checkpoints và 120 evaluation episodes. Không thêm fixed policies
vào số đếm của runner này.

Ba training seeds là vòng diagnostic ban đầu, chưa đủ cơ sở để khẳng định độ
ổn định rộng. Không coi 100 evaluation episodes là 100 lần train độc lập.

## 6. Metrics và định nghĩa logging

### Primary metric

Mean objective cost trên evaluation seeds 101–200, tính riêng cho từng training
seed tại nominal/50k. Báo cáo cả ba seed means, mean và SD giữa training seeds.

### Secondary và mechanism metrics

| Nhóm | Metrics cần ghi |
|---|---|
| Cost | Failure, downtime, maintenance, queue-waiting cost; các penalty còn lại |
| Reliability | Số failure events, machine downtime steps |
| Queue | Mean/max queue length, tổng waiting steps, waiting time/request |
| Requests | Tổng requests, busy-technician requests, invalid requests, collisions |
| Maintenance | Service starts/completions, preventive/corrective starts, defer fraction |
| Resources | Busy steps / horizon cho từng technician, chênh lệch utilization max–min |
| Training | TD loss, raw CF loss, weighted CF loss, total loss, epsilon, update count |

Yêu cầu định nghĩa trước full run:

- Phân rã cost phải khớp đúng objective đang dùng; không tạo cost mới để đủ bốn
  cột. Nếu simulator không charge một loại cost, lưu 0 và giải thích, đồng thời
  lưu chỉ số vật lý tương ứng. Tổng các thành phần và penalty phải khớp objective.
- Tách queue waiting của request chưa được phục vụ khỏi waiting time của request
  đã bắt đầu service; ghi pending count ở cuối horizon để thấy censoring.
- Technician utilization đo thời gian thực sự bận; khác biệt utilization là
  mô tả, không mặc định cân bằng tuyệt đối tốt hơn khi kỹ năng không đồng nhất.
- Không gán preventive maintenance là unnecessary maintenance. Vòng này dùng
  age/failure status tại service start để mô tả; chưa kết luận hành động là thừa.
- Raw CF loss của biến thể không dùng CF ghi null/N/A; weighted CF term bằng 0.
- Nếu metric chưa được instrument, phải bổ sung và test trước full run; không
  điền số 0 thay cho dữ liệu thiếu.

## 7. Kế hoạch phân tích

1. Kiểm tra completeness, schema, seed split và feasibility trước performance.
2. Với mỗi training seed, lấy trung bình trên cùng 100 evaluation seeds.
3. Tính paired delta giữa từng ablation và Full tại nominal/50k; báo cáo ba
   delta theo seed, mean delta, SD và khoảng min–max.
4. Báo cáo thêm delta Full so với QMIX. Nếu đưa relative delta (%), ghi rõ mẫu
   số là cost của Full cho ablation, cost QMIX cho so sánh baseline.
5. Vòng 3 seeds ưu tiên mô tả effect và độ biến thiên; không dùng p-value hoặc
   interval tính từ việc gộp 300 episodes như các independent training runs.
6. Lập bảng tương tự cho từng stress scenario và checkpoint 20k; ghi rõ là
   secondary/exploratory. Không thay endpoint chính theo kết quả.
7. Đối chiếu cost decomposition với failures, queue waiting và utilization để
   đề xuất cơ chế; tương quan diagnostic chưa chứng minh quan hệ nhân quả.

Deliverables phân tích: bảng nominal/50k, bảng delta theo seed, bảng 4 scenario
× 2 checkpoint, biểu đồ cost theo seed và biểu đồ phân rã chi phí. Giữ lại dấu
hiệu 20k→50k nominal tốt lên nhưng stress kém đi như quan sát exploratory.

Giới hạn: bỏ module có thể thay đổi capacity/optimization cùng cấu trúc. Kết
quả vòng này chưa tách hết capacity confound; experiment 2 bổ sung QMIX có số
tham số tương đương. Không sử dụng kết quả single-component cũ thay leave-one-out.

## 8. Implementation và engineering gates

Các file dự kiến:

- `src/ht_pdm_fjsp/passive_technician_value_decomposition.py`: thêm component
  configurations, ba variant IDs và metadata tham số.
- `src/ht_pdm_fjsp/passive_technician_leave_one_out.py`: runner mới, snapshots,
  bốn scenario, checkpoint/evaluation loop và summary.
- Tests tương ứng cho variant wiring, metrics, checkpoint round-trip và runner.

Trình tự:

1. Kiểm tra working tree, bảo toàn thay đổi đang có và chuẩn bị branch riêng
   `codex/ra-qmix-leave-one-out`; không gom thay đổi không liên quan vào commit.
2. Khóa protocol version, configs, seeds và metric definitions.
3. Implement variants và logging, giữ behavior của các algorithm IDs cũ.
4. Chạy tests, rồi smoke CPU trên Mac.
5. Chỉ commit/push code đã qua gates; ghi Git SHA trong manifest.
6. Sau push, bàn giao đúng compound command để người dùng chạy trên lab.
7. Người dùng kéo artifacts về Mac; chỉ phân tích bản local đã kiểm tra.

Tests cần chứng minh:

- Mỗi variant bật/tắt đúng một thành phần; output shape và action mask hợp lệ.
- Nhánh loss đúng trọng số và CF có gradient khi được bật.
- Save/load khôi phục kiến trúc, variant, settings và Q outputs trên input cố định.
- Test timeline xác minh recovery/availability semantics; test cost reconciliation
  và queue/utilization counters trên trajectory nhỏ có kết quả biết trước.
- Runner từ chối output directory không rỗng, không mở sealed panel, evaluation
  không cập nhật weights hoặc thay đổi training RNG/state.

Smoke gate: process exit 0; đúng 120 evaluation rows và 10 checkpoints; mọi
variant thực hiện ít nhất một optimizer update (smoke cần giảm learning-starts
và batch size phù hợp, ghi rõ override); losses hữu hạn; save/load đạt; schema
và feasibility audits đạt; có `tqdm` cho training và multi-seed evaluation.

Full gate: đủ 12,000 evaluation rows, 30 checkpoints và 15 trajectories; không
duplicate keys; dữ liệu loss hữu hạn; cost reconciliation đạt sai số float đã
định trước (đề xuất absolute tolerance 1e-6); không có vi phạm resource semantics.
Busy requests/collisions phải được phân biệt với vi phạm feasibility vì có thể
là hành vi hợp lệ được môi trường xử lý bằng queue/penalty.

## 9. Artifact schema và monitoring

Mỗi run dùng một UTC timestamp mới, ví dụ
`artifacts/ra_qmix_leave_one_out_full_cpu_<UTC>/`. Không ghi đè run cũ.

```text
manifest.json
benchmark_config.json
resolved_config.json
parameter_counts.csv
training_progress.csv
training_episodes.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
paired_differences.csv
summary.json
<algorithm>/train_seed_<seed>/checkpoints/<budget>/model.pt
```

Tên checkpoint thực tế có thể theo serializer hiện có; manifest phải liệt kê
đường dẫn thực tế. Console log nằm cạnh run directory dưới tên `<run_id>.log`.

- Manifest: protocol version, Git SHA/dirty status, runtime/dependencies, device,
  seeds, scenario configs, recovery semantics, component flags, hyperparameters,
  parameter counts, timestamps, status, expected/actual counts và audit results.
- `episodes.csv`: khóa duy nhất `(algorithm, train_seed, scenario, budget,
  eval_seed)` cùng toàn bộ episode metrics. `coordination.csv` chứa kết quả
  audits thực sự, không chỉ sao chép episode table.
- Training CSV lưu algorithm, train seed, episode, environment steps, update
  count, losses và elapsed time; flush định kỳ để theo dõi khi process còn chạy.
- Theo dõi `tqdm`, process-alive, thời điểm progress cuối và partial artifacts.
  Đo throughput từ smoke để ước lượng runtime; chưa đặt wall-clock timeout giả
  định. Nếu log ngừng tiến triển bất thường, chẩn đoán trước khi quyết định dừng.
- Không resume nếu chưa có resume contract được test; không xóa raw artifacts
  sau khi đọc performance.

## 10. Handoff và điều kiện hoàn thành

Tuân thủ `docs/experiment_workflow.md`: full run mặc định CPU trên Ubuntu lab
`bachng@100.111.83.52`, repository `~/ht-pdm-fjsp`. Không SSH hoặc tự chạy lab,
không dùng tmux; người dùng giữ terminal mở trong quá trình chạy.

Plan này chưa cung cấp lệnh full run có thể thực thi vì runner mới chưa được
implement/test/push. Sau smoke gate và push, handoff phải chứa:

1. Branch, commit và kết quả tests/smoke.
2. Một compound command hoàn chỉnh để fetch/switch/pull branch, sync dependencies,
   tạo timestamped directory và chạy full profile với `tee` cùng `set -o pipefail`.
3. Lệnh sau để người dùng tự kéo kết quả về Mac sau khi full run hoàn tất:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && \
rsync -avhP --partial \
  bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ \
  "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

Experiment hoàn thành khi có full artifacts hợp lệ và báo cáo phân tích nêu
rõ thành phần nào được bằng chứng ủng hộ, thành phần nào chưa rõ hoặc bất lợi,
độ biến thiên giữa seeds và giới hạn diễn giải. Việc Full RA-QMIX không thắng
không làm mất tính hợp lệ của một experiment được thực hiện đúng protocol.
