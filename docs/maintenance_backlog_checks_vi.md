# Kiểm tra tiếp theo: giả định thực tế, cascade và baseline đơn giản

## Material Passport
ARS experiment-agent/validate; 2026-10-05 Asia/Taipei. Verification Status: ANALYZED. Nguồn: các paper primary đã kiểm tra và một shock đã mở của repair v1. Replay dùng chung implementation, không phải tái lập độc lập. Experiment mới chưa mở full panel.

## Kết luận hiện tại
Chưa nên chốt problem thành “phải có cơ chế đặt lịch phức tạp”. Một lỗi quyết định tại tick9 của S có thể tạo ra gần như toàn bộ cascade trong case900215/shock901200. Đồng thời chưa có căn cứ thực nghiệm để dùng CM/PM2 hoặc4 làm đại diện cho một nhà máy cụ thể. Cần kiểm tra một baseline ngưỡng dựa trên backlog trên seed mới; protocol và code đã chuẩn bị riêng.

## Căn cứ từ paper và ranh giới mô hình

| Nguồn primary đã đọc | Bằng chứng có thể dùng | Không hỗ trợ điều gì |
|---|---|---|
| [Savsar & Çiçek2021, DOI10.31590/ejosat.1035168](https://dergipark.org.tr/en/download/article-file/2126133), PDF p391/Table2 và đoạn dưới bảng; đã đọc ảnh trang | Nhà máy đồ hộp: thời gian sửa trung bình5–60phút; PM10giờ ngoài giờ sản xuất. CM cần1thợ cơ khí+1điện+1phụ; PM cần2+2+2. | PM là đợt khác phạm vi/nhân lực; không thể quy đổi thành tỷ lệ CM/PM cho cùng một thao tác. Đây là phản ví dụ đối với giả định CM luôn dài hơn PM, không phải calibration cho simulator. Paper mô tả thời gian thu thập một năm ở p389 nhưng hai năm ở p390; giữ nguyên bất nhất này. |
| [Periodic flexible maintenance planning…2019, DOI10.1007/s40092-019-0314-x](https://link.springer.com/article/10.1007/s40092-019-0314-x), assumptions và Numerical analysis | PMduration M được sinh10–50, CMmean μR10–100, CM exponential; máy dừng khi bảo trì. | Đây là dải giả lập, không phải đo thực tế; không có tỷ lệ chung2/4. |
| [Demirci etal., online2024/volume2025, DOI10.1007/s10696-024-09544-y](https://link.springer.com/article/10.1007/s10696-024-09544-y), §§1,3.3,4 | Số thợ giới hạn tạo phụ thuộc tài nguyên; mô hình có thời gian service exponential và so sánh heuristic/index với threshold. | Không chứng minh tỉ lệ CM/PM2/4, không chứng minh cách S của ta hay MARL cần thiết. Bài đã giải quyết một phần vấn đề chung. |
| [Quintana etal.2009, DOI10.1080/00207540701824225](https://experts.azregents.edu/en/publications/corrective-maintenance-through-dynamic-work-allocation-and-pre-em/), abstract trên trang cơ quan tác giả | Case study sản xuất so sánh phân công động/pre-emption, machine availability và mechanic utilization. | Chỉ xác minh abstract/metadata; chưa đọc fulltext nên không lấy thông số duration/capacity từ bài này. |

Paper mới được tải tại /Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_backlog_checks_20261005/additional_papers/savsar_cicek_2021.pdf, kèm text. “CM làm mất sản lượng” có căn cứ; “CM luôn chiếm thợ lâu hơn PM” chưa có. Downtime tổng gồm chờ người/linh kiện và actual service cần đo riêng; simulator hiện chỉ giả định service nhân2/4 và tự sinh waiting. Không đưa tổng downtime thực tế vào service nếu chưa tách thành phần.

Để calibration cần dữ liệu cùng nhóm máy/cùng phạm vi thao tác: loại PM/CM, thời điểm yêu cầu/start/end service/return-to-production, nhân lực theo vai trò, parts/logistics delays, production calendar và các phiên không hỏng. Các tỷ lệ1/2/4 tiếp tục là sensitivity trong diagnostic synthetic; chưa diễn giải kết quả thành impact nhà máy.

## Counterfactual ca đã mở
Giữ nguyên6máy,2thợ,PM[2,6],CM[8,24],64jobs/máy,wear12,p=.6,shock901200, S lead6. Can thiệp duy nhất một yêu cầu ở tick được chỉ định; không thay hazard, không hủy queue/service cũ; sau tick đó tiếp tục S bình thường. Sáu can thiệp được ghi trước smoke; không bỏ kết quả bất lợi.

| Intervention | Có áp dụng? | Makespan | Failures |
|---|---:|---:|---:|
| S nguyên bản | — |657|53|
| m4→thợ0/tick8 |Không, thợ đang bận|657|53|
| m5→thợ0/tick8 |Không, thợ đang bận|657|53|
| m4→thợ0/tick9 |Có|666|53|
| **m5→thợ0/tick9** |**Có**|**315**|**0**|
| m4→thợ0/tick10 |Có|653|53|
| m5→thợ0/tick10 |Có, trùng hành động S|657|53|

Can thiệp m5/tick9 giảm342ticks (52.05%) trong shock này. Cơ chế cần đọc từ trace: S để thợ nhanh rảnh tick9 rồi bảo trì m5 ở10–11; m4 hỏng ở cuối11, sửa CM8ticks khóa thợ nhanh và lan sang các máy khác. Chuyển PMm5 sang9–10 tạo chỗ để S bảo trì m4 từ11, tránh failure đầu tiên. Không nên sửa S dựa riêng vào case này rồi gọi đó là bằng chứng generalization.

Đây là bằng chứng can thiệp trong simulator, trên một instance/shock đã được chọn sau khi biết kết quả. Không phải effect trung bình, không có CI24instances, không xem đây là chiến lược biết trước failure. Full run mới sẽ báo cả20shock cũ cho case này, tách hoàn toàn khỏi panel mới.

## Experiment mới sẽ trả lời gì
H: ngưỡng health cố định đã tune; A: dùng cùng dispatcher nhưng yêu cầu sớm khi thợ được chọn có backlog; S: reservation heuristic cũ. A không dùng lịch tương lai hoặc future shocks. Gain0 trùng H tuyệt đối. Tune development riêng theo ratio rồi khóa; eval24instances×20shocks hoàn toàn mới, full34,560episodes+13,824dev;140counterfactualepisodes riêng. Primary ratio2: upperCI95% của (A-S)/S≤2%; guard ratio4: upperCI95% của gap p95≤5%. Hai tiêu chí mới đánh giá adequacy/robustness baseline, không thay gate5% của experiment cũ.

Nếu A đủ tốt, ưu tiên hướng giải pháp đơn giản và xem coordination thêm được gì. Nếu chưa đủ, xem failure/tail traces và residual burden trước khi thiết kế giải pháp phức tạp. Kết quả phụ cell/ratio khác chỉ descriptive; không chốt novelty/MARL dựa trên một diagnostic.

## Fallacy scan
Đã kiểm11loại: đơn vị suy luận/pseudoreplication; effect size so với significance; absence-of-evidence; association/causality; multiplicity; post-selection; leakage/tuning; stopping/censoring; distribution/tails; matched-control confounds; phạm vi generalization. Rủi ro chính: case được chọn hậu nghiệm, ratio chưa calibrated, A có8candidate so vớiH4, realized service không bằng nhau dù rowmeans matched. Bootstrap lấy instance aggregate, giữoutlier; không gộp140CF vào34,560eval. Xem /Users/bachng/Coding/Reinforcement Learning/marl/reports/maintenance_backlog_checks_20261005/smoke_audit.json để kiểm hash/count/replay/seeds.
