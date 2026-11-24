# Backend chạy nhiều pod

Trạng thái: **đã làm và triển khai lên cluster test (image `ext-ui49`), chưa commit.** Cập nhật 2026-11-24.
Backend vẫn để 1 pod. Chạy được từ 2 pod trở lên khi có `REDIS_URL`.

## 1. Cách làm

Mọi pod backend **giống hệt nhau**: mỗi pod tự chạy lượt chat của nó và tự gửi chữ về trình duyệt, như trước giờ. Không có pod web riêng và pod worker riêng, không có hàng đợi.

Điều cần xử lý là những thứ trước đây chỉ nằm trong bộ nhớ của từng pod. Chúng được đưa vào Redis theo cách nhẹ nhất, mà không di chuyển lượt chạy nào:

| Thứ | Trước | Nay |
|---|---|---|
| Nút Dừng | Chỉ hủy được lượt chạy của pod nhận request | Gửi lệnh hủy tới đúng pod đang chạy |
| Nghiên cứu nền | Trạng thái và lệnh hủy chỉ có ở pod đã bắt đầu | Đăng ký trong Redis, pod nào cũng xem và hủy được |
| Trần sandbox | Mỗi pod tự đếm | Khóa xếp chỗ, đặt chỗ, đang dùng nằm trong Redis |
| Cập nhật, scale xuống | Pod bị tắt sau 30 giây, cắt lượt chat | Pod chờ lượt chat xong rồi mới tắt (tối đa 900 giây) |

## 2. Kiến trúc

```
Trình duyệt ──▶ Ingress ──▶ Pod backend A ─┐
                        └─▶ Pod backend B ─┤── Redis (khóa cuộc trò chuyện, kênh điều khiển,
                                           │          sổ nghiên cứu nền, xếp chỗ sandbox)
                                           ├── Postgres (lịch sử, checkpoint)
                                           └── API Kubernetes (SandboxClaim)
```

## 3. Dữ liệu trong Redis

Lịch sử hội thoại **không** nằm trong Redis; nó vẫn ở Postgres. Redis chỉ giữ trạng thái ngắn hạn, mọi mục đều có hạn dùng.

| Key | Kiểu | Nội dung | Hạn dùng |
|---|---|---|---|
| `jarvis:conv-lock:{thread}` | String | Khóa cuộc trò chuyện. Giá trị là `<tên pod>-<pid>-<hậu tố>:<chuỗi ngẫu nhiên>`, nên khóa cho biết pod nào đang chạy | 30 giây, gia hạn bằng heartbeat |
| `jarvis:ctl:{instance}` | Kênh Pub/Sub | Lệnh hủy gửi tới một pod | không lưu |
| `jarvis:cancel:run:{thread}`, `jarvis:cancel:research:{task}` | String | Khóa hủy (thời điểm yêu cầu), lưới an toàn nếu lỡ tin Pub/Sub | 30 giây |
| `jarvis:research:running` | ZSET | Các tác vụ nghiên cứu đang chạy, pod chủ gia hạn mỗi 10 giây | 30 giây sau lần gia hạn cuối |
| `jarvis:research:{task}` | Hash | Chủ của tác vụ và trạng thái cuối | 1 ngày |
| `jarvis:sbx:admit` | String | Khóa xếp chỗ sandbox (một quyết định một lúc) | 20 giây |
| `jarvis:sbx:reserved`, `jarvis:sbx:inuse` | ZSET | Sandbox đang được đặt chỗ, sandbox đang có lệnh chạy | 30 giây sau lần gia hạn cuối |

## 4. Các luồng

### Dừng một lượt chat ở pod khác
1. Request Dừng rơi vào pod B. Pod B tìm trong bộ nhớ của nó, không thấy lượt nào.
2. Pod B đọc khóa cuộc trò chuyện trong Redis, thấy chủ là pod A.
3. Pod B ghi khóa hủy rồi gửi tin vào kênh của pod A.
4. Pod A (đã đăng ký kênh lúc khởi động) nhận tin, hủy lượt chạy và dọn sandbox như khi Dừng ở chính nó. Luồng về trình duyệt kết thúc bằng `stopped` rồi `done`.
5. Nếu pod A lỡ tin (kết nối đứt thoáng qua), nó đọc khóa hủy trong vòng 3 giây và bỏ qua khóa nào cũ hơn lúc lượt chạy bắt đầu, để một lệnh Dừng cũ không hủy nhầm lượt mới.

### Nghiên cứu nền
Tác vụ ghi vào `jarvis:research:running` và hash của nó khi bắt đầu và cập nhật trạng thái cuối khi xong. Pod khác hỏi trạng thái thì đọc từ đó: đang chạy thì trả "đang chạy", đã xong thì trả đúng kết quả, và nếu chủ đã chết (không gia hạn nữa) thì coi như bị gián đoạn. Hủy từ pod khác đi theo đường Pub/Sub như trên. Trần `BACKGROUND_RESEARCH_MAX_RUNNING` tính trên tổng các pod.

### Sandbox
Số pod sandbox vẫn đếm từ SandboxClaim trên cluster. Ba thứ cluster không biết được chia sẻ qua Redis: khóa xếp chỗ (hai pod không cùng thấy chỗ cuối), đặt chỗ (sandbox vừa được phép tạo mà chưa hiện), và đang dùng (pod này không tạm dừng sandbox pod kia đang chạy lệnh). Cuộc trò chuyện đang chạy lấy từ chính các khóa cuộc trò chuyện.

### Cập nhật phiên bản, scale xuống
1. Kubernetes xóa pod; `preStop` chờ 5 giây để Ingress ngừng gửi request mới.
2. SIGTERM: uvicorn ngừng nhận kết nối mới và **chờ các luồng đang mở kết thúc**. Lượt chat vẫn chạy bình thường.
3. Sau đó ứng dụng chờ nốt các lượt không còn trình duyệt và các tác vụ nghiên cứu, tối đa `SHUTDOWN_DRAIN_SECONDS` (600 giây). Trong lúc chờ, lệnh Dừng vẫn tới được pod.
4. Thoát. Nếu quá `terminationGracePeriodSeconds` (900 giây) thì Kubernetes buộc tắt.

### Scale lên
Pod mới chỉ nhận các request mới. Lượt chạy ở các pod cũ không bị ảnh hưởng.

### Redis không dùng được
Mọi bước trên đều làm theo khả năng: lỗi Redis được ghi log (không quá 1 lần mỗi 30 giây) và mỗi pod quyết định theo những gì nó tự biết, tức hành vi của một pod. Với 2 pod thì khi đó Dừng chéo pod không hoạt động.

## 5. Cấu hình

- `REDIS_URL` (có sẵn): bắt buộc khi chạy hơn 1 pod.
- `SHUTDOWN_DRAIN_SECONDS` (mặc định 600): thời gian chờ lượt chạy và tác vụ nghiên cứu khi tắt.
- `jarvis-deploy/backend/base/backend.yaml`: `terminationGracePeriodSeconds: 900`, `preStop` 5 giây. Số `replicas` vẫn là 1.

## 5b. Số liệu cho autoscaler: `/metrics`

Cả backend và conversation-service đều trả `GET /metrics` (định dạng Prometheus) với ba con số cho từng nhóm route (`default`, và `chat` ở backend):

| Metric | Ý nghĩa |
|---|---|
| `jarvis_inflight_requests` | Số request đang xử lý ngay lúc này |
| `jarvis_inflight_limit` | Giới hạn mà từ đó request kế tiếp bị trả 503 (0 là không giới hạn) |
| `jarvis_shed_requests_total` | Số request đã bị từ chối bằng 503 kể từ khi tiến trình khởi động |

Đây là tín hiệu để scale thay cho CPU với dịch vụ chủ yếu chờ I/O: số request đang xử lý cho biết mức tải trước khi CPU kịp tăng, và khớp trực tiếp với cơ chế 503. Endpoint luôn trả lời, kể cả khi nhóm route đã đầy, và **chỉ trả cho người gọi bên trong cluster**: request đi qua Ingress có header chuyển tiếp nên nhận 404. **Đã cài Prometheus và KEDA (ngày 2026-11-24, cài tay, chưa dùng để scale thật):**
- Prometheus (`jarvis-deploy/monitoring/prometheus.yaml`, namespace `monitoring`): một pod, quét `/metrics` của các pod `backend` và `conversation-service` mỗi 10 giây, giữ 6 giờ, không có ổ đĩa. Chọn pod theo nhãn `app` nên không cần thêm chú thích vào Deployment.
- KEDA 2.21.1 (`jarvis-deploy/keda/values.yaml`, cài bằng Helm vào namespace `keda`). Cả hai cộng lại dùng khoảng 15m CPU và 110 MB RAM.
- Đã thử đầu cuối bằng một deployment giả (đã xóa): khi bắn tải thật, số request đang xử lý của conversation-service lên 13–14, KEDA đọc được và đưa deployment giả từ 1 lên 3 pod sau khoảng 20 giây; hết tải thì về 1 pod. So với HPA theo CPU (khoảng 2,5 phút) là nhanh hơn nhiều, nhưng con số 20 giây đo trên một lần thử.
- **conversation-service đã chuyển sang KEDA (2026-11-24, chỉ ở cluster test).** HPA theo CPU được thay bằng một ScaledObject (`jarvis-deploy/conversation-service/overlays/test/scaledobject.yaml`) với hai tín hiệu: tổng số request đang xử lý (ngưỡng 10 mỗi pod, khoảng 70% giới hạn 15) và tốc độ request bị từ chối (ngưỡng 2 mỗi giây mỗi pod); số pod là mức lớn hơn trong hai mức. Cluster staging chưa có KEDA nên vẫn giữ HPA theo CPU trong `base`. Đo trên cluster thật với 150 req/s từ 1 pod: quyết định thêm pod sau khoảng 15 giây, đủ 3 pod sẵn sàng sau khoảng 31 giây (HPA theo CPU trước đó mất khoảng 2,5 phút).
- **Backend cũng tự scale bằng KEDA (2026-11-24, chỉ ở cluster test)**, từ 1 đến 3 pod (`jarvis-deploy/backend/overlays/test/scaledobject.yaml`), với ba tín hiệu: request thường đang xử lý (12 mỗi pod, khoảng 70% giới hạn 17), lượt chat đang chạy (14 mỗi pod, giới hạn 20) và tốc độ bị từ chối (2 mỗi giây mỗi pod). Giảm pod chậm và nhẹ tay: sau 5 phút dưới ngưỡng, mỗi phút bớt một pod, và pod bị bớt chờ lượt chat của nó xong rồi mới tắt. Overlay test bỏ số `replicas` cố định khỏi Deployment để ArgoCD không đặt lại về 1. Đo với 150 req/s từ 1 pod: quyết định thêm pod sau khoảng 10 giây, đủ 3 pod sẵn sàng sau khoảng 26 giây; trước đó chỉ phục vụ 84–120 req/s và từ chối 28–62 req/s, sau đó phục vụ gần hết 150 req/s và hầu như không từ chối. **Đánh đổi đã chấp nhận:** pod chết đột ngột thì mất các lượt chat đang chạy trên nó.

## 6. Đã kiểm chứng

- Test tự động: 328 đạt khi không có Redis, 383 đạt khi có Redis thật (`REDIS_TEST_URL`).
- Trên cluster với 2 pod, chat thật:
  - Lượt chat chạy ở pod 1, Dừng gửi vào pod 2: trả `{"stopped": true}` trong 30 ms, luồng ở pod 1 kết thúc bằng `stopped` rồi `done`.
  - Xóa pod giữa lúc chạy một bài luận 3.000 từ: lượt chat vẫn xong bình thường 70 giây sau khi pod bị xóa (1.299 mẩu chữ, có `done`, không lỗi), rồi pod mới biến mất. Mặc định cũ chỉ cho 30 giây.
  - Xếp chỗ sandbox qua Redis: chuỗi lệnh đúng thiết kế.
- Chưa thử trực tiếp trên cluster: kiểm tra và hủy tác vụ nghiên cứu từ pod khác (chỉ có test tự động).

## 7. Giới hạn còn lại

- **Pod chết đột ngột thì mất lượt chat đang chạy trên nó**, và trình duyệt không nối lại được sau khi mất kết nối. Đây là giới hạn của việc không có hàng đợi.
- Khi hai pod cùng bắt đầu một tác vụ nghiên cứu đúng một lúc, tổng có thể vượt trần 1 tác vụ.
- Khóa xếp chỗ sandbox hết hạn sau 20 giây và không tự gia hạn: một quyết định chậm hơn thế có thể để pod khác chen vào.
- Chặn tải (503) vẫn tính theo từng pod.
- Redis trên cluster hiện 32 MB, không lưu đĩa. Lượng dữ liệu ở đây rất nhỏ nên đủ, nhưng Redis khởi động lại thì mất các mục đặt chỗ và khóa đang giữ (khóa tự lập lại nhờ heartbeat).

## 8. Phương án nặng hơn, đã cân nhắc và không làm

Tách pod web và pod worker, đưa lượt chat vào hàng đợi Redis Streams và đẩy từng mẩu chữ qua Redis. Cách này giữ được lượt chat khi pod chết và cho trình duyệt nối lại, nhưng phức tạp hơn nhiều (thêm hàng đợi, luồng sự kiện, chính sách chạy lại, Redis phải lớn hơn và bền hơn) và nằm ngoài yêu cầu scale. Đã từng viết một bản chạy được rồi gỡ bỏ ngày 2026-11-24. Chỉ cân nhắc lại nếu cần chịu được pod chết đột ngột hoặc cần trình duyệt nối lại.
