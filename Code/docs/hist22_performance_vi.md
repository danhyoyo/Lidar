# Hist22: kiểm chứng runtime và chi phí CPU

Ngày đo: 17/10/2026. Hist22 đúng hợp đồng số học đã kiểm thử, nhưng **chưa đạt mục tiêu encoding ≤2× Hist14 Numba**. Trên 20 frame KITTI thật, p50 gộp là **2,916 → 8,217 ms**, tương ứng **2,818× / +5,302 ms**. Chưa chạy full paired training; detector AP và ΔAP vẫn **pending**.

Người dùng đã chấp nhận mức **8,22 ms/frame** và yêu cầu dừng tối ưu tốc độ thêm. Mục tiêu ≤2× ở plan là mục tiêu tối ưu ban đầu, không còn là điều kiện chặn phát hành bản này.

## Nguồn cấu hình và điều kiện đo

- Cặp thật lấy từ notebook Standard và master runtime: [manifest](../../artifacts/hist22_runtime_20261017/ablation_pair/manifest.json), [baseline snapshot](../../artifacts/hist22_runtime_20261017/ablation_pair/baseline.source.json), [lệnh train/evaluate](../../artifacts/hist22_runtime_20261017/ablation_pair/commands.sh). Giữ MobilePixorNeXt, grouped head Car / Pedestrian+Cyclist, OGA/IQA, Focal C4, block SimAM C4/C5; không thay kiến trúc để lấy số nhanh hơn.
- Research protocol giữ seed 42, 50 epoch / warmup 4, batch 16, BF16, 6 worker, target backend Numba. Hai config chỉ khác ba leaf: `data.bev_encoding.name`, `data.bev_encoding.out_channels`, `experiment.name`. Paired random initialization v1 copy mọi tensor chung và zero tám lát input mới.
- CPU: AMD Ryzen 5 7640HS; Linux x86_64; Python 3.14.4, NumPy 2.4.6, Numba 0.67.0, PyTorch 2.11.0. `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1`; Torch intra/inter-op đều 1 trong pipeline. Không có CUDA khả dụng. Affinity, phiên bản và toàn bộ hash nguồn trong JSON.
- CLI mới: [benchmark_hist22_pipeline.py](../tools/benchmarks/benchmark_hist22_pipeline.py). Preflight xác minh canonical config digest, schema hash, run identity, đúng hai vai A/B, đúng ba leaf khác nhau và strict paired-init policy trước khi đọc dataset hoặc tạo artifact. Không ghi đè output đã có.

### Source khi đo và khi push branch

Các artifact v2 ghi hash của **worktree lúc đo**, bao gồm những thay đổi AP progress logging và CPU NMS đã có sẵn trước Hist22. Các thay đổi có sẵn này được giữ riêng trong staging của người dùng, ngoài commit Hist22. Vì vậy bảng pipeline dưới đây mô tả đúng worktree có hash trong artifact, không phải cam kết latency NMS của một checkout chỉ có commit Hist22. Encoder và phép đo encoding 20 frame không phụ thuộc các thay đổi AP/NMS đó; backend Hist22 được giữ nguyên. Khi tái lập toàn pipeline, cần đối chiếu source hash hoặc đo lại trên checkout đang dùng.

CLI encoding benchmark hiện phục vụ cặp đã đo với `density_norm=32`, `intensity_scale=1`; helper tạo schema theo các giá trị mặc định này. Nếu dùng cặp có normalization khác, cần truyền đúng options vào helper trước khi dùng benchmark để so sánh cặp đó. Runtime encoder vẫn hỗ trợ và kiểm tra các options theo schema.

## Encoding thật, direct CHW

[Artifact chính thức](../../artifacts/hist22_runtime_20261017/encoding_final_20frames_v2.json) chứa tất cả raw samples, thứ tự chạy, hash `.bin`, split và source. File `encoding_final_20frames.json` cũ được thay thế bởi bản **v2** sau review preflight; không dùng làm số cuối.

20 frame: showcase `000804 004425 004567 001801`; random train seed42 `005264 000892 000178 006076 002244 001994 001819 001124`; random validation `006046 000854 005545 006074 007283 004487 000722 004843`. Số điểm/frame 114.051–126.472. Cloud được preload trước đo. Mỗi frame/encoder có 3 warmup + 30 mẫu; shuffle thứ tự frame và thứ tự A/B trong từng cặp, tổng 600 mẫu/encoder.

| Encoding | p50 gộp (ms) | p95 gộp (ms) | Mean (ms) |
|---|---:|---:|---:|
| Hist14 Numba | 2,9159 | 3,5142 | 2,9622 |
| Hist22 Numba | 8,2175 | 11,3435 | 8,6085 |

Tỷ số p95 gộp 3,228×. Đây là percentile trên các raw sample được gộp với số mẫu bằng nhau/frame; không phải percentile của tỷ số từng lần gọi. Tỷ số p50 từng frame nằm trong 2,530–3,327×.

| Showcase | Hist14 p50/p95 (ms) | Hist22 p50/p95 (ms) | p50 ratio | +p50 (ms) |
|---|---:|---:|---:|---:|
| 000804 | 3,1754 / 4,1036 | 10,1525 / 10,4522 | 3,197× | 6,9771 |
| 004425 | 3,0082 / 3,1338 | 8,6774 / 8,8797 | 2,885× | 5,6692 |
| 004567 | 3,2422 / 3,4414 | 10,6051 / 11,3348 | 3,271× | 7,3629 |
| 001801 | 2,4454 / 2,6170 | 6,1864 / 6,4272 | 2,530× | 3,7410 |

Đo chỉ bao gồm cloud float32 đã có trong RAM → output float32 contiguous CHW độc lập, có allocation/zeroing, không có I/O, HWC transpose/copy, target, model hoặc augmentation; deallocation nằm ngoài clock. Hai backend dùng cùng hình học 800×704. Bốn showcase được so với NumPy Hist22 reference ngoài steady-state timing: max absolute error ≤2,981e-7; base tolerance `atol=1e-6, rtol=1e-5`, geometry `atol=2e-6, rtol=1e-5`. Reference single calls 91,96–107,38 ms chỉ phục vụ parity, không phải benchmark warmed NumPy.

First invocation: Hist14 **247,78 ms**, Hist22 **17,59 ms**. Cache `/tmp/hist22_task4_final_numba` đã tồn tại từ lần chạy trước; số này gồm dependency import / compile hoặc disk-cache loading và first execution, **không phải isolated compiler time hay cold empty-cache benchmark**. Hist14 gọi trước và trả phần import Numba dùng chung; không so trực tiếp hai first-call để kết luận tốc độ startup.

Các thử nghiệm tối ưu trước gate cuối được ghi trong [plan](plans/2026-10-17-hist22-runtime-implementation.md): dense integral tables, rolling/strip sums, sparse horizontal prefix, output allocation, analytic eigensolver và unrolled ten-stat arithmetic. Bản giữ lại fuse Hist14/moments, rolling vertical windows, sparse prefix theo occupied columns, analytic symmetric eigenvalues có exact near-degenerate fallback, cấp output từ NumPy. Tuple-helper unrolling thay đổi <1% trong benchmark interleaved, coi là noise. [Bằng chứng bốn frame](../../artifacts/hist22_runtime_20261017/hist22_unroll_bench.log), [tuple before/after](../../artifacts/hist22_runtime_20261017/hist22_tuple_real_benchmark.json). Không cap neighbor, downsample, fastmath hoặc đổi feature để đạt mục tiêu. Gate 20 frame đo lại cả hai backend trên cùng máy; không dùng wall time của prototype khác làm speedup claim.

## Full-grid pipeline và training smoke

[Artifact full-grid chính thức](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.json) đã hoàn tất trên CPU FP32, batch1, H×W=800×704; **initial output parity đạt max absolute error 0** giữa hai model paired, trong tolerance `atol=1e-6, rtol=1e-5`. Artifact chứa raw samples, lịch chạy, full config/count audit và hash cloud/label/calibration.

Inference dùng validation frames `004425`, `000804`: 3 warmup + 6 mẫu/model, cùng frame trong từng vòng, random thứ tự hai model. Sample cho optimizer là **training frame `000000`** từ split gốc, tách khỏi hai frame validation. [Train manifest](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/train.txt) và [validation manifest](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/val.txt) disjoint. Subset có cloud symlink, label chuyển bởi `prepare_kitti.convert_labels`; dùng `Dataset(task="validation")` cho cả việc chuẩn bị inference và training batch để không tạo/đọc train-only GT database và không augmentation. `000000` có một Pedestrian và tám positive regression target cells; đây là smoke trên một sample, không đại diện phân bố training.

Chỉ config smoke riêng nhận các override chung sau: device CPU; precision FP32; train/val batch1; workers0; epoch1/warmup0; selector `primary=loss` thay AP (bỏ `ap_every`/`metric_mode`); data location trỏ subset và train/val manifests riêng. Model, loss, attention, optimizer, clipping, geometry và Numba target backend giữ nguyên. Cặp research 50/4, batch16/BF16/6 worker vẫn giữ nguyên trên đĩa. Xem [Hist14 smoke config](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/hist14.smoke.config.json), [Hist22 smoke config](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/hist22.smoke.config.json) và `models.*.smoke_overrides` trong JSON.

| Stage CPU batch1 | Hist14 p50 / p95 (ms) | Hist22 p50 / p95 (ms) |
|---|---:|---:|
| I/O cloud + cached labels | 0,308 / 0,339 | 0,288 / 0,334 |
| Encoding CHW qua Dataset | 3,394 / 14,056 | 10,179 / 12,084 |
| Targets + collate, gồm copy BEV vào batch | 5,118 / 19,081 | 28,307 / 42,759 |
| Transfer traversal `.to(cpu)` | 0,041 / 0,047 | 0,045 / 0,071 |
| Eval forward | 337,098 / 354,966 | 368,150 / 377,151 |
| Decode + rotated NMS | 5.747,829 / 7.435,669 | 5.749,438 / 7.500,926 |
| Encoding → detections | 6.099,078 / 7.768,070 | 6.131,790 / 7.884,451 |
| Serial I/O → targets → detections | 6.104,010 / 7.779,623 | 6.160,424 / 7.927,522 |

Totals là tổng các stage được clock riêng trong từng sample; không cộng percentile các dòng để dựng total. `Encoding → detections` loại I/O/targets, còn serial total có cả hai. I/O đọc cloud đã được OS cache, copy label đã cache trong constructor; loại label/calibration conversion. Targets+collate có copy tensor input nên không diễn giải chênh lệch là chi phí target generation riêng. `.to(cpu)` không đo H2D. Sáu mẫu/model trên hai frame chỉ là stage smoke; dùng benchmark 600 mẫu/encoder ở trên để kết luận encoding latency.

**Random-weight decode rất dày:** cả 6 mẫu của mỗi model đều trả **500 detections sau NMS**, chạm `max_detections=500`; cap này áp dụng sau NMS, không phải giới hạn candidate trước NMS. Với threshold gốc 0,05 và NMS threshold 0,1, CPU decode/NMS mất khoảng **5,75 giây p50** và chi phối tổng. Đây là kết quả thực tế của model chưa train; không phải empty-output shortcut, cũng không phải latency của detector đã train. Không suy kết luận end-to-end overhead nhỏ từ việc NMS random đang áp đảo. Trường text `decode_limit` trong JSON là mô tả chung có nhắc “empty/few”; các mảng số `models.*.detections` mới mô tả lần chạy này và đều bằng 500.

Training smoke chạy **3 optimizer steps/model** trên cùng prepared batch `000000`, từ paired T0; dùng criterion OGA/IQA thật, backward, grad clipping và AdamW thật. Cả ba loss mỗi model hữu hạn; loss đầu cùng bằng 8.043,177246. Tám lát stem Hist22 mới đều có gradient hữu hạn khác zero trong mỗi step được kiểm tra, và cả tám đã thay đổi sau optimizer steps. Không thay loss bằng sum output hoặc synthetic surrogate.

| Prepared-batch compute smoke | Hist14 p50 / p95 (ms) | Hist22 p50 / p95 (ms) |
|---|---:|---:|
| Training forward | 418,572 / 669,097 | 427,292 / 427,591 |
| Criterion | 5,839 / 5,972 | 5,608 / 5,701 |
| Backward | 1.028,712 / 1.100,777 | 1.048,777 / 1.053,224 |
| Clip + optimizer step | 5,669 / 7,462 | 6,170 / 6,879 |
| Whole compute step | 1.458,897 / 1.783,353 | 1.486,955 / 1.492,223 |

Mean compute-step 1.569,330 / 1.488,508 ms tương ứng 0,637 / 0,672 sample/s. **Không kết luận Hist22 train nhanh hơn**: chỉ ba step, có first-step overhead, không training warmup, Hist14 train trước Hist22, hardware/cache state và trọng số sau update khác nhau. Scope này gồm forward+criterion+backward+clip+optimizer trên batch đã chuẩn bị, loại I/O/encoding/targets; không phải throughput DataLoader hay training epoch. Các first-call preparation/forward timings giữ riêng trong artifact, không trộn vào inference steady-state.

Mỗi model được lưu full checkpoint gồm model/criterion/optimizer/scheduler/scaler/RNG/identity/provenance, đọc lại từ file vào các object được dựng mới và restore bằng API train thông thường. Cả hai có `restore_result.mode="resume"`, epoch1, không backend change, và **output sau restore khớp bitwise (max absolute error 0)**. Checkpoints: [Hist14](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/hist14.smoke.checkpoint.pth), [Hist22](../../artifacts/hist22_runtime_20261017/pipeline_final_fullgrid_v2.artifacts/hist22.smoke.checkpoint.pth); SHA-256 và kích thước nằm ở `models.*.checkpoint`.

| Count từ model thật, full grid | Hist14 | Hist22 | Δ |
|---|---:|---:|---:|
| Backbone gồm neck (parameters) | 660.528 | 662.832 | +2.304 |
| Grouped heads (parameters) | 186.161 | 186.161 | 0 |
| Total detector (parameters) | 846.689 | 848.993 | +2.304 |
| Criterion chỉ dùng train (parameters) | 12 | 12 | 0 |
| Conv2d MAC/frame | 14,0428288 G | 14,3672320 G | +0,3244032 G |

Đếm bằng `profile_detector.py` với instantiated modules và Conv2d output-shape hooks trên meta device. MAC này loại attention matmul, Conv1d, normalization, activation, interpolation, elementwise và preprocessing; không phải tổng FLOP hoặc suy luận latency từ số parameter.

## Bộ nhớ và giới hạn diễn giải

Payload input FP32/frame: Hist14 **31.539.200 byte = 30,078 MiB**, Hist22 **49.561.600 byte = 47,266 MiB**; +17,188 MiB / +57,14%. Đây chỉ là input, không phải peak RAM.

Encoding process native peak RSS **488,83 MiB**, đo bằng Linux `ru_maxrss`; RSS hiện tại sau reference 245,49 MiB. Peak này gồm imports, preload cả 20 cloud, cả hai encoder, JIT/cache, scratch float64 và NumPy reference; không quy peak này riêng cho Hist14 hay Hist22. Không dùng `tracemalloc` để bỏ qua native allocation.

Pipeline chạy trong process khác, native lifetime peak RSS **2.152,82 MiB**; current RSS tại cuối report **2.084,83 MiB**. Scope gồm imports/JIT, hai model paired cùng tồn tại, prepared inputs, metadata profiler, autograd, optimizer và các bản copy trong checkpoint restore; không phải peak training riêng lẻ hoặc peak quy riêng cho từng encoder. Không lấy chênh lệch cumulative peak giữa hai lượt train để báo ΔRAM Hist22. Scratch native được tính vào RSS, nhưng chưa tách từng phase/allocator.

Không suy số đo CPU batch1 sang GPU batch16/BF16/6 worker. Chưa đo worker/prefetch memory, GPU VRAM, H2D, full DataLoader throughput hoặc trained-model NMS workload. Thư mục `Code/data/kitti/raw` và `processed` của cặp research còn trống; subset smoke riêng dùng symlink cloud và label convert bằng code hiện có, không thay dataset người dùng hoặc tạo GT database toàn bộ.

## Tái lập

Chạy từ `Code/`; đặt output mới vì CLI từ chối ghi đè:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR=/tmp/hist22_task4_final_numba MPLCONFIGDIR=/tmp/hist22_task4_mpl
export PYTHONPATH=.:detector:detector/core/datasets:tools/kitti_training_pipeline
PY=/home/duyennh/miniconda3/envs/AI_env/bin/python
$PY tools/benchmarks/benchmark_hist22_pipeline.py --mode encoding \
  --pair-manifest ../artifacts/hist22_runtime_20261017/ablation_pair/manifest.json \
  --warmup 3 --iterations 30 --per-split 8 --reference-count 4 \
  --output ../artifacts/hist22_runtime_20261017/encoding_reproduce.json
$PY tools/benchmarks/benchmark_hist22_pipeline.py --mode pipeline \
  --pair-manifest ../artifacts/hist22_runtime_20261017/ablation_pair/manifest.json \
  --warmup 3 --pipeline-iterations 6 --training-iterations 3 \
  --output ../artifacts/hist22_runtime_20261017/pipeline_reproduce.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_hist22_pipeline_benchmark.py -q
```

Tests mới: RED missing tool ở collection; GREEN 4 test ban đầu. Sau review, tampered schema test RED trước preflight fix; gate cuối **11 passed / 2,23 s**, gồm deterministic stratified selection, actual encoding/reference, CLI validation/output preservation, bảy tamper cases và grouped fixture thật qua label conversion → targets → criterion/backward/optimizer → decode/NMS → full checkpoint restore. Full-grid run là kiểm chứng manual độc lập với fixture nhỏ.

Source freshness đã pass: map hash nguồn ở đầu và cuối mỗi JSON khớp, và hai artifact dùng cùng source map. Map bao gồm schema, encoder, dataset/targets, models/losses, postprocess, train/paired helper/ablation, profiler, benchmark và test. Final SHA-256:

- `hist22_backend.py`: `4c7411f14191ce0f854de0a5c8283e2bd737573281b833d7c634ed0b8e435a33`.
- Benchmark tool: `c1fa25bb701d1f86ecd88b21bce2d1512c41e1e9e1dd23e2aab3a6be68386545`.
- Benchmark tests: `9895c99e648225b8b17c7360e22e5e524e358833e0f38790a1b2e4976d180ff1`.

Full `tests/` và independent final review do parent thực hiện riêng sau isolated timing; kết quả chưa được tự điền trong report này. Detector AP/ΔAP, full paired training, GPU latency/VRAM/H2D và DataLoader throughput/memory vẫn **pending**. Smoke chỉ xác minh pipeline thật chạy được và checkpoint khôi phục đúng.
