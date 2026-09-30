# Kế hoạch sửa GT-sampler sau review

Ngày: 2026-09-30. Trạng thái: kế hoạch triển khai, chưa sửa mã nguồn.

### Goal

Sửa các lỗi đã tái hiện trong GT sampling: xóa sai điểm do shadow masking, từ chối sai placement do line-of-sight, giữ nhãn khi augmentation xóa sạch điểm, cờ cấu hình không độc lập và các lỗi API/database. Hoàn thành bằng kiểm thử hồi quy và kiểm thử tích hợp; đánh giá chất lượng trên KITTI là bước riêng khi có dữ liệu.

### Assumptions

- Đối chiếu mã nguồn tại `81e3046`. Trước khi triển khai phải kiểm tra diff mới và giữ các thay đổi của người dùng.
- Kết quả review trước: 28 test đạt trong sandbox; test multiworker đạt khi chạy lại ngoài sandbox. Đây là baseline lịch sử, chưa phải kết quả xác minh bản sửa.
- Luồng chính dùng point cloud float32 `(N, 4)` và box `(M, 8)` theo `[class, h, w, l, x, y, z_bottom, yaw]`. API sampler tiếp tục hỗ trợ scene box `(M, 7)`, kể cả `(0, 7)`; database do builder tạo vẫn dùng box 8 cột. Không mở rộng hỗ trợ database 7 cột trong đợt này.
- Giữ nguyên quy ước tọa độ, class mapping, kiến trúc model/loss và format pickle hiện tại. Chưa cần rebuild database chỉ vì thay đổi phép che khuất.
- `sample_counts` tiếp tục là số đối tượng chèn thêm tối đa; `{}` có nghĩa không chèn. Giữ số lượng, thứ tự class và phân bố đề xuất hiện tại để giảm số biến thay đổi trong thí nghiệm.
- Phương án chọn: phép giao tia–oriented box bằng NumPy, dùng chung cho shadow và line-of-sight. Sửa khoảng góc đơn thuần không xử lý hết các phản ví dụ; dựng mesh/ray tracing bề mặt thực nằm ngoài phạm vi này. Giao tia–box đúng với hình học box, không đồng nghĩa mô phỏng chính xác hình dạng vật thể hay toàn bộ sensor.
- Giữ tham số `enable_physics` làm preset tương thích cho caller cũ. Thêm các cờ keyword độc lập với giá trị mặc định `None`: cờ được cung cấp rõ sẽ ghi đè preset. Dataset truyền các cờ thực tế thay vì ánh xạ shadow thành toàn bộ physics.
- Box collision và xóa điểm bên trong box mới là thao tác cơ bản của copy-paste, luôn thực hiện khi chèn. Các cờ ground/static/line-of-sight/shadow/density/radiometric chỉ điều khiển thành phần tương ứng. Cần ghi rõ thay đổi hành vi này cho caller cũ dùng `enable_physics=False`.
- Ground validation bật: point cloud rỗng hoặc không có ground hỗ trợ phải từ chối placement. Scene rỗng nhãn nhưng có ground hợp lệ vẫn được chèn. Fallback `snap_box_to_ground` chỉ dùng khi ground validation được tắt rõ ràng.
- Chính sách visibility được đề xuất: từ chối placement gây mất quá nhiều điểm của box đã tồn tại; không tự xóa nhãn cũ. Với baseline `n0 > 0`, yêu cầu `n_after >= min(n0, 5)` và `n_after / n0 >= 0.5`. Hai ngưỡng là cấu hình (`min_visible_points=5`, `min_visible_ratio=0.5`), chưa phải giá trị được tối ưu trên KITTI. Box vốn không có điểm được giữ nguyên và không dùng phép chia cho 0.
- Baseline visibility của box gốc lấy trước sampling; của box mới lấy khi được chèn sau subsampling. So sánh các lượt sau với baseline cố định để tránh mất điểm tích lũy vượt ngưỡng.
- Phân bố ground theo footprint, fit mặt phẳng, các ngưỡng mật độ và bảo toàn góc nhìn khi đổi yaw là cải tiến cần dữ liệu thực. Đợt sửa này xử lý fallback sai và làm rõ heuristic; chưa tuyên bố giải quyết toàn bộ tính chân thực vật lý.
- Chỉ lập kế hoạch trong lượt hiện tại. Các tên test mới dưới đây là yêu cầu cho bước triển khai, chưa tồn tại trong repository.

### Plan

Chạy các lệnh từ thư mục gốc repository. Dùng `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` để tránh plugin ROS không liên quan đang thiếu dependency; giới hạn luồng CPU cho test tích hợp. Mỗi bước có thay đổi hành vi thực hiện theo thứ tự: viết test → xác nhận thất bại đúng nguyên nhân → sửa → xác nhận đạt. Chia commit theo nhóm lỗi để có thể revert riêng.

1. Ghi nhận baseline và chuẩn bị fixture xác định được placement.
   - Files: `tests/test_gt_sampler.py`, `tests/test_physics_aug.py`, `tests/test_physics_augmentation_e2e.py`.
   - Change: tạo fixture ground phẳng, box có yaw khác nhau, đối tượng cũ với điểm nhận diện được; điều khiển candidate pose bằng test fixture/monkeypatch tối thiểu. Dùng seed khi cần, không phụ thuộc chuỗi random của toàn suite; không mock phép hình học đang kiểm thử.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_gt_sampler.py tests/test_physics_aug.py tests/test_collision_ground.py tests/test_gt_database_builder.py tests/test_physics_augmentation_e2e.py`.
   - Verify: ghi baseline; nếu sandbox chặn socket multiprocessing, chạy lại riêng `tests/test_physics_augmentation_e2e.py::test_pcu_dataloader_multiworker_batching` với quyền phù hợp, không sửa test để né lỗi môi trường.

2. Thêm primitive giao tia với box có hướng và kiểm thử toán học.
   - Files: mới `detector/core/datasets/utils_1/box_geometry.py`, mới `tests/test_box_geometry.py`.
   - Change: hàm `ray_box_intervals(points, box)` trả mask giao, `t_enter`, `t_exit` cho tia `O + t * P`, origin sensor `O=(0,0,0)`, điểm đầu vào ở `t=1`. Chuyển origin và hướng tia về local box; dùng slab intersection với X/Y đối xứng và Z thuộc `[0,h]`.
   - Change: xử lý hướng song song trục bằng mask thay vì chia cho 0, tia zero-length, giao phía sau sensor, tiếp xúc biên với tolerance có đơn vị rõ ràng, cloud rỗng và box 7/8 cột. Kiểm tra kích thước dương và dữ liệu hữu hạn; không dùng clipping mẫu số làm lệch dấu hướng tia.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_box_geometry.py`. Các expected interval được tính từ box đơn giản bằng tay; thêm bất biến khi cùng quay điểm và box quanh sensor. Không dùng chính helper mới làm oracle.

3. Thay shadow approximation bằng giao tia–box.
   - Files: `detector/core/datasets/utils_1/physics_aug.py`, `tests/test_physics_aug.py`.
   - Change: xác định shadow với giao ở phía trước và điểm ở sau mặt thoát (`t_exit < 1` có tolerance); điểm nằm trong box được xử lý riêng ở sampler. Giữ signature `mask_shadow_points` và thứ tự/feature của các điểm còn lại; cho phép sampler lấy mask để kiểm tra trước khi áp dụng.
   - Change: regression box `[h=2,w=2,l=4,x=10,y=0,z=-1,yaw=0]`: giữ `(20,3,0)`, xóa `(12.1,0,0)`; thêm yaw 0/π/2, điểm trên/dưới box, ở trước box và các giá trị gần biên.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_box_geometry.py tests/test_physics_aug.py -k 'ray or shadow'`.

4. Sửa line-of-sight bằng cùng primitive.
   - Files: `detector/core/datasets/utils_1/physics_aug.py`, `tests/test_physics_aug.py`.
   - Change: một điểm foreground chỉ có thể cản hướng nhìn tới box nếu tia qua nó giao box phía sau điểm (`t_enter > 1`); giữ các điều kiện loại ground và ngưỡng số blocking point như heuristic có tài liệu. Nếu giữ khoảng đệm, áp theo khoảng cách tới mặt vào box, không theo tâm box.
   - Change: regression 5 điểm `(x≈15,y=0,z=5)` không chặn xe ở `(30,0,-1.6)` cao 1.5 m. Kiểm tra wall thật vẫn bị phát hiện, điểm sau box không tính blocker, yaw được xét và điểm ngoài góc nhìn không cản.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_physics_aug.py -k 'line_of_sight'`.

5. Tách quá trình thử placement khỏi thao tác thay đổi scene.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tests/test_gt_sampler.py`.
   - Change: trong vòng thử pose, tính candidate point cloud và mask điểm sẽ bị xóa ở dạng tạm. Chỉ cập nhật `cur_points`, boxes, metadata và bộ đếm inserted sau khi tất cả điều kiện chấp nhận đã đạt. Placement bị từ chối tiếp tục thử trong giới hạn hiện tại.
   - Change: bảo đảm không làm thay đổi input arrays hay mẫu database; loại ứng viên không còn điểm sau xử lý; regression placement bị từ chối không xóa nền hoặc để sót box/metadata.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_sampler.py -k 'atomic or rejected or mutation or empty_candidate'`.

6. Bảo vệ visibility của box gốc.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tests/test_gt_sampler.py`.
   - Change: lưu membership điểm và baseline count theo box trước sampling; dùng candidate deletion mask để tính mức mất điểm trước khi chấp nhận. Áp hai ngưỡng trong Assumptions; không thay đổi nhãn scene gốc.
   - Change: test xe ở 20 m có 30 điểm, placement tại 10 m làm mất hết điểm phải bị từ chối; test mất ít điểm được chấp nhận, ngưỡng biên, box vốn có 0 hoặc dưới 5 điểm. Membership có thể nhiều box cùng chứa một điểm; không dùng một class-ID duy nhất để thay thế membership.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_sampler.py -k 'visibility or original_label'`.

7. Bảo vệ đối tượng chèn trước và cập nhật metadata theo scene cuối.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tests/test_gt_sampler.py`, `tools/visualization/visualize_pcu_aug.py`, `tests/test_visualization.py`.
   - Change: áp visibility cho cả đối tượng đã chèn; so với baseline cố định, không so riêng từng lượt. Theo dõi source identity của điểm chèn để `inserted_points` phản ánh điểm còn tồn tại ở output cuối, không phải bản copy trước các lần masking tiếp theo.
   - Change: regression nhiều lần che một phần không vượt tổng mức mất cho phép; metadata và visualization không vẽ lại điểm đã bị xóa. Không thêm tracking tốn kém chỉ phục vụ metadata khi `return_metadata=False`, ngoài dữ liệu thực sự cần cho visibility.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q tests/test_gt_sampler.py tests/test_visualization.py`.

8. Tách các cờ trong sampler và độc lập hóa thao tác copy-paste cơ bản.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tests/test_gt_sampler.py`.
   - Change: thêm cờ keyword `enable_ground_validation`, `enable_static_collision`, `enable_line_of_sight`, `enable_shadow_masking`, `enable_density_subsample`, `enable_radiometric_calibration`; giá trị explicit ghi đè preset `enable_physics`. Validate kiểu bool, không coi chuỗi `"false"` là False.
   - Change: luôn kiểm tra box collision và xóa điểm trong volume khi chèn; shadow chỉ điều khiển điểm phía sau. Test tác động quan sát được: density lên số điểm, radiometric lên intensity, shadow lên điểm phía sau, và các placement checks lên quyết định chấp nhận.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_sampler.py -k 'flags or physics or interior or preset'`. Bao phủ tổ hợp ba cờ augmentation bằng parameterization, không chỉ assert thuộc tính constructor.

9. Truyền config đúng và báo lỗi database ở training.
   - Files: `detector/core/datasets/dataset.py`, `tests/test_physics_augmentation_e2e.py`.
   - Change: Dataset truyền từng cờ và visibility thresholds. Khởi tạo/load sampler chỉ khi `task='train'` và GT sampling bật; val/test không cần database augmentation. Nếu train đã bật sampling nhưng đường dẫn thiếu, trỏ tới directory hoặc không đọc được, báo lỗi có đường dẫn.
   - Change: regression `shadow=true,density=false,radiometric=false` và tổ hợp ngược; kiểm tra output thực ở integration. Missing database không được âm thầm biến thành `None`; legacy augmentation vẫn hoạt động.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q tests/test_physics_augmentation_e2e.py -k 'flags or database or legacy or val or test_mode'`.

10. Loại fallback ground không có bằng chứng và sửa fixture tương ứng.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `detector/core/datasets/utils_1/collision_ground.py`, `tests/test_gt_sampler.py`, `tests/test_collision_ground.py`, `tests/test_physics_augmentation_e2e.py`.
   - Change: bỏ quyết định ground validation dựa trên `ptp > 2`; khi bật ground validation thì luôn kiểm tra support. Test scene không có điểm bị từ chối, scene không có nhãn nhưng có ground vẫn chèn được; fallback chỉ xảy ra khi tắt kiểm tra rõ ràng.
   - Change: fixture test placement phải có ground thật và canonical points `z∈[0,h]`, không dùng toàn điểm zero hoặc Z âm làm đối tượng hợp lệ. Ghi rõ kiểm tra hiện tại dựa trên neighborhood, chưa phải chứng minh hỗ trợ toàn footprint; giữ regression mô tả giới hạn này để không diễn giải quá mức.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 python -m pytest -q tests/test_gt_sampler.py tests/test_collision_ground.py tests/test_physics_augmentation_e2e.py -k 'ground or empty or support or pcu_mode'`.

11. Sửa contract input, empty scene và no-op.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tests/test_gt_sampler.py`, `tools/visualization/visualize_pcu_aug.py`, `tests/test_visualization.py`.
   - Change: lấy số cột từ shape, kể cả khi không có box; mặc định counts chỉ khi `is None`; `p=0` luôn no-op, `p=1` luôn thử. Validate `p` hữu hạn trong `[0,1]`, count nguyên không âm, shape `(N,4)`/`(M,7|8)`, giá trị finite, kích thước box dương; báo lỗi sớm thay vì để vstack/geometry lỗi sâu.
   - Change: mọi nhánh trả metadata cùng schema, gồm `final_boxes` và `num_inserted=0`; giữ dtype/shape nhất quán. Visualization cũng phân biệt `{}` và `None` để không tự khôi phục counts mặc định.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q tests/test_gt_sampler.py tests/test_visualization.py -k 'schema or empty or probability or counts or metadata or invalid'`.

12. Kiểm tra nội dung database và sửa builder output.
   - Files: `detector/core/datasets/utils_1/gt_sampler.py`, `tools/dataset_converter/create_gt_database.py`, `tests/test_gt_sampler.py`, `tests/test_gt_database_builder.py`.
   - Change: validate cấu trúc database khi load: class được yêu cầu với count>0 phải có mẫu, box8/points4 hữu hạn, dimension dương, range nguồn dương, points không rỗng, `num_points` khớp. Lỗi nêu class và sample index; không quét lại toàn database mỗi `__call__`. Counts rỗng/toàn 0 không đòi hỏi mẫu theo class.
   - Change: tạo parent bằng `Path(output_file).parent.mkdir(...)` để chấp nhận filename đơn; khi tất cả frame thiếu hoặc không trích được mẫu thì builder báo lỗi có thống kê thay vì xuất database rỗng như thành công. Giữ hỗ trợ hai layout và manifest `id;kitti`.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_gt_database_builder.py tests/test_gt_sampler.py -k 'database or builder or layout or manifest'`. Test bare filename dùng `monkeypatch.chdir(tmp_path)` để không ghi artifact vào repo.

13. Đồng bộ config và tài liệu về hành vi thực tế.
   - Files: `configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json`, `configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json`, `tools/kitti_training_pipeline/README.md`, `detector/core/datasets/utils_1/gt_sampler.py`, `detector/core/datasets/utils_1/physics_aug.py`, mới `tests/test_gt_sampler_config.py`.
   - Change: ghi explicit các physics flags và visibility thresholds; tài liệu precedence với `enable_physics`. Sửa comment 14 lượt corridor/6 lượt rectangle, phân tầng theo X, heuristic tối thiểu 5 điểm và ý nghĩa counts. Bổ sung note rằng thiết kế cũ không phải bằng chứng về latency hay độ chính xác vật lý.
   - Change: test cả hai config thực qua Dataset và xác nhận effective settings; mỗi cờ tắt có tác động riêng. Không đổi counts, class order, gamma hay ngưỡng subsampling mặc định trong commit sửa config.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 python -m pytest -q tests/test_gt_sampler_config.py tests/test_physics_augmentation_e2e.py -k 'config or flags or legacy'`.

14. Kiểm thử hồi quy toàn luồng và review diff.
   - Files: tất cả file đã đổi ở bước 1–13; kiểm tra caller trong `tools/visualization/visualize_pcu_aug.py` và `3D_Lidar_Object_Detection_Notebook_standard.ipynb`.
   - Change: kiểm tra builder → canonical object → sampler → geometric augmentation → BEV → loss/backward. Bổ sung round-trip yaw khác 0, sampler không thay đổi database/input, metadata không có điểm ma, loss và gradient hữu hạn. Chỉ sửa notebook nếu API mới thực sự làm caller hiện tại hỏng.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg python -m pytest -q tests/test_box_geometry.py tests/test_gt_sampler.py tests/test_gt_sampler_config.py tests/test_gt_database_builder.py tests/test_collision_ground.py tests/test_physics_aug.py tests/test_physics_augmentation_e2e.py tests/test_visualization.py tests/test_standard_training_notebook.py tests/test_regressions.py`.
   - Verify: `git diff --check`; review diff theo correctness, test assertion và chi phí vòng lặp. Nếu test mở rộng lỗi từ môi trường hoặc baseline, ghi nguyên nhân cụ thể; không tính chúng là đạt và không sửa ngoài phạm vi để làm xanh suite.

15. Tạo công cụ đo sampler khi có dữ liệu thực.
   - Files: mới `tools/visualization/audit_gt_sampler.py`, mới `tests/test_gt_sampler_audit.py`, `detector/core/datasets/utils_1/gt_sampler.py`, `tools/kitti_training_pipeline/README.md`.
   - Change: bổ sung diagnostics tùy chọn trong metadata: attempted/accepted theo class, số bị từ chối theo lý do đầu tiên, số điểm xóa bên trong/shadow không đếm trùng, count trước/sau của từng box, X và range của mẫu đã chèn. Counter cho placement bị từ chối không được cộng vào số điểm thực sự đã xóa.
   - Change: audit CLI nhận config, manifest frame và seed; dùng cấu hình giống train nhưng đo riêng sampler trước geometric jitter. Xuất JSON, ví dụ hình và latency p50/p95 sau warm-up, có điểm/frame và cấu hình máy; đo bản baseline và bản sửa trong checkout riêng, cùng frame/seed. Khác nhánh random sau sửa có thể làm khác placement, cần ghi nhận thay vì đòi output giống nhau.
   - Verify: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg python -m pytest -q tests/test_gt_sampler_audit.py`. Đối chiếu tổng counter bằng fixture biết trước số trial và số điểm, không chỉ kiểm tra JSON tồn tại.
   - Verify khi có dữ liệu: `python tools/visualization/audit_gt_sampler.py --config configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json --frames /tmp/gt-audit-frames.txt --seed 42 --output-dir /tmp/gt-sampler-audit`. Đây là giao diện CLI dự kiến triển khai, chưa chạy được ở thời điểm lập plan.

Tiêu chí hoàn thành phần sửa code (bước 1–14): các phản ví dụ shadow/LOS có kết quả đúng; placement bị từ chối không làm đổi scene; visibility luôn đạt policy đã cấu hình sau mọi lượt chèn; từng flag hoạt động riêng; no-op/empty/schema nhất quán; train thiếu database báo lỗi; val/test không phụ thuộc database augmentation; các test hồi quy và tích hợp đạt.

Tiêu chí đánh giá tiếp theo (bước 15): có thống kê từ frame thực về tỷ lệ chèn, mất điểm, phân bố khoảng cách và throughput. User chỉ cần cung cấp đường dẫn processed KITTI, database và config train thực tế ở bước này. Không đánh dấu kiểm chứng dữ liệu thực hoàn thành khi mới có fixture tổng hợp.

Sau code fix và audit mới thực hiện ablation: cùng split/model/optimizer/budget, so sánh GT sampling tắt, GT sampling cơ bản, rồi thêm từng thành phần physics. Thực hiện nhiều seed nếu tài nguyên cho phép; ghi effective flags và database identity. Không suy ra mAP tăng từ test unit hoặc một hình visualization. Ground theo footprint/plane, ngưỡng visibility, min-point floor, phân bố proposal và yaw/viewpoint được đánh giá như các thay đổi riêng.

### Risks & mitigations

- Policy visibility và ground chặt hơn có thể làm giảm số mẫu được chèn. Tách counter từ chối để xác định nguyên nhân; không nới ngưỡng chỉ để đạt counts mong muốn. Hai ngưỡng visibility đề xuất cần xác nhận trên dữ liệu thực.
- Giao tia đúng với box nhưng box vẫn là vật thể đặc xấp xỉ; khoảng trống giữa chân pedestrian hoặc phần rỗng của cyclist không được mô phỏng. LOS vẫn dùng heuristic số điểm, không phải tỷ lệ che phủ hay ray tracing bề mặt đầy đủ.
- Ground support hiện tại có thể chấp nhận điểm nằm ngoài footprint hoặc từ chối mặt đường khác cao độ. Đợt này chỉ bỏ bypass sai; chưa tuyên bố khắc phục heuristic đó. Nếu audit cho thấy vấn đề đáng kể, lập thay đổi ground riêng với ca slope/curb/sparse-road.
- Chi phí tính mask/visibility có thể tăng theo số điểm, box và trial. Vectorize theo điểm, lọc sơ bộ bảo toàn kết quả, cache membership theo trạng thái scene; tránh tensor đặc `N_points × N_candidates × N_boxes`. Đo p50/p95 và bộ nhớ trước khi đặt ngân sách latency; không dùng cam kết `<2ms/frame` từ tài liệu cũ làm kết quả đo.
- Fixture cũ dùng point cloud zero và object Z âm có thể fail sau validation đúng. Thay fixture bằng dữ liệu hợp lệ, giữ assertion chứng minh behavior; không giảm kiểm tra để giữ fixture phi thực tế.
- `enable_physics=False` sẽ vẫn xóa điểm trong box, là thay đổi có chủ ý của basic GT sampling. Tài liệu hóa và có regression để người chạy ablation hiểu baseline mới.
- Constructor validation tăng chi phí load database; chỉ làm một lần và báo lỗi có ngữ cảnh. Không tự chuyển đổi database không rõ schema hay class mapping.
- Review hiện chưa có dữ liệu thực. Điều kiện hoàn thành code và điều kiện hoàn thành audit/mAP phải được báo cáo riêng.

### Rollback plan

- Tách các nhóm commit: geometry/occlusion; transaction/visibility; config/ground/API/database; docs/audit. Mỗi commit kèm test liên quan, không gộp thay đổi model hoặc tuning vào đây.
- Nếu gặp vấn đề trong một run, đặt `enable_gt_sampling=false` trong bản config của run đó để tắt sampling; ghi rõ run này dùng cấu hình khác. Không đổi tên nó thành thí nghiệm có GT sampling.
- Khi cần so sánh behavior cũ, chạy commit baseline trong checkout riêng với database và seed tương ứng; không thêm chế độ cố tình dùng shadow sai vào production chỉ để rollback.
- Revert commit sửa có vấn đề thay vì reset workspace. Không sửa/xóa dữ liệu KITTI, database nguồn hoặc checkpoint của các run cũ.
- Lưu config và kết quả audit theo từng phiên bản. Giữ nguyên bằng chứng test/benchmark của mỗi bản để truy lại khác biệt.
