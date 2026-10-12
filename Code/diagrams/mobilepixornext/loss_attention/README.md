# Loss và attention theo code hiện tại

[Mở gallery](index.html) · [PDF gồm 6 hình](loss_attention_atlas.pdf) · [Audit nguồn và số học](audit.json).

| Hình | PNG | SVG chỉnh sửa | PDF vector |
| --- | --- | --- | --- |
| OGA | [PNG](oga.png) | [SVG](oga.svg) | [PDF](oga.pdf) |
| Q-OGA | [PNG](q_oga.png) | [SVG](q_oga.svg) | [PDF](q_oga.pdf) |
| Focal Context | [PNG](focal_context.png) | [SVG](focal_context.svg) | [PDF](focal_context.pdf) |
| SimAM | [PNG](simam.png) | [SVG](simam.svg) | [PDF](simam.pdf) |
| Gaussian focal loss | [PNG](focal_loss.png) | [SVG](focal_loss.svg) | [PDF](focal_loss.pdf) |
| T-SBUW task weighting | [PNG](task_weighting.png) | [SVG](task_weighting.svg) | [PDF](task_weighting.pdf) |

Nguồn config: `/home/duyennh/AI_projects/research_lidar/Lidar/Code/configs/experiments/under1m/hist14_local3_focal_simam_fusion32_shared_grouped_oga_iqa.json`; bản sao giữ trong [config.resolved.json](config.resolved.json).
Đây là config OGA/IQA và Focal/SimAM. Q-OGA được vẽ như một lựa chọn loss riêng,
không phải loss đang bật cùng OGA trong config này. Hình loss áp dụng cho Gaussian BEV.

- **OGA:** cls focal, ba masked Smooth-L1, và geometry từ offset/size/yaw. Geometry
  gồm projection GIoU loss và corner distance chọn min(direct, đảo 180°), chia đường
  chéo GT. IQA tùy chọn dùng BCE với target quality detached; config này bật IQA
  với MGIoU proxy và weight=1. Mỗi group có module T-SBUW riêng, 6 logits học được
  khi bật IQA, hoặc 5 khi tắt. Masked Smooth-L1 chuẩn hóa trên số tọa độ scalar,
  còn geometry chuẩn hóa trên số box positive.
- **Q-OGA:** thay focal ở peak bằng CQFL với target q^beta, dùng range weights cho
  ba coordinate losses và soft-min corner matching. Default quality_target=mgiou
  dùng clamp(mean GIoU trên 4 trục,0,1), **không phải polygon IoU**. Opt-in
  quality_target=rotated_iou dùng exact area IoU; curriculum optional có
  a=clip(epoch/Ewarm,0,1), qpeak=(1-a)+a IoU. Epoch tính từ 0; Ewarm mặc định 0.
  Nhánh target quality không truyền gradient về regression. Gaussian neighbors
  vẫn được supervision negative, không chuyển thành positive quality targets.
  Full pipeline Q-OGA không dùng IQA head riêng. Q-OGA có 5 adaptive task logits
  cho mỗi group. RDA theo đúng code dùng grid tại (j+0.5,i+0.5), cộng GT offset;
  coordinate loss dùng sum 2 tọa độ rồi mean box, khác normalization OGA.
  Soft-min trong code không cộng tau*log(2); giá trị corner term có thể âm nhỏ.
- **Focal Context:** C4=96×100×88; BN + Conv1×1 chia thành Q64/S64/G4. Ba DWConv
  dilation 1→2→3 nối tiếp, SiLU sau mỗi lớp. Global context là mean của lớp cuối.
  Softmax G trên 4 levels, fusion theo từng pixel rồi Conv1×1, nhân query, Conv1×1
  và BN. Residual dùng 96 channel scales gamma, khởi tạo .001. 25,120 tham số.
- **SimAM:** C3=48×200×176; mean/variance độc lập cho từng batch/channel, trên HW.
  d=(X-meanHW(X))², v=sumHW(d)/max(HW-1,1), e=d/[4(v+lambda)]+.5.
  Core T(X)=X*sigmoid(e) không có tham số; wrapper Y=X+beta(T(X)-X) thêm một
  scalar beta học được, khởi tạo .001. Lambda=.0001; các thống kê dùng FP32.
  Đặt sau Stage2, trước Down3 và lateral C3.
- **Focal loss** là loss heatmap riêng, khác **Focal Context** trong backbone.
- **T-SBUW:** s'=3*tanh(s/3), w=M*softmax(s'/2), loss calibration Li/max(EMAi,1e-4).
  Lgroup=mean(EMA)*sum(w*calibrated loss). EMA dùng detach và update trong train,
  init bằng loss clamp đầu tiên, sau đó momentum .99. Group weights phía ngoài
  được chuẩn hóa thành .5/.5 trong config này; không phải các task weights.

Generator chạy các module production, kiểm tra phép dựng lại Focal/SimAM,
shape, số tham số, forward/backward OGA/Q-OGA, quality detached, curriculum
epochs 0/2/4, và CQFL(q=1)=modified focal loss. Không sửa model/loss hay chạy
huấn luyện dataset. PNG 300 DPI; SVG giữ text; PDF vector; TikZ dùng PlotNeuralNet
từ third_party, cần `.tex` và `layers/` khi tải Overleaf. TikZ chưa biên dịch
vì môi trường thiếu LaTeX compiler; PDF được xuất bằng Matplotlib.

Tạo lại từ folder Code:

```bash
python tools/visualization/draw_loss_attention_diagrams.py
```
