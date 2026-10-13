# Citation audit

Ngày xác minh: 13/10/2026. Nguồn chính: proceedings chính thức, trang tác giả/project chính thức, publisher và arXiv. Web search đã lưu tại các tệp `web_papers*.txt`; PDF tham khảo của người dùng đã được đọc toàn văn.

| BibTeX key | Claim dùng trong manuscript | Nguồn primary |
|---|---|---|
| `geiger2012kitti` | Dataset KITTI object detection; split cụ thể lấy từ manifest project. | [Nguồn](https://www.cvlibs.net/datasets/kitti/) |
| `geiger2013raw` | KITTI raw sequences dùng trong video minh họa. | [Nguồn](https://www.cvlibs.net/datasets/kitti/) |
| `yang2018pixor` | Dense BEV oriented detection và tiền đề PIXOR. | [Nguồn](https://openaccess.thecvf.com/content_cvpr_2018/html/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.html) |
| `lang2019pointpillars` | Learned pillar representation và 2D backbone. | [Nguồn](https://openaccess.thecvf.com/content_CVPR_2019/html/Lang_PointPillars_Fast_Encoders_for_Object_Detection_From_Point_Clouds_CVPR_2019_paper.html) |
| `yin2021centerpoint` | Center-based LiDAR object detection. | [Nguồn](https://openaccess.thecvf.com/content/CVPR2021/html/Yin_Center-Based_3D_Object_Detection_and_Tracking_CVPR_2021_paper.html) |
| `li2023pillarnext` | Thiết kế mạng pillar hiện đại; không dùng score công bố làm baseline cùng protocol. | [CVPR proceedings](https://openaccess.thecvf.com/content/CVPR2023/html/Li_PillarNeXt_Rethinking_Network_Designs_for_3D_Object_Detection_in_LiDAR_CVPR_2023_paper.html) |
| `zhou2023fastpillars` | Deployment-oriented pillar detector; ghi là arXiv preprint. | [Nguồn](https://arxiv.org/abs/2302.02367) |
| `chandorkar2025dense` | Nghiên cứu backbone LiDAR gọn gần đây; dùng bản ICCV Workshops 2025 đã xác minh. | [CVF proceedings](https://openaccess.thecvf.com/content/ICCV2025W/EVW/html/Chandorkar_Rethinking_Backbone_Design_for_Lightweight_3D_Object_Detection_in_LiDAR_ICCVW_2025_paper.html) |
| `nguyen2024indoor` | Bài tham khảo do người dùng cung cấp: IRBGHR-PIXOR, Gaussian heatmap, JRDB. | [Nguồn](https://ieeexplore.ieee.org/abstract/document/10385189) |
| `sandler2018mobilenetv2` | Inverted residual architecture motivation. | [Nguồn](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) |
| `liu2022convnext` | Large-kernel depthwise convolution và channel transformation. | [Nguồn](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html) |
| `lin2017fpn` | Top-down/lateral multiscale fusion. | [Nguồn](https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) |
| `yang2022focalnet` | Focal context modulation; bản project là adaptation riêng. | [Nguồn](https://proceedings.neurips.cc/paper_files/paper/2022/hash/1b08f585b0171b74d1401a5195e986f1-Abstract-Conference.html) |
| `wang2020eca` | ECA channel attention, nguồn cho alternative P15. | [Nguồn](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.html) |
| `yang2021simam` | SimAM attention; residual scalar của project được mô tả riêng. | [Nguồn](https://proceedings.mlr.press/v139/yang21o.html) |
| `lin2017focal` | Focal classification principle. | [Nguồn](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html) |
| `zhou2019objects` | Gaussian center heatmap/object-as-center supervision. | [Nguồn](https://arxiv.org/abs/1904.07850) |
| `rezatofighi2019giou` | Generalized overlap supervision. | [Nguồn](https://openaccess.thecvf.com/content_CVPR_2019/html/Rezatofighi_Generalized_Intersection_Over_Union_A_Metric_and_a_Loss_for_CVPR_2019_paper.html) |
| `le2026mgiou` | Multi-axis projection geometry; attribution cho projection part của OGA. | [Nguồn](https://ojs.aaai.org/index.php/AAAI/article/view/37505) |
| `li2020gfl` | Quality-aware classification; không khẳng định project dùng nguyên bản distributional GFL. | [Nguồn](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html) |
| `kendall2018uncertainty` | Multi-task loss balancing context; project không dùng đúng probabilistic objective của nguồn này. | [Nguồn](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) |
| `loshchilov2019adamw` | AdamW optimizer. | [Nguồn](https://arxiv.org/abs/1711.05101) |
| `openpcdet` | Software citation cho pinned evaluator; không phải paper baseline. | [Nguồn](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/pcdet/datasets/kitti/kitti_object_eval_python) |

## Các quyết định metadata và phạm vi

- 23 entry có author/title/year và DOI hoặc URL truy nguyên. FastPillars và Objects as Points dùng bản arXiv; bài lightweight backbone 2025 dùng bản ICCV Workshops (không mô tả như ICCV main conference). Không tự tạo journal/volume/pages.
- PillarNeXt được xác minh ở CVPR proceedings: 17567–17576; metadata này đã đưa vào BibTeX. AdamW dùng arXiv của tác giả vì truy cập OpenReview bị hạn chế.
- Focal Modulation Networks dùng author list của proceedings NeurIPS 2022. Nguồn repository/preprint có thể khác author list của bản proceedings.
- Bài IRBGHR-PIXOR có hai tác giả cùng tên Duy Anh Nguyen trong PDF; giữ cả hai theo publication. DOI 10.1109/ACCESS.2024.3351868, IEEE Access 12:9162–9176.
- OpenPCDet được trích như software với revision cụ thể; hash evaluator được lưu trong final evaluation JSON.
- Không citation giả, citation placeholder hoặc số liệu SOTA lấy từ protocol khác. Số AP/latency của project lấy từ artifact, không lấy từ literature.
- Bản main và supplementary có bibliography riêng nên số thứ tự citation có thể khác; chúng dùng chung `references.bib`.
