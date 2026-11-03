# Bộ ablation đầy đủ cho bài BEV MobilePixorNeXt

## Phạm vi và proposed dự kiến

Đây là đề xuất tổ chức bài theo thiết kế hiện tại trong repo, chưa phải xác nhận novelty và chưa phải chốt final model. Người dùng chọn BEV. Kế hoạch cũ Rep/LiteMLA/Bi-SGFPN và kế hoạch mới hist14/focal/grouped không được gộp thành một proposed mặc định.

Proposed candidate P: hist14 → MobilePixorNeXt depths3/4/2 → focal context stride8 → FPN → heads Car / Pedestrian+Cyclist → OGA objective + optional BEV IQA quality ranking. Dùng hybrid GT như một training recipe có đối chứng. ECA/SimAM, scale gate, learned pillar và reparameterization không phải thành phần bắt buộc của candidate này.

Các câu hỏi nghiên cứu: biểu diễn height/geometry có giúp; context đa mức có giúp dưới budget nhẹ; tách regression theo nhóm có giảm xung đột; geometry/adaptive weighting và quality ranking có giúp localization/ranking. Các cơ chế chỉ được giữ trong final model và claim khi có bằng chứng; histogram/focal/ECA đều có prior art, không tự coi tên module là đóng góp mới. Nếu OGA là đóng góp chính, phải thêm ablation từng hạng loss và đối thủ loss phổ biến.

## Có gì rồi

R0/R1: MobilePIXOR/NeXt legacy35, mAP41,40/47,60; không phải cặp backbone cô lập: headBN/activation khác, lịch thực tế khác, chọn loss. R0 log38/100, R1 đủ100.

R2/R3/R4/R8: NeXt3/4/2, rich10/rich12/rich8/hist14, singleH16/F24, standard50epoch, AP policy10epoch. mAP46,82/46,98/47,37/50,14. Đây là encoder screening, với khác biệt val batch và source cần kiểm tra khi tái sử dụng.

R5: rich8 groupedH16/F24 standard49,01; lịch100/warmup8, log70epoch; chưa cô lập head so R4.

R6/R7: hist14/rich8 groupedH32/F32 hybrid,50epoch; mAP67,13/66,79; no focal, no scale gate, baseline loss, no IQA. Đây là hai anchor mạnh. Hist14 so rich8 có đối chứng config tốt nhưng source commit khác.

Tất cả9run cùng validation3.769frame/split hash/input asset hash. Không có trained result trong thư mục này cho focal/ECA/SimAM/OGA/IQA/gate/reparam. File preset, structural profile hoặc smoke test không thay thế kết quả trained. Tài liệu archive20run trong repo dùng validation1.497frame, không được ghép AP vào bảng hiện tại.

## Protocol chung cho core matrix

- NeXt3/4/2, F32/H32, gateoff, local-attention none; grid0,1m và ROI hiện tại.
- Cùng hybrid recipe cho tất cả core matrix, kể cả baseline. Vì thế không nhầm architecture gain với augmentation gain.
-50epoch/warmup4; AdamW lr0,0007, wd0,001; batchtrain16, val16; trainBF16; primary APModerate macroR40; APevery10/final; score0,05, NMS0,1,maxdet500.
- B: rich8, single, contextnone, baseline loss, IQAoff. E=hist14; G=grouped; F=focal; L=OGA; I=IQA. L0 vẫn giữ loss baseline thông thường, không nghĩa là tắt regression.
- IQA target phải là BEV và đặt rõ target type; nếu là MGIoU proxy thì ghi proxy, không gọi true IoU. P có chất lượng dự đoán dùng ranking với alpha được khóa theo sweep validation. Không dùng AP3D.

## Main table: cumulative + leave-one-out + encoding/context interaction

| ID | E | G | F | L | I | Vai trò | Có kết quả hiện tại? |
|---|---|---|---|---|---|---|---|
| M0 |0|0|0|0|0| Strong baseline B | Chưa: R4 khácF/H vàaugmentation |
| M1 |1|0|0|0|0| B+encoding | Chưa: R8 khácF/H vàaugmentation |
| M2 |1|1|0|0|0| +grouped | R6, tái dùng có điều kiện |
| M3 |1|1|1|0|0| +focal | Chưa |
| M4 |1|1|1|1|0| +OGA, P không IQA | Chưa |
| M5 |1|1|1|1|1| Proposed candidate đầy đủ | Chưa |
| M6 |0|1|1|1|1| P bỏhist14, thayrich8 | Chưa |
| M7 |1|0|1|1|1| P thaygrouped bằngsingle | Chưa |
| M8 |1|1|0|1|1| P bỏfocal | Chưa |
| M9 |1|1|1|0|1| P thayOGA bằngbaseline | Chưa |
| M10 |0|1|0|0|0| rich8 anchor, interaction control | R7, tái dùng có điều kiện |
| M11 |0|1|1|0|0| rich8+focal, interaction control | Chưa |

P bỏIQA chính là M4, không train thêm duplicate. Có12config khác nhau;10chưa có run tương ứng; có thể cần chạy lại M2/M10 nếu không chứng minh được source/data/training tương thích với batch mới.

M2/M3/M10/M11 kiểm tra2×2encoding×context. M3/M4/M9/M5 kiểm tra2×2loss×IQA. Không chỉ dùng cumulative, vì gain từng module có thể phụ thuộc module khác. Các leave-one-out là ablation thay hệ thống tương ứng, thay đổi cả params; báo params để không gán mọi lợi ích cho một cơ chế khi capacity khác.

## Những bảng hỗ trợ phải có tùy claim

### Backbone và capacity

Nếu claim NeXt cải tiến backbone, chạy MobilePIXOR/NeXt2-4-2/NeXt3-4-2 trên cùng legacy35, cùng single headH16 BN+SiLU, gateoff, cùng50epoch và augmentation. Head interface phải khớp thực tế; không ép configH32 vào backbone không hỗ trợ. Đây là3recipe; các runR0/R1 chỉ là thăm dò.

Trên P hoặc anchor cố định, factorialF24/H16,F24/H32,F32/H16,F32/H32 tách fusion width với output/head capacity. Ba dòng bổ sung ngoài anchorF32/H32 nếu cần claim về width. Tăng width là tuning/capacity, không tự đặt thành contribution mới.

### Encoder mechanism

R4/R8 chứng minh screeningrich8/hist14 trong recipe yếu; M5/M6 là đối chứng chính trong recipeP.

Nếu claim histogram/offset là cơ chế mới: thêm P dùnghist14 bỏhistogram và P dùnghist14 bỏxyoffsets. Quy định normalization/channels và profile lại; đây là schemas chưa có sẵn, cần triển khai/kiểm chứng, không đưa flag tùy ý vào JSON. Baselinelegacy35 và rich10/rich12 trên một anchor cố định là bổ sung nếu cần bảng nhiều encoder, không bắt buộc rerun mọi encoder trên mọi architecture.

### Context mechanism

M5/M8 trả lời focal có giúp. Nếu claim multiscale/global fusion: thêm focal single-level, multilevel khôngglobal, fullmultilevel/global; khóa width/placement/những phần khác. Nếu claim learned gate: thêm uniform fusion thay learnedgate; cần code/configcontrols nếu chưa hỗ trợ. Khi bỏglobal, phải xác định gate renormalization và inactiveparameter policy. Bảng none/ECA/SimAM/focal dùng cùng hook nếu claim so thuật toán; ECAstride4 so focalstride8 là so hai thiết kế placement, không gọi attention-onlycomparison.

### Grouped mechanism

M1/M2 và M5/M7 tách grouped ở recipe đầu và cuối. Nếu claim partitionCar/(Ped,Cyc) tối ưu, so single,2groups,3groups với cùngrecipe; reportparams và thêmcapacity-matchedsingle nếu muốn loại nguyên nhân thêmcapacity. Thêm diagnostic sốcell có regressionownershipcollision và AP/error ở các cell đó; không cần training riêng cho diagnostic. Group weights phải được khai báo/chuẩn hóa nhất quán.

### OGA mechanism

Đặt Ioff, architecture/encoding/head/context cố định. Bảng: baseline; baseline+normalizedpi-symmetriccorner vớifixedweights; baseline+projectionMGIoU vớifixedweights; baseline+both vớifixedweights; fullOGA gồmgeometry+adaptiveweighting. L0/L4 có thể dùng M3/M4;3term-ablation giữa cần thêmrecipe và implementationcontrols. Không lấy baseline vsOGA duy nhất để claim từng hạngloss. Nếu adaptiveweighting là contribution chính, thêm cùngloss terms vớiequalweights và đối thủ adaptiveweighting phổ biến; nếu claim EMA/boundedsoftmax, ablateEMA/bound riêng. Các controls này chưa có đầy đủ trong runtimeJSON hiện tại.

### IQA training và inference

M3/M4/M9/M5 là factorialbaseline/OGA×IQA. Để tách auxiliarytraining vớiqualityranking, lấy cùngcheckpointIQAon, evaluatealpha0 vàalphapositive. Sweepalpha[0,0.25,0.5,0.75,1] không cần retrain, trừ khi alpha được dùng trong trainingobjective. Đổi qualitysupervisiontarget MGIoUproxy/rotatedBEVIoU là training khác và phải train lại nếu claim target tốt hơn. Calibration plot/bin quality-vsactualIoU và falsepositive có thể lấy từ prediction hiện có sau khi train.

### Augmentation

Đặt cùng architecture P hoặc anchor; train4recipe: OneOf cũ; compositionalglobal khôngsampler; compositionalglobal+standardGT; compositionalglobal+hybridGT. Hybrid/standard phải cùngdb,quotas,probability,globaltransforms; P đãhybrid nên3recipe mới ngoàiP. Các configA2/A3/A4 đã tạo trước là trên anchorno-focal/baseline/noIQA; không tự coi chúng là ablation trên fullP.

Nếu hybrid là contribution chính, thêm riêng disable densitysubsampling,groundvalidation,visibilityprotection hoặcstaticcollision, theo đúng module active. Không cần ablate shadowmasking/radiometric/lineofsight nếuP không dùng. GT database phải từtrainIDs.

## Main comparison, thống kê và efficiency

Ablation chỉ so thành phần nội bộ. Main comparison cần ít nhất implementationbaselineMobilePIXOR matched và một externalmethod phù hợp như PointPillars vớiAPBEV ba lớp. PIXOR gốc phù hợp bảngBEVCar nếu dùng nó; không tự lấy kết quảCar củaPIXOR làmmacroba lớp. Externalmethod giữ trainingrecipe hợp lý cho phương pháp và khai báo điều kiện; không ép cùngLR cho mọiarchitecture để gọi fair. Dataset/split/evaluator/sampling/classdomain phải giống.

Reference wrapper hiện không chạy trực tiếp BEVcheckpoint; cần kiểm chứng BEVreference path trước comparison vớipaper. Internal localBEV có thể dùng để screening, ghi đúngprotocol. Không lấy mAParchivevalidation1497 so current3769.

Sauseed42screening, các rows hỗ trợ claim chính cần pairedseeds42/43/44 vàmean±sampleSTD. Ít nhấtbaseline/P/P-bỏtừngcontribution, không chỉ repeatsP. Mộtseed tăng0,34AP không phải bằng chứng ổn định. Statisticalresampling frame không thay thếtrainingseedvariability.

ReportEasy/Mod/Hard từnglớp; Moderate macro; backbone/neck/heads/totalparams vàcriterion riêng; MACs đúnginput; latencybatch1/preprocess/H2D/model/NMS/e2e, warmup vàhardware/precision nhấtquán; modelFPS không thay pipelineFPS. DeploymentTensorRT/Jetson chỉ bắtbuộc nếupaperclaim trên chúng.

Distance/density/occlusion diagnostics, PRcurves,qualitativeTP/FP/FN vàclasswise localizationerrors dùngcheckpoint, không cần train mới. Giữ sample counts; cácbandfar ítGT không dùng làm kếtluận chính. Datasetthứhai chỉ cần nếuclaimgeneralization rộng.

## Thứ tự thực hiện và điều kiện hoàn thành

1. Khóa claims/protocol và dùngR6/R7 làmanchors. ChạyM3/M11 để biết focal có gain trênhist14 vàrich8; gateoff vàbaseline/IQAoff ở vòng này.
2. ChạyM4/M9/M5 để giải quyếtOGA/IQA, vàM0/M1 để giải quyếtsingle/grouped trêncùngrecipe. Hoàn thành mainmatrix12rows; nếu mộtmodule bịloại, viếtlạifinalmatrix theoPđãchọn.
3. Chạyaugmentationcontrols vàmatchedbackbonecontrols; thực hiệnterm/mechanismablations chỉ cho những contributionthật sự giữ trongpaper. Không cần20moduleoptionalđể gọihoànchỉnh.
4. Referenceevaluation/maincomparisons,pairedseeds,efficiency vàerroranalysis. Chốtours bằng bằngchứng, cậpnhậtsơđồ/configresolved.

Tổngsốrun không thểchốt thànhmộtcon số trước khiquyếtđịnh claims. Core là12recipe/10–12trainingmới ởseed42; thêm3augmentationcontrols và3matchedbackbonecontrols đưa lên18recipe. NếuOGA làcontribution mới: thêm3termlosscontrols; nếuhist14claim từngfeature: thêm2schemacontrols. Như vậy mộtbộ bao gồm cảencoding+context+grouped+OGA/IQA có thểkhoảng20–25recipe ởseedđầu, trướcseedsxácnhận; sweep/diagnostics khôngtính làtraining. Không tạo25JSON rồi mặcđịnhchúngđãrunnable: cácmechanismcontrols chưaexposed cầnimplementation vàverification.

Hoànthành khi tấtcảclaims chính cómatchedcontrols, modelP/final vàablations cóseeds/phươngphápđánhgiá phùhợp, AP/latency đủmetadata, và cácmoduletrongfiguretrùngfinalcheckpoint. Hiện đãcóscreening/anchors; chưacótrainedfullP haycompletepaperablation.
