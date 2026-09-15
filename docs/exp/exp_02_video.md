# Exp-02. Train-Free Geometric Flow Matching: 에피폴라 제약 비디오 실험

상태: **데이터 입력·대응점 제약 구현 / 생성 파이프라인 미구현** (2026-09-15). CLEVRER 기반 물리 인식·충돌 제약 계획을 종료하고, RealEstate10K의 카메라 기하를 이용한 비디오 생성 실험으로 전환한다. [Exp-02-sub](exp_02_sub_video_recognition.md)는 취소된 실험 기록으로 보존하며 선행 조건에서 제외한다.

핵심 질문은 **동결된 비디오 Flow Matching 모델의 추론에 기하 보정을 넣었을 때, 영상 품질과 대응점 검출률을 유지하면서 지정 카메라의 에피폴라 오차를 줄일 수 있는가?**이다. 대응점 공간의 hard projection을 먼저 검증하고, 영상·잠재 공간으로 전달하는 과정은 별도의 연구 가설로 검증한다. 최종 비디오의 3D consistency 또는 100% 제약 만족을 사전에 보장하지 않는다.

## 1. 채택하는 방향과 수정할 가정

정적 장면과 알려진 카메라를 사용하면 CLEVRER의 객체 ID·world 상태·충돌 인식기를 직접 학습할 필요가 없다. 다만 다음 구분이 필요하다.

| 제안 | 이번 계획의 판단 |
| :--- | :--- |
| RealEstate10K의 카메라로 F 계산 | 채택. 제공 포즈를 기준값으로 사용하되 실제 영상에서 오차와 퇴화를 검증 |
| Wan 잠재 속도를 에피폴라 선에 직접 사영 | 불가. 잠재 채널 변화율은 2D optical flow가 아니며 차원과 의미가 다름 |
| 2D 대응점/변위에 닫힌 형태 투영 | 채택. 고정된 기준점과 비퇴화 에피폴라 선에서 정확하게 정의 가능 |
| 접선 속도 투영만으로 hard constraint 달성 | 초기 제약 만족이 필요. 이미 존재하는 위치 잔차는 별도 위치 투영으로 제거 |
| 역전파 없는 O(N) 전체 파이프라인 | O(N)은 N개 대응점 투영에 한정. 매칭·VAE·warping·DiT 비용은 별도 |
| 에피폴라 만족이면 3D 일관성 확보 | 필요조건의 일부. 깊이, 양의 깊이, 다중 뷰 일관성, 외관, 실제 카메라 이동은 별도 |
| SAM 2로 정적 배경 자동 판별 | SAM 2는 분할·추적 도구. 객체의 정적/동적 여부는 별도 판정 필요 |

SAM 2는 첫 실험의 필수 의존성으로 두지 않는다. 사람이 확인한 정적 클립부터 시작하고, 필요할 때 지정 객체의 제외 마스크를 전파하는 용도로 추가한다. [SAM 2 공식 구현](https://github.com/facebookresearch/sam2)

## 2. 모델·데이터·입출력

### 2.1 백본 선택

- **첫 생성 실험:** 기존 자산을 활용하는 Wan2.1-T2V-1.3B. 텍스트 조건으로 장면을 생성하고, 생성된 첫 프레임을 대응점 기준으로 삼는 `T2V + target-camera geometry` 모드다. RealEstate10K의 첫 프레임을 native I2V 입력으로 처리하지 않는다.
- **후속 장면 조건 실험:** 첫 프레임 I₁이 필수이면 Pyramid Flow의 공식 I2V 경로를 우선 검토한다. Wan2.1의 공식 I2V 모델은 14B 계열이므로 가용 VRAM·offload 비용을 먼저 측정한다.
- 모델 변경 시 같은 백본 내부에서 baseline과 보정 방법을 다시 비교한다. T2V와 I2V 결과를 하나의 방법 효과로 합치지 않는다.

위 기능 구분은 [Wan2.1 공식 구현](https://github.com/Wan-Video/Wan2.1)과 [Pyramid Flow 공식 I2V 예제](https://github.com/jy0205/Pyramid-Flow)를 따른다. 모델 가중치는 모두 동결하며 학습·fine-tuning은 수행하지 않는다.

T2V 모드에서 데이터셋 카메라는 **목표 제어 입력**이다. 생성 장면의 실제 카메라 정답이라고 부르지 않는다. 생성 장면과 원본 장면의 픽셀 재현은 성공 기준이 아니다. I2V도 카메라 조건을 원래 받는 모델이라고 가정하지 않으며, 포즈는 외부 보정기가 사용한다.

### 2.2 RealEstate10K 구성

공식 배포는 비디오 URL, timestamp, 카메라 파라미터를 담은 텍스트 파일이며 train/test로 나뉜다. 영상 확보와 timestamp 정렬을 먼저 검증한다. 다운로드 가능한 클립 목록과 실패 목록을 고정한다. 카메라는 world-to-camera 행렬이며 intrinsics는 해상도 정규화 좌표다. [공식 데이터 형식](https://google.github.io/realestate10k/download.html)

- 개발: train에서 정적·텍스처 충분·시점 중첩이 있는 10클립. 임계값·보정 강도·시간 구간은 여기서 결정한다.
- 파일럿 평가: test에서 독립 20클립 × 3 seed. 같은 원본 URL이 개발/평가에 겹치지 않도록 검사한다.
- 확대: 파일럿 통과 후 test 100클립 × 3 seed. 소규모 FVD를 주 판정 지표로 쓰지 않는다.
- 첫 기하 실험은 2프레임 쌍, 이어서 짧은 다중 프레임으로 확장한다. 생성은 기존 33프레임·480×832를 초기 후보로 두고 공식 모델 지원 규격과 메모리 측정으로 확정한다.
- 균일 요청 시각과 실제 선택 timestamp를 함께 저장한다. 포즈가 붙은 실제 프레임을 우선 사용하고, 무기록 포즈 보간이나 단순 프레임 번호 정렬을 피한다.
- 동적 물체, 장면 전환, 큰 가림, 반복 무늬, 순수 회전·극소 baseline은 별도 표기한다. 데이터 선별은 생성 결과를 보기 전에 고정한다.

입력 manifest는 `clip_id, source_url, split, timestamps, K, R, t, spatial_transform, valid_region, prompt, seed, pair_list`를 포함한다. I2V에서만 `initial_frame`을 모델 조건에 추가한다. 원본 미래 RGB는 오라클 검증·평가 전용이며, 생성 보정기가 접근하지 않는다. 프롬프트는 첫 프레임에서 작성하고 모든 방법이 공유한다.

출력은 압축 전 `video_raw`, `video_final`, 단계별 잔차·보정량·매칭 수, 실패 사유, 실행 시간·VRAM, 모델/코드 revision·scheduler·전처리 설정이다. MP4는 시각화 산출물이다.

## 3. 기하 정의와 정확한 투영 범위

영상 프레임 번호 k, 물리적 영상 시간 s, 생성 ODE 시간 τ를 구분한다. 아래 식은 직접 유도한 구현 계약이며, [좌표계 분리 원칙](../../data/NOTICE.md)에 따라 제약을 이미지 픽셀 좌표에서 계산한다.

### 3.1 상대 포즈와 F

world-to-camera 규약 Xᵢ = RᵢXw + tᵢ에서:

$$R_{k1}=R_kR_1^\top,\qquad t_{k1}=t_k-R_{k1}t_1,$$
$$E_{k1}=[t_{k1}]_\times R_{k1},\qquad F_{k1}=K_k^{-\top}E_{k1}K_1^{-1}.$$

정규화 intrinsics를 원본 W,H로 픽셀화하고, resize/crop/padding 변환 Aᵢ를 적용해 Kᵢ′ = AᵢKᵢ로 갱신한다. 기존 F를 변환한다면 F′ = Aₖ⁻ᵀFA₁⁻¹이다. 픽셀 중심 규약도 전처리와 일치시킨다.

t=0이면 F=0이므로 에피폴라 점수를 정의하지 않는다. 순수 회전은 회전 homography KₖRₖ₁K₁⁻¹ 기반 별도 진단 대상으로 둔다. 작은 baseline의 퇴화 판정은 임의 스케일의 translation norm만으로 하지 않고 시차·매칭의 관측 가능성을 함께 본다.

### 3.2 대응점의 위치 투영: MVP의 hard constraint

기준점 p₁=(x₁,y₁,1)과 목표 프레임 추정 좌표 q=(xₖ,yₖ)를 두고:

$$l=Fp_1=(a,b,c)^\top,\quad n=(a,b)^\top,\quad r=n^\top q+c,$$
$$q^\star=q-\frac{r}{n^\top n}n.$$

이는 **p₁을 고정한 채 q만 움직이는 최소 Euclidean 거리 투영**이다. 비퇴화 선에서 nᵀq★+c=0이며, 양쪽 점을 동시에 최적화하는 Sampson 보정과는 다르다. optical flow u=q−(x₁,y₁)에 적용할 때는:

$$u^\star=u-\frac{n^\top((x_1,y_1)^\top+u)+c}{n^\top n}n.$$

분모가 작으면 invalid로 처리한다. epsilon을 더해 얻은 근사해를 exact projection이라고 부르지 않는다. 화면 밖 투영은 실패/가림으로 기록한다. 단순 clamp는 선 제약을 다시 깨뜨린다. 화면 내부까지 제약하려면 선과 유효 사각형의 교집합 선분에 투영하고, 교집합이 없으면 infeasible이다.

### 3.3 접선 투영과 latent ODE의 차이

2D 좌표가 ODE 상태이고 기준점·F가 고정된 경우에만:

$$\dot q_{\mathrm{tan}}=\dot q-\frac{n^\top\dot q}{n^\top n}n.$$

이 식은 잔차의 변화를 막지만 이미 있는 잔차를 없애지 않는다. 기준점이나 F가 바뀌면 그 미분 항도 필요하다. 우선 위치 투영을 수행하는 이산 보정으로 구현한다.

Wan 상태 z와 속도 vθ는 `[B,C,T′,H′,W′]`의 잠재 특징이다. 채널이나 프레임 차분을 임의의 2D 이동으로 간주할 수 없다. 영상 decode D와 좌표 추출 M을 거친 C(z)=C_epi(M(D(z)))에 대해 진짜 잠재 접공간 투영을 하려면, 미분 가능하고 국소 선형화가 유효하다는 가정 아래:

$$v_{\mathrm{tan}}=v-J_C^\top(J_CJ_C^\top)^\dagger J_Cv.$$

이 방식에는 Jacobian 연산이 필요하며 비선형 제약·이산 적분·매칭 교체로 인한 drift가 남는다. 학습 없음(train-free)은 입력 gradient 없음(backprop-free)과 다르다. 이 경로는 MVP에서 제외한다.

## 4. 실행 파이프라인: 투영에서 ODE까지

### 단계 A — 생성 모델 없는 기하 검증

1. 합성 3D 점과 알려진 두 카메라로 투영·상대 포즈·F·좌표 변환을 검증한다.
2. 정답 대응점에 알려진 수직/접선 잡음을 넣고 q★의 잔차·최소 이동·멱등성을 검사한다.
3. RealEstate10K 원본 쌍에서 SuperPoint+LightGlue로 대응점을 얻고 제공 F의 오차·coverage를 측정한다. 공식 [LightGlue 구현](https://github.com/cvg/LightGlue)을 사용한다.
4. 불충분한 매칭·순수 회전·잘못된 포즈 방향·crop 오류를 구분한다. 실제 데이터 오차를 바탕으로 평가 허용치를 개발 split에서 고정한다.

A에서 확인하는 hard guarantee는 **좌표 테이블에만** 적용된다. 매칭된 점 자체를 투영한 뒤 그 점으로 점수를 계산하는 결과는 수학 검산이지 영상 개선 증거가 아니다.

### 단계 B — 영상 보정의 실현 가능성

생성 전에 실제 프레임에 알려진 warp를 가한 통제 실험을 만든다. 관측 대응점 q를 q★로 이동시키는 보정장을 구성하고, 보정 영상에서 대응점을 새로 추출한다.

- 초기 구현 후보: sparse 제어점 변위의 보간 + confidence/validity mask + RGB warp. 보간 방식과 정규화 강도는 개발 split에서 고정한다.
- forward splatting(q→q★)과 inverse sampling을 구분한다. inverse grid에 forward 변위를 그대로 넣지 않는다.
- sparse 보간은 비제어점의 기하를 보장하지 않는다. 충돌하는 warp, disocclusion, hole, blur를 기록한다. 미관측 픽셀은 유효 마스크로 남기고 임의 복제를 성공으로 세지 않는다.
- 첫 프레임은 보정 기준으로 고정한다. 독립 쌍 보정은 시간적 깜빡임을 만들 수 있어 다중 프레임에서 별도 확인한다.
- 무보정, encode/decode만, warp만을 비교해 VAE 손실과 기하 보정 효과를 분리한다.

**진행 조건:** 투영 후 영상의 새 대응점에서도 오차가 줄고, coverage·선명도·hole 비율이 개발 기준을 통과해야 한다. 실패하면 ODE 연결로 확대하지 않고 표현/warp 설계를 수정한다.

### 단계 C — 동결 Flow Matching 샘플러 연결

noise-to-data τ∈[0,1]에서 Euler 속도가 vᵢ라면 종단 예측을 다음과 같이 둔다.

$$\hat z_1=z_i+(1-\tau_i)v_i.$$

신뢰도가 충분한 후반 스텝에서만 아래 보정 Q를 적용한다.

```text
prompt (+ initial_frame in I2V) → frozen model → v_i
z_i + (1−τ_i)v_i → predicted clean latent
VAE decode → matching → q★ projection → RGB warp
VAE encode (deterministic posterior mean + model normalization) → z_geo
z_target = z_clean + α_i (z_geo − z_clean)
z_next = z_i + Δτ_i/(1−τ_i) × (z_target − z_i)
```

α=0이면 같은 Euler baseline과 동치여야 한다. τ=1에서 나누지 않으며 마지막 유효 스텝과 종단 보정을 명시적으로 처리한다. 모델의 실제 sigma 격자·prediction type·부호·CFG를 기준 구현에 맞춰 변환한다. 이 갱신은 **종단 예측을 영상에서 보정한 뒤 ODE에 되먹임하는 휴리스틱**이며 잠재 접공간의 정확한 직교 투영은 아니다.

매칭 불능, 과도한 warp, hole, 허용량 초과이면 해당 보정을 건너뛰고 이유를 저장한다. 초기 noisy latent에 직접 매칭하지 않는다. 게이팅 시점·빈도·α·최대 이동량은 개발 split에서 고정한다. 마지막 decode와 재매칭을 통과해야 영상 수준 성공으로 센다. encode/decode와 이후 DiT가 좌표 제약을 보존한다고 가정하지 않는다.

이 경로는 no-grad 추론으로 구성할 수 있지만 역전파가 없다는 이유만으로 저비용이라고 주장하지 않는다. 기존 Y-Flow의 PGD 알고리즘과 동일하지 않으므로 `YFlow-Geo (experimental)`로 구분하고 차이를 기록한다.

## 5. 비교와 평가

주 비교는 **FlowMatch와 YFlow-Geo**다. 추가 항목은 효과를 분해하기 위한 ablation이다.

| 실행 | 목적 |
| :--- | :--- |
| FlowMatch | 동일 백본·조건·seed·Euler 격자의 무보정 기준선 |
| FlowMatch + terminal warp | 최종 영상 후처리만의 효과 |
| FlowMatch + VAE round-trip | 동일 횟수 encode/decode가 주는 영향 |
| YFlow-Geo | 후반 종단 보정을 ODE에 주입 |

같은 최초 노이즈·프롬프트·해상도·프레임 수·CFG를 공유한다. DiT 호출 수와 전체 시간은 별도로 보고한다. 초기 비교에서 HardFlow/SafeFlow/UniConFlow/GuideFlow 추가 구현은 하지 않는다.

### 5.1 Sampson 거리와 실패 처리

최종 영상에서 **새로 추출한** 동차 대응점 p₁,pₖ에 대해:

$$d_S=\frac{(p_k^\top Fp_1)^2}{(Fp_1)_x^2+(Fp_1)_y^2+(F^\top p_k)_x^2+(F^\top p_k)_y^2}.$$

픽셀 좌표 기준 dS는 px², √dS는 px로 표기한다. 매칭에 사용한 해상도가 다르면 좌표와 F를 평가 해상도로 맞춘다. 0에 가까운 분모는 invalid다.

- 주 지표: 클립별 √dS median/p90, 고정 허용치 내 비율, 유효 프레임 쌍 비율, 공간 grid coverage, 대응점 수.
- 인접 쌍과 첫 프레임→후속 프레임 쌍을 사전에 정하고 별도 보고한다. 공유 가시성이 없는 장거리 쌍도 사유를 저장한다.
- 목표 F의 잔차로 outlier를 제거한 후 같은 F의 성능을 재는 순환 평가를 금지한다. descriptor confidence·상호 매칭·사전 마스크로 필터를 고정하고, 기하 필터를 추가한 결과는 보조 지표로만 분리한다.
- 투영기를 평가기로 재사용한 편향을 확인하기 위해 일부 결과는 [LoFTR](https://github.com/zju3dv/LoFTR) 등 별도 matcher와 시각적 대응점 검토로 교차 확인한다.
- 무매칭·coverage 부족을 0 오차로 처리하지 않는다. 전체 클립을 분모로 한 성공률과 유효 클립에서의 오차를 함께 보고한다.

### 5.2 퇴화·품질·비용

정지 영상 복제도 특정 에피폴라 기하를 만족할 수 있다. 정지 첫 프레임 반복과 잘못된 카메라 시퀀스를 음성 대조군으로 평가하여 오라클의 판별력을 먼저 확인한다. F는 translation scale과 선 위의 이동량을 결정하지 못한다.

보조 지표는 움직임 크기, 시간적 flicker, 선명도, warp hole/유효 면적, 프롬프트·장면 보존, 블라인드 육안 비교다. 충분한 시차·매칭이 있는 쌍에서만 상대 회전/이동 방향 추정, track cycle, triangulation cheirality를 진단한다. 이들 역시 완전한 3D 보장을 뜻하지 않는다. I2V에서는 첫 프레임 일치도도 별도 기록한다.

시간/클립, peak VRAM, DiT·VAE·matcher 호출 수, 단계별 소요 시간, 보정 skip 비율을 보고한다. 통계는 클립 단위로 집계하고 같은 클립의 seed를 묶어 paired bootstrap을 사용한다. 성공은 오차 감소와 사전 고정한 coverage·품질 하한을 함께 만족하는 것으로 정의한다.

## 6. 구현 순서와 종료 기준

| 순서 | 산출물 | 다음 단계 진행 조건 |
| :--- | :--- | :--- |
| 1 | 카메라 parser, F/투영 유틸, 합성 검증 | 포즈 방향·좌표 변환·퇴화 처리와 투영 잔차 검증 |
| 2 | 개발 10클립 원본 영상 오라클 리포트 | 유효 매칭 확보, 음성 대조군과 구분 가능 |
| 3 | 통제 warp·재매칭·VAE round-trip 리포트 | 좌표 개선이 영상 개선으로 전달됨 |
| 4 | 공식 파이프라인과 동치인 baseline 1클립 | seed/CFG/시간 부호/latent scaling 검증 |
| 5 | 후반 보정 1클립 → 개발 10클립 | 수치 안정성과 비용·품질 확인 |
| 6 | 설정 동결, test 20×3 → 100×3 | 보정 효과와 실패율을 함께 보고 |

`data/realestate10k.py`는 카메라 parser, 준비된 프레임 로더, Wan causal-VAE 입력 padding/mask, 픽셀 대응점용 에피폴라 잔차와 투영 연산을 제공한다. 좌표 투영의 제약과 영상 재측정 오라클을 분리한다. `BaseConstraint.project_feasible`의 엄밀 보장 계약은 대응점 표현에만 적용할 수 있다. RGB/latent 보정을 동일한 exact projector로 등록하지 않는다. 대응점 matcher, 원본 영상 오라클 평가, RGB warp, scheduler·text-conditioning을 포함한 Wan 생성 baseline, `eval/video_geometry.py`, `eval/y_flow_geo.py`는 아직 구현 대상이다.

현재 `configs/exp_02_video.yaml`은 CLEVRER 설정이며 기존 실행 스크립트는 새 계획을 실행하지 않는다. `configs/realestate10k.yaml`은 개발 데이터 로더 설정이다. `model/wan.py`의 VAE helper는 Wan latent 정규화와 deterministic encode를 적용하지만, transformer 가중치 로드·text conditioning/CFG·scheduler 시간과 부호를 포함한 생성 경로는 아직 기준 구현과 대조하고 연결해야 한다.

첫 실행 목표는 **Wan 대규모 생성이 아니라 RealEstate10K 10클립에서 F 기반 평가가 작동하는지 확인하는 것**이다. 영상 보정 전달이 실패하면 결과를 명확히 남기고, 미분 가능한 latent 최적화 또는 명시적인 depth/reprojection 표현을 후속 설계로 검토한다. 그 경우 backprop-free 가정과 실험 범위를 다시 명시한다.
