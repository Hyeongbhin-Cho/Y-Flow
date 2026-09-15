# data 패키지 (data/)

이 패키지는 데이터셋 생성/로드/캐싱 및 도메인 제약(Constraint) 정의, 범용 CFMBF Active-Set QP 솔버를 통합 제공합니다.

매 학습마다 점을 다시 뽑지 않으며, 모든 알고리즘(Flow Matching 및 training-free/guided 제약 방법)이 동일한 데이터와 제약 함수 $h$를 일관되게 공유하도록 `train.npy` / `eval.npy` / `meta.json`과 제약 객체를 `DataBundle`로 패키징하여 제공합니다.

---

## 1. 관련 README 및 가이드 링크
*   [Y-Flow 메인 설명 문서](../README.md)
*   [제약조건 개발 및 협업 가이드 (NOTICE.md)](NOTICE.md)
*   [Train 패키지 설명 문서](../train/README.md)
*   [Eval 패키지 설명 문서](../eval/README.md)

---

## 2. 파일 목록 및 요약
* `base.py`:
  * `BaseConstraint`: 도메인 제약 추상 베이스 클래스 ($h$, $C$, $P$, 립시츠 추정, 에너지, 진행도 등).
  * `DataBundle`: 데이터 텐서와 바인딩된 제약 객체를 담는 컨테이너.
  * 데이터셋 레지스트리: `@register_dataset`, `build_dataset(cfg)`을 통한 동적 라우팅.
  * 범용 CFMBF Active-Set QP 솔버: `solve_composite_fmbf`, `solve_single_fmbf`, `barrier_gain`.
  * 유틸리티: `normalize_tensor`, `denormalize_tensor`, `build_anchor_vocabulary`.
* `swiss_roll.py`: 2D spiral 데이터 생성, 캐시 저장/로드, `SwissRollConstraint`, `SwissRollFMBF`.
* `clevrer.py`: CLEVRER MP4 지연 로딩, 원본 객체·운동·충돌 주석, `VideoDataBundle`, `collate_clevrer`.
* `clevrer_state.py`: 인식용 packed 상태 $S$, `CLEVRERStateConstraint` ($h$, $C$, $P$, `project_feasible`, 3장벽 FMBF), `CLEVRERRecognitionDataset`.
* `../datasets/swiss_roll/default/`: Exp-01 기본 dump (`train.npy`, `eval.npy`, `meta.json`).

---

## 3. 세부 명세

### base.py

#### BaseConstraint
*   모든 도메인 제약의 추상 기반 클래스.
*   `h(p)`: 안전 조건 ($h(p) \le 0$). 배치 텐서 지원 및 PyTorch Autograd 미분 가능.
*   `C(p)`: 비용 함수 ($C(p) = \frac{1}{2}\sum \max(0, h_i(p))^2$). HardFlow, YFlow 최적화에 사용.
*   `project_physical(p)`: 물리적 매니폴드 투영 $P(p)$ (기본값: 항등 사상 $P(p) = p$).
*   `estimate_lipschitz(p)`: 제약 기울기/손실의 립시츠 상수 추정 (기본값: $1.0$).
*   `energy(p)`: 가이드 흐름용 에너지 함수 (기본값: $C(p)$).
*   `progress(p)`: 시퀀스/궤적 진행도 측정.
*   `command_bins(p)`: 이산 모드/조건 라벨 분할 (GuideFlow CFG 학습에 사용).

#### DataBundle
*   `train_data`, `eval_data`: `torch.Tensor` 데이터셋.
*   `meta`: 데이터셋 메타데이터 딕셔너리.
*   `constraint`: 해당 데이터셋에 바인딩된 `BaseConstraint` 인스턴스.
*   딕셔너리 호환 인터페이스 (`bundle["train"]`, `bundle["constraint"]` 등) 제공.

#### 동적 라우터 (`register_dataset` / `build_dataset`)
*   `@register_dataset(name)` 데코레이터로 신규 데이터셋 팩토리 등록.
*   `build_dataset(cfg)`: `cfg.data.name`을 읽어 데이터셋 빌더를 호출한다. Swiss roll은 `DataBundle`, CLEVRER는 `VideoDataBundle`을 반환한다.

#### 범용 Active-Set QP 솔버 (CFMBF)
*   SafeFlow 및 안전 필터링에 사용되는 Active-Set QP 솔버.
*   배치 단위로 KKT 최적성 조건과 active constraint 조합을 순회하며 정확한 닫힌형(closed-form) 해를 계산.
*   비활성/실패 시 slack 최소화 및 폴백 처리 내장.

---

### swiss_roll.py

#### SwissRollConstraint
*   `BaseConstraint`를 상속하여 2D Swiss Roll 기하학적 제약 구현.
*   내경/외경 반경 경계 ($r \in [R_{min}, R_{max}]$), 나선 폭 경계 ($|r - a \theta| \le w$), 양 끝단 경계 등 5개 부등식 제약 $h(p) \le 0$ 제공.
*   매니폴드 투영 `project_physical` 및 $C(p)$ 헤시안 상한 기반 `estimate_lipschitz` 제공.

#### SwissRollFMBF
*   Swiss Roll 도메인 전용 composite SafeFlow 솔버.
*   `base.py`의 `solve_composite_fmbf`를 호출하여 최적 속도 필터링 수행.

#### build_swiss_roll
*   `cfg.data.cache_dir`에 캐시가 있고 `regenerate=false`면 로드, 없으면 생성 후 저장.
*   생성된 텐서들과 `SwissRollConstraint`를 묶어 `DataBundle`로 반환.

캐시 디렉터리:

```
datasets/swiss_roll/default/
├── train.npy
├── eval.npy
└── meta.json
```

---

### clevrer.py

#### CLEVRERDataset

* `CLEVRERDataset(root, split="validation")`: MP4를 샘플 접근 시 디코딩한다. 기본은 원본 전체 프레임과 해상도이며 반환 영상은 float32 `[C,T,H,W]`, `[-1,1]`이다.
* `n_frames`, `fps`, `start_time`: 초 단위 시각에 가장 가까운 원본 프레임을 선택한다. 짧은 영상은 반복·padding하지 않고 오류를 낸다. `frame_indices`, 요청 시각 `frame_times`, 실제 시각 `source_frame_times`를 반환한다.
* `height`, `width`: 둘 다 지정하면 비율 유지 resize + 검정 letterbox를 적용한다. `valid_region`과 `spatial_transform`으로 padding과 좌표 변환을 확인할 수 있다.
* `annotation`: 원본 영상 전체의 JSON을 그대로 보존한다. `frame_annotations`는 선택한 원본 프레임에 해당하는 상태 주석이다. 충돌의 `frame_id`는 원본 인덱스이며, 클립 외 사건도 `annotation`에 남아 있다.
* 위치·속도 등 주석은 원본 시뮬레이터 좌표다. 이미지 픽셀 좌표로 변환하거나 객체를 검출하지 않는다.
* train/validation은 주석이 필수이며 누락·잘못된 영상 대응은 오류로 처리한다. test는 `annotation=None`, `frame_annotations=None`이다.
* `load_questions=True`는 별도 다운로드한 `questions/{split}.json`을 읽는다.

#### VideoDataBundle / build_clevrer

`build_dataset(cfg)`에 `data.name: clevrer`를 지정한다. `bundle.eval`은 지연 로딩 Dataset이며, `train_split`을 지정한 경우에만 `bundle.train`을 만든다. `n_eval`/`n_train`은 scene ID 오름차순의 최대 로드 개수다. 전체 split 다운로드 크기를 줄이는 옵션은 아니다.

`bundle.meta`와 `bundle.meta_dict`는 동일한 메타데이터를 제공하며 `bundle["eval"]`도 지원한다. 현재 물리 평가기는 구현하지 않아 `constraint=None`이다. VAE latent인 `eval_z`나 Swiss-roll 정규화 통계는 만들지 않는다. 기존 2D train/eval 루프에 그대로 넣는 인터페이스가 아니며, 비디오 생성·평가 경로는 별도 연동 대상이다.

설치 및 데이터 준비:

```bash
conda activate yflow
python -m pip install av
python scripts/setup_clevrer.py --splits validation
```

실험 설정으로 로드:

```python
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import build_dataset, collate_clevrer

cfg = OmegaConf.load("configs/exp_02_video.yaml")
bundle = build_dataset(cfg)
loader = DataLoader(bundle.eval, batch_size=1, collate_fn=collate_clevrer)
batch = next(iter(loader))
print(batch["video"].shape)  # [1, 3, 33, 480, 832]
print(batch["annotation"][0]["object_property"])
```

객체 수와 이벤트 수가 영상마다 다르므로 `collate_fn=collate_clevrer`를 사용한다. 영상 크기·길이가 서로 다른 원본을 그대로 로드하면 `batch_size=1`을 사용하거나 동일한 출력 규격을 지정한다.

직접 원본 로드:

```python
from data import CLEVRERDataset

dataset = CLEVRERDataset("datasets/clevrer", split="validation")
sample = dataset[0]
video = sample["video"]
annotation = sample["annotation"]
```

다운로드 옵션과 저장 구조는 [scripts/README.md](../scripts/README.md)를 참고한다.

#### CLEVRERRecognitionDataset / CLEVRERStateConstraint

`data.name: clevrer_recognition`. 클립 $V$와 packed 상태 $S$를 같이 반환한다. $S$는 슬롯 $K=6$, 33프레임 world 좌표(속성 원-핫, 가시성, 위치·속도, 충돌 상삼각). 제약 $h(S)\le 0$은 어휘·개수·테이블·운동학·충돌 접촉이다. train split이 없으면 `python scripts/setup_clevrer.py --splits train`.

### RealEstate10K (Exp-02 개발 데이터)

`realestate10k.py`는 준비된 PNG만 지연 로드한다. [setup 스크립트](../scripts/setup_realestate10k.py)를 먼저 실행한다. 학습이나 다운로드는 로더에서 수행하지 않는다.

```python
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from data import build_dataset, collate_realestate10k

bundle = build_dataset(OmegaConf.load("configs/realestate10k.yaml"))
loader = DataLoader(bundle.eval, batch_size=1, collate_fn=collate_realestate10k)
sample = next(iter(loader))
```

개별 sample의 `video`는 관측된 RGB 프레임 float32 `[3,T,H,W]`, 범위 `[-1,1]`이다. `wan_video`는 같은 클립을 Wan2.1 causal VAE의 `T=1+4k` 계약에 맞춰 마지막 관측 프레임을 반복한 `[3,1+4k,H,W]` 입력이다. `wan_frame_mask`와 `wan_timestamps_us`는 반복 padding을 구분한다. 설정 영상 크기는 VAE의 8배 축소와 Transformer의 2×2 patch를 고려해 높이·너비 모두 16의 배수여야 한다. 배치에서는 `collate_realestate10k`를 사용하면 각 tensor가 batch 축으로 쌓인다. Wan VAE encode는 `WanVelocityNet.encode_video(batch["wan_video"])`로 수행한다.

`K`, `K_normalized`, `world_to_camera`, `spatial_transform`은 float64이며 `timestamps_us`(선택 포즈), `requested_timestamps_us`, `actual_timestamps_us`(영상 PTS), `camera_indices`, `source_sizes_wh`도 반환한다.

`pair_indices`는 첫 프레임 기준 쌍과 인접 쌍의 합집합이다. 같은 순서의 `fundamental_matrices`는 픽셀 좌표에 적용하며 Frobenius norm으로 정규화한다. `fundamental_valid=False`인 순수 회전/0 translation 쌍은 F를 0으로 반환하므로 **반드시 마스킹**해야 한다. 이 유효성은 대수적 판정이며 작은 시차·정적 장면·매칭 성공을 의미하지 않는다. `valid_region`은 crop/padding이 없는 준비 프레임 전체이며 정적 배경/가림 마스크가 아니다.

`VideoDataBundle`을 반환하고 `train=None`이다. `constraint`는 sample별 F를 쓰는 `RealEstate10KEpipolarConstraint`다. 대응점 좌표 `p1,p2`와 대응하는 F를 전달해 `constraint.h(p1,p2,F,valid=...)`로 픽셀 단위 선 잔차를 얻고, `constraint.project_feasible(...)`로 목표 점을 에피폴라 선에 최소 거리 투영할 수 있다. `h<=0`은 지정 허용 오차 안을 뜻한다. 퇴화/무효 쌍은 `valid=False`로 전달하며 결과에서 제외된다. 이는 `BaseConstraint`의 전역 상태공간 제약이 아닌 대응점 공간 연산이다. RGB 프레임·Wan latent에 직접 투영할 수 없으며, 정적 장면/매칭 성공도 데이터만으로 보장하지 않는다. 개발 train 10클립 로더 설정은 [configs/realestate10k.yaml](../configs/realestate10k.yaml)이며, 기존 CLEVRER 생성 설정과 구분된다.

공식 포즈 형식: [RealEstate10K](https://google.github.io/realestate10k/download.html). Timestamp + intrinsics 4개 + reserved 2개 + world-to-camera 3×4 행렬, 총 19개 열을 파싱한다.
