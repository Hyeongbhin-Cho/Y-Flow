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
*   `build_dataset(cfg)`: `cfg.data.name`을 읽어 적절한 데이터셋 빌더를 호출하고 `DataBundle` 반환.

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
