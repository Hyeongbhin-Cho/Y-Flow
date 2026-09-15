# Exp-05. Autonomous Driving Trajectory Hard-Constraint 실험

비교 방법:

- **FlowMatch** (무제약 baseline. 같은 pretrained ${v_t^\theta}$)
- **HardFlow**
- **SafeFlow**
- **UniConFlow**
- **GuideFlow**
- **YFlow**

목적: Argoverse 2 Motion Forecasting의 구조화된 궤적과 HD map을 사용하여 미래 차량 궤적을 생성하고, 동일한 Flow Matching backbone에서 각 제약 방법의 안전성·분포 보존성·추론 비용을 비교한다.

카메라 영상과 LiDAR 원본은 사용하지 않는다. 데이터셋이 제공하는 actor state와 vector map만 사용하므로 실제 차량이나 센서 장비는 필요하지 않다.

---

## 1. 실험 범위

### 1.1 데이터셋

- Argoverse 2 Motion Forecasting
- 1차 구현은 전체 데이터가 아니라 소규모 scenario subset으로 파이프라인을 검증한다.
- focal agent 중 `VEHICLE` 궤적을 우선 사용한다.
- 공식 scenario는 11초 길이이며 actor state에 위치, heading, velocity와 observed 여부가 포함된다.
- 원본 scenario와 가공된 cache를 분리하여 원본 데이터를 Git에 포함하지 않는다.

### 1.2 예측 문제

- 관측 구간: 과거 5초
- 생성 구간: 미래 6초
- 샘플링 주기: 10 Hz
- 미래 궤적의 기본 형상: `[T_future, 2]`, ${T_{future}=60}$
- Flow Matching에 전달하는 기본 형상: `[2T_future]`
- 좌표계: 마지막 관측 시점의 focal agent 위치를 원점으로 하고 heading을 ${+x}$축으로 맞춘 ego-centric 좌표계
- 물리 단위: 위치 m, 속도 m/s, 가속도 m/s²

조건 ${c}$는 과거 focal-agent 궤적과 local vector map으로 정의한다. 다만 현재 공통 backbone은 unconditional 입력을 기본으로 하므로 구현을 두 단계로 나눈다.

1. **MVP**: ego-centric 미래 궤적의 무조건부 생성과 속도·가속도 제약을 먼저 검증한다.
2. **Conditional 확장**: 공통 conditional backbone 인터페이스를 합의한 뒤 과거 궤적과 local map을 조건으로 추가한다.

MVP 결과를 scene-conditioned motion forecasting 성능으로 주장하지 않는다.

주변 actor 통계상 50m 이내 완전한 5초 history 수는 90 percentile 14개, 95 percentile 17개였다. 따라서 context는 focal vehicle 1대와 현재 거리순 주변 actor 최대 16개로 고정한다. 부족한 actor 슬롯은 0으로 padding하고 mask로 제외한다.

---

## 2. 데이터 표현과 캐시

한 샘플의 물리 공간 궤적을 다음과 같이 둔다.

$${
p=(q_1,\ldots,q_T),\qquad q_t=(x_t,y_t)\in\mathbb{R}^2
}$$

Flow Matching 입력에서는 이를 ${p\in\mathbb{R}^{2T}}$로 평탄화한다. 정규화 공간의 궤적은 ${z}$로 표기한다.

- 신경망 학습 및 ODE 적분: 정규화 공간 ${z}$
- 제약, 투영, 지표 계산: 역정규화된 물리 공간 ${p}$

가공 캐시의 최소 구성:

```text
datasets/autonomous_driving/exp_05/
├── train.npy
├── eval.npy
├── train_context.npz       # conditional 확장에서 사용
├── eval_context.npz        # conditional 확장에서 사용
└── meta.json
```

`meta.json`에는 최소한 다음을 기록한다.

| 키 | 의미 |
|---|---|
| `dataset` | 데이터셋 이름과 버전 |
| `scenario_ids` | 사용한 scenario 식별자 또는 split manifest 경로 |
| `sample_hz` | 궤적 샘플링 주파수 |
| `history_steps` | 관측 구간 스텝 수 |
| `future_steps` | 생성 구간 스텝 수 |
| `coordinate_frame` | 좌표 변환 정의 |
| `mean`, `std` | 정규화 통계 |
| `v_max` | 최대 속도 기준 |
| `a_max` | 최대 가속도 기준 |
| `seed` | subset 및 평가 재현용 seed |

MVP subset의 780개 train focal-vehicle 궤적에서 0.5초 구간 차분 통계를 측정한 결과, 전체 속도의 99.9 percentile은 24.14 m/s이고 궤적별 최대 가속도의 99.5 percentile은 14.63 m/s²였다. 정상 데이터의 극단값을 거의 제거하지 않는 hard validity envelope로 `v_max=25.0`, `a_max=15.0`을 사용한다. 이는 comfort 기준이 아니며 편안한 주행 가속도는 별도 지표로 평가한다.

Train/eval split은 공식 split을 유지한다. 정규화 통계는 train split에서만 계산하여 eval leakage를 방지한다.

---

## 3. 1차 Hard Constraint

모든 제약은 `data/NOTICE.md`에 따라 물리 공간에서 ${h_j(p)\le 0}$이면 안전하도록 정의한다.

시간 간격을 ${\Delta t=1/10}$초로 두고 속도와 가속도를 다음과 같이 계산한다.

$${
v_t=\frac{q_t-q_{t-1}}{\Delta t},\qquad
a_t=\frac{v_t-v_{t-1}}{\Delta t}
}$$

### 3.1 최대 속도

$${
h_{speed}(p)=\max_t\|v_t\|_2-v_{max}\le 0
}$$

### 3.2 최대 가속도

$${
h_{accel}(p)=\max_t\|a_t\|_2-a_{max}\le 0
}$$

`max`와 norm의 kink는 SafeFlow 장벽에서 직접 사용하지 않는다. SafeFlow용 FMBF는 smooth maximum과 smooth norm으로 별도 구성한다.

### 3.3 도로 경계

Conditional 확장에서 local drivable-area polygon ${\mathcal D(c)}$를 사용할 때 다음 제약을 추가한다.

$${
h_{road}(p;c)=\max_t d_{signed}(q_t,\mathcal D(c))\le 0
}$$

도로 내부에서는 signed distance가 0 이하가 되도록 부호를 맞춘다. MVP에는 scene별 map 조건이 없으므로 이 제약을 사용하지 않는다.

### 3.4 충돌 방지

주변 actor 궤적 ${o_{k,t}}$와 안전거리 ${d_{safe}}$가 있을 때 다음 제약을 고려한다.

$${
h_{collision}(p;c)=\max_{k,t}\left(d_{safe}-\|q_t-o_{k,t}\|_2\right)\le 0
}$$

주변 actor의 미래를 정답으로 사용할 경우 정보 누출이 생기므로, 충돌 제약은 주변 actor의 관측 궤적 또는 별도 예측 결과를 사용하는 2차 확장으로 둔다.

---

## 4. 제약 연산자 구현 계획

`AutonomousDrivingConstraint`는 `data.base.BaseConstraint`를 상속한다.

| 메서드 | MVP | Conditional 확장 |
|---|:---:|:---:|
| `h(p)` | speed, acceleration | road, collision 추가 |
| `cost(p)` | 위반량의 smooth squared hinge | 동일 |
| `project_feasible(p, buffer)` | 속도·가속도 제한 궤적 투영 | 도로·충돌 포함 |
| `project_physical(p)` | kinematic smoothing | lane/drivable-area projection |
| `estimate_lipschitz(p)` | 보수적인 상한 또는 유한차분 | map projection 포함 재추정 |
| `energy`, `energy_grad` | kinematic violation energy | map·collision energy 추가 |
| `progress(p)` | 궤적의 정규화된 시간 진행도 | route progress 검토 |
| `get_fmbf()` | smooth speed/acceleration barrier | smooth map/collision barrier 추가 |

`project_feasible`는 이미 안전한 궤적을 불필요하게 이동시키지 않아야 하며, 투영 후 모든 활성 제약이 ${-buffer}$ 이하인지 다시 검사한다.

---

## 5. 비교 프로토콜

1. 동일한 train/eval cache와 정규화 통계를 사용한다.
2. 동일한 FlowMatch backbone을 모든 training-free 방법이 공유한다.
3. 동일한 평가 seed와 초기 가우시안 노이즈 ${x_0}$를 사용한다.
4. 동일한 ODE step 수를 사용한다.
5. 모든 제약과 지표는 물리 공간에서 계산한다.
6. terminal projection 전후 지표를 함께 기록하여 사후 필터 의존성을 구분한다.
7. MVP에서는 모든 방법이 speed·acceleration 제약만 사용한다.
8. Conditional 확장에서는 모든 방법에 동일한 history·map context를 제공한다.

---

## 6. 평가 지표

### 6.1 안전성

- `safe_ratio` ↑
- `speed_viol_rate`, `speed_viol_mean` ↓
- `accel_viol_rate`, `accel_viol_mean` ↓
- Conditional 확장: `road_viol_rate`, `collision_rate` ↓
- terminal projection 전후 안전성 및 수정 비율

### 6.2 분포 및 궤적 품질

- MMD ↓: 생성 궤적과 eval 궤적의 분포 차이
- 속도·가속도 분포 거리 ↓
- 경로 길이 분포 차이 ↓
- endpoint 분포 차이 ↓

Conditional 확장에서는 다음 예측 지표를 추가한다.

- minADE ↓
- minFDE ↓
- Miss Rate ↓

### 6.3 계산 비용

- 전체 추론 시간
- 1,000개 궤적당 추론 시간
- peak GPU memory

MMD만으로 물리적 타당성이나 조건부 예측 정확도를 주장하지 않는다. Safety, 분포 지표, 예측 지표를 함께 해석한다.

---

## 7. 구현 순서

1. 공식 데이터의 소규모 subset과 split manifest 준비
2. focal vehicle 궤적 추출 및 ego-centric 변환
3. `DataBundle` 캐시 생성·재로드 확인
4. speed·acceleration oracle과 단위 테스트 구현
5. 실제 eval 데이터의 oracle 통계 확인
6. FlowMatch MVP 학습 및 무제약 baseline 평가
7. 제약 방법 공통 평가
8. conditional backbone 인터페이스 설계
9. local vector map과 road constraint 추가
10. 필요 시 collision constraint 추가

---

## 8. MVP 성공 기준

1. 데이터 subset을 동일하게 재생성할 수 있다.
2. 데이터와 제약 출력의 배치 차원이 일치한다.
3. `cost`에 대한 PyTorch Autograd가 작동한다.
4. 의도적으로 만든 과속·급가속 궤적을 제약 오라클이 탐지한다.
5. `project_feasible` 이후 speed·acceleration 제약을 만족한다.
6. FlowMatch와 제약 방법들이 동일한 backbone 및 ${x_0}$에서 평가된다.
7. 결과에 terminal projection 의존성이 명시된다.

---

## 9. 범위 밖

- 실제 차량 주행 및 실차 검증
- 카메라·LiDAR 원본의 end-to-end 학습
- 센서 융합 및 객체 탐지
- 폐루프 차량 제어 성능 주장
- 주변 actor 미래 정답을 추론 입력으로 사용하는 평가

---

## 10. 한 줄

Exp-05의 1차 목표는 **Argoverse 2 차량 미래 궤적을 물리 좌표에서 생성하면서 속도·가속도 hard constraint를 만족하는지 공통 Flow Matching backbone으로 비교하는 것**이며, 과거 궤적·지도 조건과 도로·충돌 제약은 공통 conditional 인터페이스가 준비된 뒤 확장한다.

## MoFlow teacher

`moflow` is an Argoverse 2 adaptation of MoFlow's conditional K-shot flow-matching
teacher. It predicts the focal vehicle's 6-second future from its 5-second history
and up to 16 nearby actors. The compact GRU scene encoder replaces MoFlow's
human-trajectory dataset-specific encoder, so results must be reported as a
"MoFlow-style AV2 adaptation", not as an exact reproduction of the official model.

Train and evaluate:

```bash
python main.py moflow --config configs/exp_05_autonomous_driving.yaml --mode train --run_name exp_05_moflow --device cuda
python main.py moflow --config configs/exp_05_autonomous_driving.yaml --mode eval --run_name exp_05_moflow --device cuda
```

For a smoke test, add `--train.steps 2 --train.batch_size 4 --sample.n_steps 2`.

### Shared-backbone constraint comparison

The five constraint methods below load the same
`runs/<run_name>/moflow/last.pt` checkpoint and reuse the same initial noise.

```bash
for method in hardflow safeflow uniconflow guideflow yflow; do
  python main.py "$method" --config configs/exp_05_autonomous_driving.yaml \
    --mode eval --run_name exp_05_moflow --device cuda
done
```

HardFlow, SafeFlow, UniConFlow, GuideFlow, and YFlow are adapted to the AV2
conditional K-shot state. SafeFlow uses differentiable speed/acceleration
certificates plus exact terminal projection; GuideFlow uses inference-time
energy refinement. The output `adaptation` field records this scope.

For YFlow, the domain physical operator is a three-pass non-expansive temporal
smoother anchored at the observed origin while retaining the predicted endpoint.
The hard feasibility projection remains a separate speed/acceleration operation.

Every constrained evaluation also regenerates the matched unconstrained MoFlow
sample from the same initial noise and reports `intervention_ADE_m`,
`intervention_FDE_m`, and `intervention_max_m`. These quantify how much safety
guidance changes the teacher prediction instead of hiding terminal-projection cost.

The AV2 feasibility projection uses a forward kinematic tracker at 10 Hz. It clips
per-step velocity and velocity change before reconstructing positions, preserving
the original path more closely than uniformly shrinking the entire trajectory.

For repeated sampling evaluations, set `moflow.backbone_run_name` to the trained
run while changing `run_name` and `seed`. Aggregate completed runs with
`python -m eval.summarize_autonomous --runs <run1> <run2> ...`.

Run the registered five-seed YFlow iteration and terminal-projection ablations:

```bash
bash run_exp_05_ablations.sh
```

The script evaluates YFlow with `max_iter` in `{1, 3, 5, 10, 20}` and evaluates
SafeFlow, UniConFlow, and YFlow with their terminal projection/refinement disabled.
Every run loads the frozen `exp_05_moflow` checkpoint and writes a separate summary.

## 11. 5-seed 결과와 최종 설정

평가는 AV2 validation의 149개 scenario, scenario당 6개 후보, Euler 100 step,
sampling seed 0~4에서 수행했다. 모든 방법은 동일한 MoFlow teacher checkpoint를
사용한다. 아래 값은 5회 평균 ± 표본 표준편차다.

| 방법 | minADE ↓ | minFDE ↓ | safe ratio ↑ | inference time ↓ |
|---|---:|---:|---:|---:|
| MoFlow | 4.072 ± 0.091 | **8.788 ± 0.246** | 0.000 | **0.169 ± 0.004 s** |
| GuideFlow | 3.998 ± 0.092 | 8.824 ± 0.264 | 0.007 ± 0.003 | 0.394 ± 0.007 s |
| UniConFlow | **3.820 ± 0.079** | 8.963 ± 0.404 | **1.000** | 0.498 ± 0.016 s |
| SafeFlow | 3.823 ± 0.077 | 8.994 ± 0.378 | **1.000** | 1.135 ± 0.085 s |
| YFlow (`max_iter=1`) | 3.854 ± 0.080 | **8.879 ± 0.381** | **1.000** | 4.280 ± 0.170 s |
| HardFlow | 9.845 ± 0.210 | 18.167 ± 0.754 | **1.000** | 23.417 ± 0.488 s |

굵은 minFDE는 전체 최고와 safe ratio 1인 방법 중 최고를 각각 표시한다.
YFlow는 안전한 방법 중 minFDE와 top-1 정확도가 가장 좋지만, minADE는
UniConFlow가 가장 좋고 YFlow의 추론 비용도 더 크다.

### 11.1 Terminal projection ablation

| 방법 | terminal off safe ratio ↑ | minADE ↓ | minFDE ↓ |
|---|---:|---:|---:|
| SafeFlow | 0.000 ± 0.000 | 3.992 ± 0.091 | 8.822 ± 0.266 |
| UniConFlow | 0.000 ± 0.000 | 4.015 ± 0.090 | 8.806 ± 0.260 |
| YFlow | **0.832 ± 0.008** | **3.820 ± 0.079** | 8.880 ± 0.382 |

SafeFlow와 UniConFlow의 최종 100% 안전성은 이 구현에서 terminal projection에
의존한다. YFlow는 마지막 refinement를 제거해도 평균 83.2%를 만족하며, terminal
refinement는 작은 정확도 변화로 이를 100%까지 높인다. 따라서 최종 안전성을
guidance만의 효과로 표현하지 않고 terminal projection 사용 여부를 함께 보고한다.

### 11.2 YFlow 반복 횟수 ablation

`max_iter={1,3,5,10,20}`은 minADE 3.8533~3.8536, minFDE
8.8712~8.8785, safe ratio 1.0으로 사실상 같은 결과를 보였다. 추론 시간은
`max_iter=1`에서 4.280초, `max_iter=20`에서 12.643초였다. 이에 따라 Exp-05의
최종 기본값은 `max_iter=1`, `terminal_refinement=true`로 정한다.
