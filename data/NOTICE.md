# data 패키지 제약조건(Constraints) 개발 및 협업 가이드 (NOTICE)

본 문서는 `data/` 패키지에 정의된 **도메인 제약조건(Constraints)**의 수학적 정의, 각 메서드의 구체적인 기능과 역할, 만족해야 하는 필수 수학적·수치적 조건, 그리고 다양한 제약 흐름 생성 알고리즘(Flow Matching, HardFlow, YFlow, SafeFlow, UniConFlow, GuideFlow)과의 연계 요구사항을 상세히 설명합니다.

연구자 및 개발자마다 제약조건을 다루는 관점(경계 부등식, 페널티 함수, 매니폴드 투영, 배리어 함수 등)이 서로 다르므로, 코드베이스 내 일관된 동작과 새로운 데이터셋 추가 시의 혼선을 방지하기 위해 본 가이드를 필히 준수합니다.

---

## 1. 제약조건 시스템 핵심 설계 원칙

1. **데이터셋-도메인 제약 일체형 번들링 (`DataBundle`)**
   * 안전 영역(Safe set), 경계면, 물리적 매니폴드는 모델이나 평가 스크립트의 속성이 아니라 **데이터셋 도메인이 본질적으로 갖는 고유한 기하학적·물리적 성질**입니다.
   * `data.base.build_dataset(cfg)`은 데이터 텐서(`train`, `eval`), 정규화 통계량(`mean`, `std`), 그리고 해당 데이터셋에 바인딩된 제약 인스턴스(`bundle.constraint`)를 하나의 `DataBundle`로 묶어 반환합니다.
   * 이를 통해 모든 알고리즘이 동일한 제약 오라클(Ground-Truth Oracle)을 공유하여 공정한 벤치마크를 보장합니다.

2. **좌표계 분리 원칙 (Physical Space vs. Normalized Space)**
   * **물리 공간 (Physical Space $p \in \mathbb{R}^D$)**: 데이터의 원래 물리 단위 및 기하학적 형상이 유지되는 좌표계입니다. **모든 제약 함수($h, C, P, E$ 등)는 물리 공간 $p$에서 정의되고 계산됩니다.**
   * **정규화 공간 (Normalized Space $z \in \mathbb{R}^D$)**: Flow Matching 신경망 $v_t^\theta(z)$이 학습되고 추론되는 영평균·단위분산 좌표계입니다.
   * 알고리즘 내부에서 제약을 적용하거나 평가할 때는 반드시 $p = \text{denormalize}(z, \text{mean}, \text{std})$로 변환한 뒤 제약 연산자를 호출해야 합니다.

---

## 2. BaseConstraint 인터페이스 명세 및 메서드별 세부 요구사항

모든 도메인 제약 클래스는 `data.base.BaseConstraint` 추상 클래스를 상속하며, 아래의 기능별 역할을 수행합니다.

### 2.1. $h(p)$ : 안전 제약 부등식 함수 (Hard Constraint)

* **기능**:
  * 상태 $p$에 대해 $K$개의 부등식 제약 $h_j(p) \le 0$의 위반 여부와 위반량을 계산합니다.
  * 딕셔너리 형태로 각 제약 조건의 이름과 텐서/배열을 매핑하여 반환합니다 (예: `{'tube': ..., 'core': ..., 'box': ...}`).
* **역할 (Consumers)**:
  * `eval/metrics.py`: 평가 오라클로 사용되어 개별 제약 및 전체 제약에 대한 위반율(`*_viol_rate`), 평균 위반량(`*_viol_mean`), 완전 안전 비율(`all_safe`)을 측정.
  * `eval/unicon_flow.py`: Pre-specified Time Zeroing Function (PTZF) certificate 조건 $${L_f h_j(p_t) + \alpha(t, h_j(p_t)) \le 0}$$의 잔차(Residual) 계산 및 QP 보정 대상.
  * `data/base.py` (`build_anchor_vocabulary`): Farthest Point Sampling (FPS) 수행 전 안전 영역 내부 점들만 필터링 (${\max_j h_j(p) \le 0}$).
* **만족해야 하는 필수 조건**:
  1. **안전 영역 정의**: 안전 집합(Feasible Set) $\mathcal{C}$는 모든 제약이 0 이하인 점들의 교집합입니다:
     $${\mathcal{C} = \{p \in \mathbb{R}^D \mid \forall j \in \{1, \dots, K\}, h_j(p) \le 0\}}$$
  2. **부호 규약**:
     * $h_j(p) \le 0$ : 제약 만족 (안전 영역 내부 및 경계면).
     * $h_j(p) > 0$ : 제약 위반 (안전 영역 외부). $h_j(p)$의 크기는 위반 거리에 비례해야 함.
  3. **미분 가능성 (Differentiability)**:
     * `torch.Tensor` 입력 시 PyTorch Autograd 계산 그래프가 끊기지 않아야 합니다 (`torch.autograd.grad` 지원).
     * 경계면 근방에서 구분적으로 연속 미분 가능($C^1$ piecewise smooth)해야 수치적 지터링이 발생하지 않습니다.
  4. **배치 차원 지원**:
     * 입력 형상 `(..., D)`에 대해 반환되는 각 딕셔너리 값은 `(...)` 형상을 가져야 합니다.

---

### 2.2. $C(p)$ (`cost`) : 제약 위반 비용 함수 (Constraint Violation Cost)

* **기능**:
  * 안전 영역 $\mathcal{C}$를 벗어난 정도를 나타내는 단일 스칼라 비용 함수 $C(p)$를 계산합니다.
  * 기본 정의:
    $${C(p) = \frac{1}{2} \sum_{j=1}^K \max(0, h_j(p))^2}$$
* **역할 (Consumers)**:
  * `eval/hard_flow.py`: 샘플 생성 과정의 매 적분 스텝마다 경사 하강법 $${p \leftarrow p - \eta \nabla C(p)}$$를 수행하여 위반을 최소화.
  * `eval/y_flow.py`: Flow Matching 속도장에 보정 가이던스 항 $${- \mu_t \nabla C(p_t)}$$를 결합하여 궤적을 안전 영역으로 유도.
* **만족해야 하는 필수 조건**:
  1. **최소성 및 무영향성**:
     * $p \in \mathcal{C} \iff C(p) = 0$.
     * $p \notin \mathcal{C} \iff C(p) > 0$.
     * 안전 영역 내부에서는 ${\nabla C(p) = 0}$이어야 하므로, 기저 생성 모델의 자연스러운 속도장(Flow)을 전혀 왜곡하지 않아야 합니다.
  2. **경계면 연속성 ($C^1$ 연속)**:
     * 경계면($h_j(p) = 0$)에서 $C(p)$뿐만 아니라 그 기울기 $\nabla C(p)$도 0으로 연속 수렴해야 합니다. 불연속 기울기는 경사 하강 시 발산 또는 궤적 진동을 유발합니다.
  3. **미분 가능성**:
     * `cost(p)`는 배치 단위로 `torch.autograd.grad(cost.sum(), p)`가 가능하도록 순수 PyTorch 텐서 연산으로 구현되어야 합니다.

---

### 2.3. $P(p)$ (`project_physical`) : 물리적 매니폴드 투영 연산자 (Physical Prior Projection)

* **기능**:
  * 임의의 점 $p$를 데이터가 존재하는 이상적인 기하학적/물리적 저차원 매니폴드 $\mathcal{M}_{\text{phys}}$ (예: 1D Swiss roll 나선 중심선, 기구학 매니폴드 등) 상의 가장 가까운 점 $P(p)$로 투영합니다.
* **기본값 (Default)**:
  * **항등 사상 $${P(p) = p}$$**: 도메인 고유의 물리적 매니폴드 사전 정보가 명시되지 않은 경우, 기본적으로 입력을 그대로 반환합니다.
  * $P(p) = p$일 때 YFlow는 $z_{\text{phys}} = z_{\text{raw}}$가 되어, PGD 목적식이 비용 함수와 Flow Matching 원래 궤적 보존 항의 합 $${C(p) + \frac{1}{2}(\lambda + \mu) \|z - z_{\text{raw}}\|^2}$$로 환원됩니다. 즉, 별도의 매니폴드 수식이 없어도 HardFlow와 같이 제약 완화/안전 투영 방식으로 매끄럽게 동작(Graceful Degradation)하며, 미구현에 따른 `NoneType` 오류를 원천 차단합니다.
* **역할 (Consumers)**:
  * `eval/y_flow.py`: 물리적 오차 잔차 ${\|p - P(p)\|}$를 평가하고, 매니폴드 도달 신뢰도에 따라 가이던스 스텝 크기 $\mu_t$를 적응적으로 조절.
* **만족해야 하는 조건**:
  1. **멱등성 (Idempotence)**:
     $${P(P(p)) = P(p) \in \mathcal{M}_{\text{phys}}}$$ ($P(p) = p$인 경우 $P(P(p)) = P(p) = p$로 자명하게 만족).
  2. **오버라이드 권장**:
     * Swiss Roll의 나선 곡면처럼 명시적인 저차원 매니폴드가 존재하는 도메인에서는 해당 매니폴드로의 투영 수식을 오버라이드하여 가이던스 성능을 극대화할 수 있습니다.

---

### 2.4. `project_feasible(p, buffer)` : 실행 가능 집합 엄밀 투영자 (Feasible Projection)

* **기능**:
  * 임의의 점 $p$를 안전 집합 $\mathcal{C}$의 내부(또는 경계로부터 `buffer`만큼 안쪽)로 직접 투영합니다.
* **역할 (Consumers)**:
  * `eval/hard_flow.py`, `eval/unicon_flow.py`, `eval/y_flow.py`: 최종 터미널 시점($t=1$)에서 잔여 미세 위반을 즉각 제거하여 100% 안전성을 보장하는 사후 필터(Terminal Projection).
  * `data.base.build_anchor_vocabulary`: 안전한 앵커 풀이 부족할 경우 점들을 안전 집합으로 보정.
* **만족해야 하는 조건**:
  1. **엄격한 안전성**:
     * 투영된 점 $p_{\text{proj}} = \text{project\_feasible}(p, \text{buffer})$는 모든 제약에 대해 $${h_j(p_{\text{proj}}) \le -\text{buffer} \le 0}$$을 엄격히 만족해야 합니다.
  2. **최소 이동성 (Proximity)**:
     * 이미 안전한 점($p \in \mathcal{C}$)에 대해서는 $p$를 가능한 한 이동시키지 않아야 합니다 (${\|p_{\text{proj}} - p\| \approx 0}$).

---

### 2.5. `estimate_lipschitz(p, eps)` : 국소 립시츠 상수 추정기

* **기능**:
  * 물리 투영 연산자 $P(p)$ 또는 비용 기울기 $\nabla C(p)$의 국소 립시츠 상수 $L$ (즉, 헤시안 $\nabla^2 C(p)$의 스펙트럼 노름 상한)을 추정합니다.
* **기본값 (Default)**:
  * **$L = 1.0$**: 기본 항등 사상 $P(p) = p$의 립시츠 상수는 1이므로, 기본 구현체는 $1.0$을 반환합니다. 이는 게이팅 조건 $L \le 1.0 + \delta$를 안정적으로 만족시킵니다.
* **역할 (Consumers)**:
  * `eval/y_flow.py`: 최적화 경사 하강 스텝 크기 $\eta \le \frac{1}{L}$를 안정적으로 설정하여 PGD 가이던스 업데이트의 수렴성을 수학적으로 보장.
* **만족해야 하는 조건**:
  1. **상한(Upper Bound) 추정**:
     * 경사 하강법에서 스텝 크기가 $\frac{2}{L}$를 초과하면 발산하므로, 추정된 $L$은 실제 국소 곡률보다 작아서는 안 되며(과소평가 금지), 보수적인 상한값이어야 합니다.
  2. **해석적 상한 권장**:
     * 기하학적 곡률의 최대 상한이 해석적으로 유도 가능한 경우 해석적 값을 사용하고, 어려운 경우 유한 차분 섭동(Finite perturbation directional derivative)으로 계산합니다.

---

### 2.6. $E(p)$ (`energy`) 및 `energy_grad(p)` : 에너지 기반 제약 함수

* **기능**:
  * GuideFlow의 Reward-Free Energy (RFE) 및 가이던스에 사용하는 스칼라 잠재 에너지 $E(p)$ 및 그 해석적 기울기 $\nabla_p E(p)$를 계산합니다.
* **역할 (Consumers)**:
  * `train/guide_flow.py`: RFE 손실 함수 $${L_{\text{RFE}} = \mathbb{E}[\|v_t^\theta(z_t) - v_t(z_t) + \lambda \nabla_z E(p_t)\|^2]}$$ 학습에 사용.
  * `eval/guide_flow.py`: 샘플링 시 속도장에 에너지 유도 벡터 $- \nabla E(p)$를 더해 제약 영역으로 가이던스.
* **만족해야 하는 조건**:
  1. 안전 영역 $\mathcal{C}$ 내부에서는 $E(p) = 0$ 및 $\nabla E(p) = 0$.
  2. 안전 영역 경계에서 벗어날수록 외곽을 향해 가파르게 증가하는 단조 볼록 특성을 지녀야 함.

---

### 2.7. `progress(p)` 및 `command_bins(p, n_commands)` : 진행도 및 모드 분류

* **기능**:
  * 점 $p$가 전체 매니폴드 또는 궤적 시퀀스 상에서 얼만큼 진행되었는지 정규화된 스칼라 $[0, 1]$를 계산하고, 이를 $n_{\text{commands}}$개의 이산 구간 라벨로 분류합니다.
* **역할 (Consumers)**:
  * `train/guide_flow.py`: Classifier-Free Guidance (CFG) 모델 학습 시 데이터 점에 대한 조건(Class / Command label $c$) 부여.
  * `eval/guide_flow.py`: 특정 커맨드 조건 $c$를 주입하여 목표 영역으로의 조건부 생성 유도.
* **만족해야 하는 조건**:
  1. $\text{progress}(p) \in [0.0, 1.0]$.
  2. $\text{command\_bins}(p, n)$은 정수형 텐서 (`torch.long`)로 $0$ 이상 $n-1$ 이하의 값을 균등하게 분포시켜야 함.

---

### 2.8. Smooth FMBF 및 CFMBF Active-Set QP 솔버 연계 (`get_fmbf`, `solve_composite_fmbf`)

* **기능**:
  * SafeFlow 알고리즘에서 매 적분 시점마다 원래 속도 벡터 $v$를 가장 작게 수정하면서 안전 불변성(Safety Invariance)을 만족하는 속도 $u^*$를 QP(Quadratic Programming)로 계산합니다:
    $${\min_{u \in \mathbb{R}^D} \frac{1}{2} \|u - v\|_2^2 + w \|\delta\|_2^2 \quad \text{s.t.} \quad a_j + b_j^T u + \delta_j \ge 0, \quad \delta_j \ge 0}$$
    여기서 $b_j = \nabla_p \bar{h}_j(p)$이며, $a_j = \phi(t, \bar{h}_j) \bar{h}_j(p)$입니다 ($\bar{h}_j(p) = -h_j(p) \ge 0$인 Smooth Barrier).
* **역할 (Consumers)**:
  * `eval/safe_flow.py`: ODE 생성 궤적이 경계면에 도달할 때 속도를 반사/편향시켜 실시간으로 궤적을 제약 내부에 가둠.
* **만족해야 하는 조건**:
  1. **평활화 (Smoothness)**:
     * $L_1$ 노름이나 $\max$ 연산자와 같이 미분 불가능한 킹크(Kink)가 존재하는 제약은 QP 행렬식 $b_j$의 불연속을 초래하므로, Huber 평활화 또는 $\text{Softplus}$를 통해 $C^1$ 평활 장벽 함수로 변환해야 합니다.
  2. **Active-Set 가용성**:
     * 동시 활성화될 수 있는 제약의 개수 $K$는 조합 탐색 비용($2^K$)을 고려하여 최대 12개 이하로 유지해야 합니다 (`max_constraints=12`).

---

## 3. 알고리즘별 제약조건 요구사항 매트릭스

새로운 데이터셋이나 제약조건을 구현할 때, 대상 알고리즘에 따라 필수로 구현해야 하는 기능은 아래와 같습니다:

| 알고리즘 | $h(p)$ (오라클) | $C(p)$ (비용함수) | $P(p)$ (물리투영) | `project_feasible` | `estimate_lipschitz` | `energy` / `grad` | `progress` / `command` | Smooth FMBF QP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flow Matching (Vanilla)** | 평가용 필수 | - | - | - | - | - | - | - |
| **HardFlow** | 평가용 필수 | **핵심 (미분)** | - | **선택 (터미널)** | - | - | - | - |
| **YFlow** | 평가용 필수 | **핵심 (미분)** | **권장 (없으면 fallback)** | **선택 (터미널)** | **핵심 (학습률결정)** | - | - | - |
| **SafeFlow** | 평가용 필수 | - | - | **선택 (터미널)** | - | - | - | **핵심 ($C^1$ 장벽)** |
| **UniConFlow** | **핵심 (미분)** | - | - | **선택 (터미널)** | - | - | - | - |
| **GuideFlow (RFE/CFG)**| 평가용 필수 | - | - | - | - | **핵심 (RFE)** | **핵심 (CFG)** | - |

---

## 4. 새로운 데이터셋 및 제약조건 추가 시 구현 체크리스트

새로운 데이터셋(예: 고차원 로봇 관절 궤적, 장애물 회피 맵 등)을 `data/` 패키지에 등록하려면 아래 단계를 거쳐야 합니다:

1. **`data/{dataset_name}.py` 파일 생성 및 `BaseConstraint` 상속**:
   * [ ] `h(p)`: 각 제약의 물리적 의미를 담은 이름 키를 갖는 딕셔너리 반환 (`torch.Tensor` 및 `np.ndarray` 모두 지원).
   * [ ] `cost(p)`: 위반량의 제곱합 또는 페널티 함수 구현 (`torch.autograd.grad` 가능 필수).
   * [ ] `project_feasible(p, buffer)`: 안전 영역 내부로 점을 되돌리는 투영 함수 구현.
   * [ ] (선택) `project_physical(p)`: 물리 매니폴드가 명확하면 구현, 없으면 기본값(`None`) 유지.
   * [ ] (선택) `energy(p)` / `progress(p)`: GuideFlow 지원이 필요한 경우 구현.
   * [ ] (선택) `get_fmbf()`: SafeFlow 지원이 필요한 경우 평활화된 배리어 객체 구현.

2. **데이터셋 빌더 함수 작성 및 레지스트리 등록**:
   ```python
   from data.base import BaseConstraint, DataBundle, register_dataset

   @register_dataset("my_dataset")
   def build_my_dataset(cfg) -> DataBundle:
       # 1. train / eval 텐서 로드 또는 생성
       # 2. 메타데이터 딕셔너리 구성
       # 3. 제약 인스턴스 생성
       constraint = MyDatasetConstraint(...)
       return DataBundle(
           train=train_dataset,
           train_raw=train_raw,
           eval_raw=eval_raw,
           eval_z=eval_z,
           mean=mean,
           std=std,
           constraint=constraint,
           meta=meta,
           meta_dict=meta_dict,
       )
   ```

3. **Autograd 연산 시 주의사항 (In-place 수정 금지)**:
   * PyTorch에서 미분을 계산해야 하는 $h(p)$, $C(p)$, $E(p)$ 내부에서 `p[:, 0] = ...`과 같은 in-place 텐서 변형 연산을 절대 사용하지 마십시오 (autograd backward 오류 발생).
   * 항상 `torch.stack`, `torch.cat`, `torch.where` 등을 사용해 새로운 텐서를 생성하십시오.

4. **단위 테스트 작성**:
   * `test/test_constraints.py`에 신규 제약조건에 대한 오라클 안전성, 미분 가능성, 텐서 차원 일치 여부 테스트 케이스를 필히 추가합니다.
