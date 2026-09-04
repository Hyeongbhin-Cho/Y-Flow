# eval 패키지 (eval/)

이 패키지는 공통 평가 루프와 방법별 샘플러를 둔다. `evaluate.py`가 `flow_match.py`, `hard_flow.py`, `guide_flow.py` 등을 불러 차이를 낸다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Train 패키지 설명 문서](../train/README.md)
*   [Data 패키지 설명 문서](../data/README.md)

---

## 2. 파일 목록 및 요약
* `evaluate.py`: 데이터, 시간 측정, metrics, scatter, JSON
* `metrics.py`: `safe_ratio`, tube/core/box 위반, MMD, radius MAE
* `_backbone.py`: frozen FlowMatch $v_t^\theta$ 로드
* `sample_result.py`: 방법별 추가 진단 지표를 공통 평가기로 전달
* `flow_match.py`: unguided Euler
* `guide_flow.py`: CVF / CF / RFE 제약 주입 샘플링, 옵션 CFG
* `safe_flow.py`: CFMBF-QP 속도 보정, Euler/Dopri5, terminal safety filter
* `safe_flow_t_on_ablation.py`: 같은 체크포인트와 $x_0$로 `t_on` 네 값을 재평가하고 설정별 산출물 저장
* `unicon_flow.py`: PTZF certificate, batched slack QP, terminal refinement
* `hard_flow.py`: terminal $h,C$ SLSQP 후 affine 복원
* `y_flow.py`: 물리 투영, terminal PGD, 선형 보간


---

## 3. 세부 명세

### evaluate.py

#### run_eval
*   **설명**: `eval.{method}.sample(cfg, device, x0)`을 호출한다. 결과는 `runs/{run_name}/{command}/metrics.json`과 `runs/{run_name}/metrics.json`.

### flow_match.py

#### sample
*   **설명**: 사전학습 $v_\theta$를 Euler로만 적분. $h$는 평가에만 쓴다.

### hard_flow.py

#### sample
*   **설명**: $t\ge t_{\mathrm{on}}$에서 예측된 종단점에 대해 PyTorch Autograd GPU 배치 PGD로 $h,C$ 최적화를 수행하고, HardFlow 고유의 Affine 역매핑($w_0$)으로 복원한다. 마지막 스텝에서 $h(x_N)\le 0$을 만족한다.

#### solve_terminal_pgd_hardflow
*   **설명**: 목적함수(비용 $C$ + FlowMatch 정규화 패널티)에 대해 GPU 배치 투영 경사하강법(PGD)을 수행하여 safe set 내부의 최적 종단점을 구한다.

### y_flow.py

#### sample
*   **설명**: 동결 FlowMatch 속도를 따르다 $t\ge t_{\mathrm{on}}$에서 예측 종단점에 대해 물리 투영 $P(\hat{x}_1^{\mathrm{raw}})$ warm start 후 PyTorch Autograd GPU 배치 PGD로 $h,C$ 최적화를 수행한다. 국소 립시츠 추정치 $\widehat{L}_P$에 따른 적응형 게이팅과 선형 보간을 적용한다.

#### estimate_lipschitz
*   **설명**: 물리 투영 연산자 $P$의 국소 립시츠 상수를 유한차분 배치 연산으로 추정한다 ($L_P \le 1+\delta$ 안정성 검증).

#### optimize_x1_pgd
*   **설명**: 목적함수(비용 $C$ + nominal 복원 + 물리 투영 가이던스)에 대해 GPU 배치 투영 경사하강법을 수행하여 safe set 내부의 최적 종단점을 구한다.

### safe_flow.py

#### sample
*   **설명**: 동결 FlowMatch 속도를 원좌표로 변환하고 $t\ge0.5$에서 composite FMBF slack-QP의 최소 보정을 더한다. Euler와 Dopri5를 지원하며 마지막에 안전하지 않은 점만 terminal filter로 보정한다.

### safe_flow_t_on_ablation.py

#### run_ablation
*   **설명**: 기본 `t_on=[0.5, 0.7, 0.8, 0.9]`를 동일한 4,000개 `x0`로 실행한다. 각 설정의 config, samples, metrics, scatter와 전체 비교 그림·$u$ 히스토그램을 `runs/{run_name}/safeflow/t_on_ablation/`에 저장한다.

### unicon_flow.py

#### sample
*   **설명**: PTZF(Prescribed-Time Zeroing Function) 기반 제약 certificate와 배치 slack QP를 통해 매 스텝 nominal 속도에 최소 보정 가이던스를 더한다. 종료 시점에서는 매니폴드 투영(terminal refinement)으로 잔여 수치 오차를 제거한다.

#### constraint_values / constraint_jacobian
*   **설명**: tube/core/box 제약 $h(z)\le 0$ 값과 Autograd를 통한 자코비안 $\partial h/\partial z$를 배치 계산한다.

#### ptzf_reference / qp_guidance
*   **설명**: 종단 $t=1$에서 0으로 수렴하는 기준 궤적 $r(t)$와 그 도함수를 구하고, certificate 조건을 만족하는 closed-form 배치 최소 노름 QP 가이던스를 계산한다.

### guide_flow.py

#### sample
*   **설명**: 동결 $v_t^\theta$에 세 제약을 주입한다. CVF는 속도장 보정, CF는 $k_c$에서 flow 상태 재설정, RFE는 $t\ge\tau^{*}$에서 에너지 하강. 세 모듈을 모두 끄면 `flow_match.sample`과 같은 결과가 나온다.

#### build_anchor_vocabulary
*   **설명**: $h\le 0$인 train 점에 farthest point sampling. 논문의 $\mathcal{V}_a$ ($N=256$)에 대응.

#### energy_grad
*   **설명**: $h$의 제곱 hinge와 $w_{\mathrm{cost}}C$에 대한 닫힌 형태 기울기. 원좌표 $p$에서 계산한다.

#### _GuidedVelocity
*   **설명**: Eq. (13)의 $v^{\mathrm{guide}}=(1-\gamma)v(x,t)+\gamma v(x,t,c)$. `guidance.enabled`일 때만 쓰며 조건부 backbone이 필요하다.

#### energy_torch
*   **설명**: 미분 가능한 $E(p)$. EBM 결합 학습이 쓰며, 투영을 상수로 둔 subgradient가 `energy_grad`와 일치한다.

#### owns_backbone
*   **설명**: CFG나 EBM이 켜져 있으면 GuideFlow가 자체 backbone을 쓴다는 판정.

#### command_bins / ego_progress
*   **설명**: $C_d$(나선 구간 one-hot)와 $C_r$(중심선 진행도 EP)를 계산한다.

#### energy_weight
*   **설명**: Eq. (5)의 $\varepsilon(t)$. $\tau^{*}$ 전에는 0, $[\tau^{*},1]$에서 선형 증가, $t\ge 1$에서 $\varepsilon_{\max}$.
