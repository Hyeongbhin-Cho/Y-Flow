# train 패키지 (train/)

이 패키지는 Flow Matching 학습과, 이후 training-free 방법의 진입점을 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Eval 패키지 설명 문서](../eval/README.md)
*   [Data 패키지 설명 문서](../data/README.md)

---

## 2. 파일 목록 및 요약
* `trainer.py`: 공통 학습 루프. 체크포인트는 `runs/{run_name}/flowmatch/`
* `flow_match.py`: 무제약 linear CFM loss
* `ema.py`: exponential moving average
* `checkpoint.py`: `last.pt` 저장/로드
* `hard_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `guide_flow.py`: 기본은 training-free. `guidance.enabled` 또는 `rfe_train.rfe_loss`이면 자체 backbone 학습
* `y_flow.py`, `safe_flow.py`, `unicon_flow.py`: 아직 미구현

---

## 3. 세부 명세

### trainer.py

#### run_train
*   **설명**: Swiss roll 캐시를 읽고 CFM을 학습한다. 산출물은 `runs/{run_name}/{method}/last.pt`.

### flow_match.py

#### ConditionalFlowMatching
*   **설명**: $x_t=(1-t)x_0+t x_1$, $\|v_\theta-u\|^2$. $h,C,P$ 없음. `velocity()`는 Euler가 호출.

### hard_flow.py / y_flow.py

#### ensure_flowmatch_ckpt
*   **설명**: 체크포인트가 있으면 경로를 출력하고 반환. 없으면 `run_train(..., method="flowmatch")`.

### guide_flow.py

#### ensure_flowmatch_ckpt
*   **설명**: CFG와 EBM이 모두 꺼져 있을 때 GuideFlow는 training-free다. 원논문의 EBM 결합 학습($\mathcal{L}_{\mathrm{RFE}}$) 대신 제약을 추론 시점에 해석적으로 평가한다. 사유는 `docs/GuideFlow.md`.

#### run_train_guideflow
*   **설명**: GuideFlow 자체 backbone을 학습한다. `guidance.enabled`면 Eq. (12)의 조건 마스킹을, `rfe_train.rfe_loss`이면 Eq. (18)의 에너지 항을 더한다. 생성 종단은 기본적으로 샘플러 격자를 그대로 따라 rollout하며, `rollout_steps`로 저비용 근사로 바꿀 수 있다. 둘은 독립적으로 조합된다. 산출물은 `runs/{run_name}/guideflow/last.pt`.

#### build_conditions
*   **설명**: 학습 점마다 $C_p$(최근접 앵커), $C_d$(나선 구간 one-hot), $C_r$(중심선 진행도)를 만든다.
