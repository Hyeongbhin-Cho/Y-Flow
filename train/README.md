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
* `y_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `safe_flow.py`, `unicon_flow.py`, `guide_flow.py`: 아직 미구현

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
