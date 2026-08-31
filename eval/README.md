# eval 패키지 (eval/)

이 패키지는 공통 평가 루프와 방법별 샘플러를 둔다. `evaluate.py`가 `flow_match.py`, `hard_flow.py` 등을 불러 차이를 낸다.

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
* `flow_match.py`: unguided Euler
* `hard_flow.py`: terminal $h,C$ SLSQP 후 affine 복원
* `y_flow.py`: terminal $h,C$ + $P$ warm start SLSQP 후 선형 보간
* `safe_flow.py`, `unicon_flow.py`, `guide_flow.py`: 아직 미구현

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
*   **설명**: $t\ge t_{\mathrm{on}}$에서 예측된 $x_1$에 SLSQP. 마지막 스텝에서 $h(x_N)\le 0$을 목표로 한다.

### y_flow.py

#### sample
*   **설명**: $t\ge t_{\mathrm{on}}$에서 물리 투영 $P(\hat{x}_1)$ warm start 기반 $h,C$ SLSQP 최적화 후 현재 상태와 선형 보간. 마지막 스텝에서 $h(x_N)\le 0$을 목표로 한다.
