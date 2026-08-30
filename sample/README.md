# sample 패키지 (sample/)

이 패키지는 무제약 ODE 적분을 둔다. 제약 가이던스는 `eval/`에 있다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Eval](../eval/README.md)

---

## 2. 파일 목록 및 요약
* `euler.py`: 고정 스텝 Euler, $x\leftarrow x+\Delta t\, v_\theta(x,t)$

---

## 3. 세부 명세

### euler.py

#### EulerSampler
*   **설명**: $t$를 $0$에서 $1$까지 $N$등분. `sample`은 최종 $x_1$, `trajectory`는 경로 `[B, N+1, D]`.

학습 중 중간 scatter와 FlowMatch eval이 이 샘플러를 쓴다.
