# SafeFlow Exp-01 구현

논문: [SafeFlow: Safe Robot Motion Planning with Flow Matching via Control Barrier Functions](https://arxiv.org/abs/2504.08661)

이 구현은 논문의 training-free FMBF/CFMBF 추론을 Exp-01의 2D Swiss roll 좌표점에 적용한다. 논문의 입력은 길이 $H+1$인 로봇 궤적이지만 Exp-01의 샘플 하나는 점 $p\in\mathbb{R}^{2}$이므로 $H=0$인 축소 검증이다. Maze navigation이나 7-DoF manipulation 전체 재현은 아니다.

---

## 1. 공통 backbone

SafeFlow 전용 모델은 학습하지 않는다. `runs/{run_name}/flowmatch/last.pt`의 동결 EMA 속도장 $v_t^\theta$를 FlowMatch, HardFlow, YFlow와 동일하게 사용한다. 체크포인트가 없을 때만 `train/safe_flow.py`가 FlowMatch 학습을 호출한다.

기존 제약 API는 위반량 $g_j(p)\leq0$을 사용한다. SafeFlow는 논문 규약에 맞춰

$$
h_j(p)=-g_j(p)\geq0
$$

으로 변환한다. Exp-01의 composite barrier는 `tube`, `core`, `box` 세 개다.

---

## 2. CFMBF 보정

원좌표에서 $v_p=\sigma\odot v_z$를 계산하고 다음 relaxed QP를 푼다.

$$
\min_{u,\delta\geq0}\|u\|^2+w_\delta\|\delta\|^2
$$

$$
a_j+b_j^\top u+\delta_j\geq0,
\quad b_j=\nabla h_j(p),
\quad a_j=b_j^\top v_p+\varphi(t,h_j)h_j.
$$

제약이 세 개뿐이므로 `constraints/fmbf.py`는 $2^3$ active set을 배치 열거해 전역 최적해를 구한다. 보정은 $u_z=u_p/\sigma$로 정규화 공간에 되돌린다.

기본 gain은 논문 실험의 piecewise schedule이다.

$$
\varphi_1(t)=
\begin{cases}
1+4t^3,&t<0.9\\
(1-t)^{-1},&t\geq0.9
\end{cases}
$$

안전한 $h_j\geq0$에는 $\varphi_0=1$을 쓴다. CFMBF는 기본적으로 $t\geq0.5$에서 켠다.

---

## 3. 적분과 terminal filter

`euler`는 다른 Exp-01 방법과 같은 `sample.n_steps`를 사용한다. `dopri5`는 `torchdiffeq`로 $[0,t_{on}]$과 $[t_{on},1-\epsilon]$을 나눠 적분한다. 기본 $\epsilon=10^{-3}$이다.

마지막 결과가 안전하지 않으면 원좌표에서 최소거리 SLSQP terminal filter를 실행한다. 수치 최적화가 실패하면 가장 가까운 안전한 Swiss-roll 중심선 점을 사용한다. terminal filter 후 `safe_ratio=1.0`을 완료 조건으로 둔다.

---

## 4. 실행과 진단

```bash
python main.py safeflow --mode train --run_name exp_01_swiss_roll
python main.py safeflow --mode eval --run_name exp_01_swiss_roll
python main.py safeflow --mode eval --run_name exp_01_swiss_roll \
  --safeflow.integrator dopri5
```

`metrics.json`에는 공통 safety/MMD/time 외에 `nfe`, `pre_filter_safe_ratio`, terminal-filter 발동률, 보정량, slack, FMBF residual, QP 및 terminal fallback 횟수를 기록한다. 적분기별 결과는 `metrics_euler.json`과 `metrics_dopri5.json`에도 보존한다. 최종 안전성뿐 아니라 terminal filter가 대부분의 결과를 대신 고친 것은 아닌지도 함께 확인한다.
