# SafeFlow Exp-01 구현

논문: [SafeFlow: Safe Robot Motion Planning with Flow Matching via Control Barrier Functions](https://arxiv.org/abs/2504.08661)

이 구현은 논문의 training-free FMBF/CFMBF 추론을 Exp-01의 2D Swiss roll 좌표점에 적용한다. 논문의 입력은 길이 $H+1$인 로봇 궤적이지만 Exp-01의 샘플 하나는 점 $p\in\mathbb{R}^{2}$이므로 $H=0$인 축소 검증이다. Maze navigation이나 7-DoF manipulation 전체 재현은 아니다.

---

## 1. 공통 backbone

SafeFlow 전용 모델은 학습하지 않는다. `runs/{run_name}/flowmatch/last.pt`의 동결 EMA 속도장 $v_t^\theta$를 FlowMatch, HardFlow, YFlow와 동일하게 사용한다. 체크포인트가 없을 때만 `train/safe_flow.py`가 FlowMatch 학습을 호출한다.

기존 제약 API는 위반량 $g_j(p)\leq0$을 사용한다. 평가 지표는 이 원래 제약을
그대로 사용하지만, FMBF에는 논문의 $C^1$ 가정을 맞추기 위한 매끄럽고 보수적인
대리 barrier $\tilde h_j$를 사용한다. 논문의 부호 규약에 맞춰 안전 영역에서는
$\tilde h_j\geq0$이며,

$$
\tilde h_j(p)\geq0\quad\Longrightarrow\quad g_j(p)\leq0
$$

가 성립하도록 구성한다. 매끄럽게 바꿨기 때문에 일반적으로
$\tilde h_j=-g_j$인 것은 아니다. 이하 코드와 식의 $h_j$는 이 대리 barrier를
뜻한다. Exp-01의 composite barrier는 `tube`, `core`, `outer`, `box` 네 개다.
`outer`는 주기식으로 표현한 무한 나선 중 원래의 유한한 $u$ 구간만 남기는
반경 guard다.

원래의 최근접 나선 거리, $\|p\|_2$, $\|p\|_\infty$는 최근접점이 바뀌는 곳,
원점, 박스 축 동률에서 미분 불가능하다. `swiss_roll_fmbf.py`는 이를 다음과 같이
바꾼다.

- $r_\epsilon=\sqrt{\|p\|_2^2+\epsilon_r^2}$로 반경을 매끄럽게 만든다.
- Archimedean spiral의 위상 $r_\epsilon/a$와 점의 정렬도를 삼각함수로 계산해
  `tube`를 만든다. 최근접점 선택, `atan2`, 절댓값 분기가 없다.
- `core`와 `outer`는 $r_\epsilon$의 선형 barrier다.
- `box`는 log-sum-exp smooth maximum을 사용한다.

따라서 네 barrier와 코드의 해석 gradient는 모든 유한 입력에서 $C^\infty$다.
smooth safe set은 원래 Exp-01 safe set의 부분집합이 되도록 tube margin과 radial
guard를 둔다. 다만 논문 Proposition 2의 “모든 상태에서 $\nabla h\ne0$” 가정은
원형 장애물이나 닫힌 박스의 내부 임계점 때문에 이런 기하에서 전역적으로
만족시킬 수 없다. 구현과 테스트는 안전 경계에서 gradient가 0이 아닌 표준 CBF
regularity를 확인한다. 이는 논문의 로봇 실험 barrier에도 존재하는 일반적인
이론-구현 간 한계다.

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

제약이 네 개뿐이므로 `constraints/fmbf.py`는 $2^4$ active set을 배치 열거해 전역 최적해를 구한다. 보정은 $u_z=u_p/\sigma$로 정규화 공간에 되돌린다. 유효한 active set을 수치적으로 찾지 못하면 다른 해를 대신 반환하지 않고 실행을 중단한다.

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

마지막 결과가 안전하지 않으면 원좌표에서 논문의 최소거리 문제를 SLSQP로 푼다.
Swiss-roll 중심선 점은 SLSQP의 feasible 초기값으로만 사용한다. solver 실패,
비유한 결과, 제약 위반이 발생하면 즉시 예외를 내며 중심선 투영 등의 terminal
fallback을 반환하지 않는다. terminal filter 후 `safe_ratio=1.0`을 완료 조건으로
둔다.

---

## 4. 실행과 진단

```bash
python main.py safeflow --mode train --run_name exp_01_swiss_roll
python main.py safeflow --mode eval --run_name exp_01_swiss_roll
python main.py safeflow --mode eval --run_name exp_01_swiss_roll \
  --safeflow.integrator dopri5

# 동일한 x0로 t_on=0.5/0.7/0.8/0.9 재현
python -m eval.safe_flow_t_on_ablation
```

`metrics.json`에는 공통 safety/MMD/time 외에 `nfe`, `pre_filter_safe_ratio`,
terminal-filter 발동률, 보정량, slack, FMBF residual을 기록한다. 적분기별 결과는
`metrics_euler.json`과 `metrics_dopri5.json`에도 보존한다. 최종 안전성뿐 아니라
terminal filter가 대부분의 결과를 대신 고친 것은 아닌지도 함께 확인한다.

`t_on` ablation은 `runs/exp_01_swiss_roll/safeflow/t_on_ablation/`에 설정별
`config.yaml`, `eval_samples.npy`, `eval_samples.png`, `metrics.json`을 저장한다.
루트의 `summary.json`, `comparison.png`, `u_histogram.png`는 네 설정의 비교용이다.
