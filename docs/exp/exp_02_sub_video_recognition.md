# Exp-02-sub. CLEVRER FlowMatch 비디오 인식 실험

Wan2.1 픽셀 생성(Exp-02) **앞**에 두는 중간 실험이다.  
CLEVRER 원본 클립과 공개 정답 주석만 사용해, Flow Matching이 **장면을 구조화된 상태로 인식**하게 한다.

비교 방법 (Exp-01과 동일, 인식용 $v_t^\theta$ 위에서 training-free):

- **FlowMatch** (무제약 baseline)
- **HardFlow**
- **SafeFlow**
- **UniConFlow**
- **GuideFlow**
- **YFlow**

목적: Mask R-CNN / PropNet 가중치가 없는 상태에서, 프로젝트 본래 도구인 Flow Matching으로

1. 객체 인식
2. 경로(궤적) 파악
3. 충돌 검출

을 정의하고, 각 과제에 **동일한 $h(S)\le 0$ 규칙**을 건다. 픽셀 비디오를 생성하지 않는다.

---

## 1. 왜 Wan 생성 전에 이 실험인가

Exp-02는 생성 영상에서 객체 궤적·충돌을 읽어 물리 제약을 평가해야 한다. 현재 상태:

| 도구 | 상태 | Wan 생성 영상에 |
| :--- | :--- | :---: |
| CLEVRER-finetuned Mask R-CNN | 가중치 미공개 | 불가 |
| 공식 `derender_proposals` | 원본 클립 parser 산출물 | 불가 |
| 공식 `propnet_preds` | 원본 클립 동역학 산출물 | 불가 |
| COCO Mask R-CNN | 실행 가능 | 부적합 (원본에서도 cup/sports ball, 검출 수 부족) |

정답 주석은 validation에 이미 있다. 인식 모델을 **주석 공간**에서 먼저 학습·제약하면,

- Exp-01과 같은 $h,C,P$ API를 비디오 픽셀 없이 검증하고
- 이후 Exp-02의 `generated_tracks` 평가기를 같은 상태 표현으로 연결할 수 있다.

생성 영상의 정확한 재현은 성공 조건이 아니다. 이 실험의 기본 모드는 **원본 CLEVRER 클립을 조건으로 상태 $S$를 인식**하는 것이다.

---

## 2. 한 샘플은 픽셀이 아니다

Exp-01이 2D 좌표점이었듯이, 이 실험의 한 샘플은 RGB 텐서가 아니라 **고정 슬롯 장면 상태** $S$이다.

| 방식 | 쓰는가 | 설명 |
| :--- | :---: | :--- |
| 클립 조건 $V$에서 구조화 상태 $S$를 인식 | **예 (본실험)** | 객체 속성·궤적·충돌 |
| Wan DiT로 33프레임 영상을 생성 | **아니오** | Exp-02 |
| COCO Mask R-CNN으로 박스를 침 | **아니오** | 도메인 불일치가 이미 확인됨 |
| 공식 visual-mask JSON을 Wan 오라클로 사용 | **아니오** | 원본 전용 산출물 |

조건 영상 $V$는 인코더 입력이다. 제약 오라클 $h$는 $V$가 아니라 $S$에서 정의한다 (`data/NOTICE.md` 좌표계 분리).

---

## 3. 상태 표현 $S$와 좌표계

### 3.1 물리 공간

CLEVRER 시뮬레이터 **world 좌표**를 물리 공간으로 둔다. 이미지 픽셀 좌표로 나누어 쓰지 않는다.

eval 100 (scene 10000–10099)에서 관측된 범위. 임계값 확정은 **train/dev에서만** 하고 eval 100에는 맞추지 않는다.

| 양 | 관측 | 설계에 쓰는 초안 |
| :--- | :--- | :--- |
| 객체 수 | 4–6 (평균 4.84) | 슬롯 $K=6$, 빈 슬롯은 비활성 |
| 충돌 수 | 1–4 / 클립 | 이벤트 행렬 |
| 색 | 8종 | `blue, brown, cyan, gray, green, purple, red, yellow` |
| 재질 | 2종 | `metal, rubber` |
| 모양 | 3종 | `cube, cylinder, sphere` |
| 위치 $xy$ | 대략 $[-11,11]$ | 테이블 반경 $R_{xy}=12$ |
| 높이 $z$ | $0.199$–$0.238$ (거의 상수) | 평면 운동, $z$는 보조 |
| 속도 $\|v_{xy}\|$ | 최대 $\approx 3$ | $v_{\max}=3.2$ |
| 충돌 시 중심 거리 | $0.40$–$0.48$ | 접촉 대역 $[0.38, 0.52]$ |
| 25 fps 한 프레임 이동 | 평균 $0.023$, 최대 $0.083$ | 가시 구간에 텔레포트 금지 |

시간 격자는 Exp-02와 맞춘다. 원본 128프레임 / 25 fps에서

$${
s_k = k/16,\qquad k=0,\dots,32
}$$

로 33프레임·16 fps·2초를 고르고, 가장 가까운 원본 `frame_id`의 주석을 쓴다. 전체 128프레임 인식은 ablation이다.

### 3.2 슬롯 텐서

최대 객체 수 $K=6$, 클립 길이 $T=33$.

$${
S=\big(a,\; m,\; q,\; \nu,\; p,\; u,\; c\big)
}$$

| 기호 | 형상 | 의미 |
| :--- | :--- | :--- |
| $a$ | $\{0,\dots,7\}^{K}$ | 색 |
| $m$ | $\{0,1\}^{K}$ | 재질 |
| $q$ | $\{0,1,2\}^{K}$ | 모양 |
| $\nu$ | $[0,1]^{K\times T}$ | 가시성 (`inside_camera_view`) |
| $p$ | $\mathbb{R}^{K\times T\times 3}$ | world 위치 |
| $u$ | $\mathbb{R}^{K\times T\times 3}$ | world 속도 |
| $c$ | $\{0,1\}^{T\times K\times K}$ | 프레임 $t$에서 슬롯 $i,j$ 충돌. 대칭, 대각 0 |

비활성 슬롯: $\nu_{k,t}=0$ 전부, 속성은 don't-care, 제약에서 제외.  
활성 슬롯 수 $N_{\mathrm{obj}}=\lvert\{k:\max_t\nu_{k,t}>\tfrac12\}\rvert$.

Flow Matching은 이산 성분을 원-핫/로짓의 연속 완화로 두고, 터미널에서만 argmax로 되돌린다. $h$ 평가는 되돌린 뒤 **물리 공간**에서 한다.

### 3.3 조건 영상

$${
V\in[-1,1]^{3\times T\times H\times W}
}$$

기본은 원본 해상도 $320\times 480$ (letterbox 없음). Exp-02와 같은 $480\times 832$ letterbox를 쓰면 `valid_region` 밖의 픽셀은 인코더에 넣지 않는다.

기본 모드에서 모델은 **미래 주석을 생성 중 제약으로 쓰지 않는다.** 주석은 학습 타깃과 평가 전용이다.

---

## 4. 인식 과제와 제약

세 과제는 한 상태 $S$의 서로 다른 좌표다. 평가는 과제별로 끊고, Total Safety는 세 과제 제약을 동시에 만족한 비율이다.

부호 규약: $h_j(S)\le 0$이면 만족. 비용은

$${
C(S)=\frac12\sum_j w_j\max\big(0,h_j(S)\big)^2
}$$

로, $S$가 전부 可行이면 $C=0$이고 경계에서 $C^1$이다.

### 4.1 과제 A — 객체 인식

**모델이 할 일**: 클립에 등장하는 강체의 **개수, 속성, 슬롯 정체성**을 맞춘다. ID는 속성과 궤적으로 대응하고, 임의로 다시 붙이지 않는다.

**맞출 것 (지표, 제약이 아님)**:

- 속성 집합 $\{(a_k,m_k,q_k)\}$ vs GT `object_property` (Hungarian, 활성 슬롯만)
- 객체 수 $N_{\mathrm{obj}}$
- 프레임별 가시 객체 수 vs `inside_camera_view`

**Hard constraint $h\le 0$** (인식 결과가 CLEVRER 장면에 속할 것):

1. **어휘**
   $${
   h_{\mathrm{vocab}}(S)=\max_k\Big(
     [a_k\notin\mathcal{A}]+[m_k\notin\mathcal{M}]+[q_k\notin\mathcal{Q}]
   \Big)\le 0
   }$$
   연속 완화에서는 원-핫이 심플렉스 안에 있게 두고, 터미널에서 최근접 어휘로 투영한다.

2. **슬롯 수**
   $${
   h_{\mathrm{count}}(S)=\max\big(N_{\mathrm{obj}}-6,\; 3-N_{\mathrm{obj}}\big)\le 0
   }$$
   (전 validation은 3–6. eval 100은 4–6이었으나 제약은 데이터셋 전체를 따른다.)

3. **속성 시간 불변**  
   활성 슬롯의 $(a,m,q)$는 프레임마다 바뀌면 안 된다. 연속 상태에서는
   $${
   h_{\mathrm{ident}}(S)=\max_{k,t,t'}\nu_{k,t}\nu_{k,t'}\,\lVert \tilde e_{k,t}-\tilde e_{k,t'}\rVert_1-\epsilon_{\mathrm{ident}}\le 0
   }$$
   $\tilde e$는 속성 원-핫.

4. **빈 슬롯**  
   $\max_t\nu_{k,t}\le\tfrac12$인 슬롯은 충돌 행·열과 위치 제약에서 빠진다. 빈 슬롯이 궤적만 혼자 가지면 위반:
   $${
   h_{\mathrm{null}}(S)=\max_k\big((1-\max_t\nu_{k,t})\cdot \max_t\lVert p_{k,t}\rVert_2\big)-\epsilon_{\mathrm{null}}\le 0
   }$$

초안 허용: $\epsilon_{\mathrm{ident}}=0.05$, $\epsilon_{\mathrm{null}}=10^{-3}$.

### 4.2 과제 B — 경로 파악

**모델이 할 일**: 활성 객체마다 world 궤적 $p_{k,0:T-1}$과 속도 $u_{k,0:T-1}$, 가시성 $\nu_{k,t}$를 복원한다. CLEVRER의 **화면 진입·퇴장은 허용**한다. 화면 밖 구간에 저가속도를 전 구간에 강제하지 않는다.

**맞출 것 (지표)**:

- ADE / FDE (가시 프레임만, Hungarian 이후)
- 가시성 F1
- 속도 MAE (가시 프레임)

**Hard constraint**:

1. **테이블 평면 (박스)**
   $${
   h_{\mathrm{table}}(S)=\max_{k,t}\nu_{k,t}\big(\lVert p_{k,t,xy}\rVert_\infty-R_{xy}\big)\le 0
   }$$
   $R_{xy}=12$. $z$는 관측상 거의 $0.20$이므로
   $${
   h_{\mathrm{plane}}(S)=\max_{k,t}\nu_{k,t}\lvert p_{k,t,z}-z_0\rvert-\tau_z\le 0
   }$$
   $z_0=0.20$, $\tau_z=0.05$.

2. **텔레포트 금지 (가시 연속 구간만)**  
   $\nu_{k,t}=\nu_{k,t+1}=1$이면
   $${
   h_{\mathrm{step}}(S)=\max_{k,t}\nu_{k,t}\nu_{k,t+1}\big(\lVert p_{k,t+1}-p_{k,t}\rVert_2-v_{\max}\Delta s\big)\le 0
   }$$
   $\Delta s=1/16$, $v_{\max}=3.2$이면 한 스텝 상한 $0.20$. 원본 25 fps 최대 이동 $0.083$보다 넉넉하다.

3. **위치–속도 정합**
   $${
   h_{\mathrm{kin}}(S)=\max_{k,t}\nu_{k,t}\nu_{k,t+1}\big(\lVert p_{k,t+1}-p_{k,t}-u_{k,t}\Delta s\rVert_2-\tau_{\mathrm{kin}}\big)\le 0
   }$$
   $\tau_{\mathrm{kin}}=0.05$.

4. **비충돌 구간의 급가속 금지**  
   충돌로 표시되지 않은 프레임에서
   $${
   h_{\mathrm{acc}}(S)=\max_{k,t}\omega_{k,t}\big(\lVert u_{k,t+1}-u_{k,t}\rVert_2-\tau_{\mathrm{acc}}\big)\le 0
   }$$
   $$\omega_{k,t}=\nu_{k,t-1}\nu_{k,t}\nu_{k,t+1}\prod_{j\neq k}(1-c_{t,k,j})$$
   $\tau_{\mathrm{acc}}=0.40$ (충돌이 아닌 프레임의 속도 점프). 충돌 프레임은 이 항에서 뺀다.

### 4.3 과제 C — 충돌 검출

**모델이 할 일**: 어느 두 슬롯이 어느 시각에 충돌하는지 $c_{t,i,j}$를 예측한다. PropNet처럼 미래를 롤아웃하지 않는다. **관측된 클립 안의 충돌 사실**만 대상으로 한다.

**맞출 것 (지표)**:

- 이벤트 precision / recall (속성 쌍 + 프레임 허용 $\pm 5$ 원본 프레임, 16 fps면 $\pm 2$ 스텝)
- 충돌 시각 MAE (매칭된 이벤트)

**Hard constraint**:

1. **대칭·자기충돌 금지**
   $${
   h_{\mathrm{sym}}(S)=\max_{t,i,j}\lvert c_{t,i,j}-c_{t,j,i}\rvert+\max_{t,i}c_{t,i,i}\le 0
   }$$

2. **둘 다 가시**
   $${
   h_{\mathrm{colvis}}(S)=\max_{t,i,j}c_{t,i,j}\big(2-\nu_{i,t}-\nu_{j,t}\big)\le 0
   }$$

3. **접촉 거리에서만 충돌**  
   충돌이면 중심 거리가 접촉 대역 안에 있어야 한다.
   $${
   h_{\mathrm{contact}}(S)=\max_{t,i,j}c_{t,i,j}\big(\lvert \lVert p_{i,t}-p_{j,t}\rVert_2-d_0\rvert-\tau_{\mathrm{col}}\big)\le 0
   }$$
   $d_0=0.45$, $\tau_{\mathrm{col}}=0.07$ (관측 충돌 거리 $0.40$–$0.48$).

4. **관통 금지 (비충돌 포함 전 가시 쌍)**
   $${
   h_{\mathrm{penetrate}}(S)=\max_{t,i<j}\nu_{i,t}\nu_{j,t}\big(d_{\min}-\lVert p_{i,t}-p_{j,t}\rVert_2\big)\le 0
   }$$
   $d_{\min}=0.38$. 가까워도 충돌이 아닐 수 있으므로, 근접만으로 $c=1$을 강제하지 않는다.

5. **충돌 시 속도 점프**  
   접촉만 있고 상대속도가 안 바뀌면 스침으로 본다.
   $${
   h_{\mathrm{impulse}}(S)=\max_{t,i,j}c_{t,i,j}\big(\delta_v-\lVert \Delta u_{i,t}\rVert_2-\lVert \Delta u_{j,t}\rVert_2\big)\le 0
   }$$
   $\Delta u_{k,t}=u_{k,t+1}-u_{k,t-1}$, $\delta_v=0.15$. 클립 양 끝 프레임은 제외.

### 4.4 과제 요약

| 과제 | 인식 대상 | 핵심 $h$ | GT 대응 필드 |
| :--- | :--- | :--- | :--- |
| A 객체 | 개수·색·재질·모양·슬롯 정체성 | vocab, count, ident, null | `object_property` |
| B 경로 | world 위치·속도·가시성 | table, plane, step, kin, acc | `motion_trajectory` |
| C 충돌 | 쌍·시각 | sym, colvis, contact, penetrate, impulse | `collision` |

픽셀 MSE, FVD, CLIP은 이 실험의 지표가 아니다.

---

## 5. $P(S)$와 `project_feasible`

YFlow warm start $P(S)$ (1-Lipschitz를 목표로 하는 휴리스틱):

1. 속성 로짓을 어휘 원-핫으로 양자화
2. $p_{xy}$를 $\lVert p_{xy}\rVert_\infty\le R_{xy}$로 클램프, $p_z\leftarrow z_0$
3. 가시 연속 구간만 짧은 시간 창으로 위치 스무딩 (충돌로 표시된 $t$는 건너뜀)
4. $c\leftarrow \tfrac12(c+c^\top)$, 대각 0
5. 비활성 슬롯 궤적·충돌을 0으로

명시적 $P$가 불안정하면 $\mu=0$으로 HardFlow형 터미널 최적화만 쓴다 (`docs/YFlow.md` 4.3.1).

`project_feasible`: 위 클램프 + 관통 쌍을 $d_{\min}$ 위로 밀어내기 + 어휘 양자화. 터미널에서만 쓰고, 경로 중간을 매 스텝 투영하지 않는다.

---

## 6. FlowMatch 학습

무제약 baseline은 **비디오 조건 Conditional Flow Matching**이다. Wan은 쓰지 않는다.

$${
S_\tau=(1-\tau)S_0+\tau S_1,\qquad
\mathcal{L}=\mathbb{E}\big\lVert v_\theta(S_\tau,\tau,E(V))-(S_1-S_0)\big\rVert^2
}$$

- $S_1$: 위 33프레임 격자에 맞춘 GT 상태
- $S_0\sim\mathcal{N}(0,I)$ (같은 형상)
- $E(V)$: 작은 2D CNN + 시간 풀, 또는 프레임 독립 임베딩. 학습 대상
- $h$는 학습에 쓰지 않는다 (Exp-01과 같음)

데이터:

- **학습**: CLEVRER train (scene 0–9999). 아직 내려받지 않았으면 `setup_clevrer.py --splits train`
- **임계값/조기종료**: train에서 떼낸 dev, eval 100과 겹치지 않게
- **평가**: 고정 eval 100 (scene 10000–10099, `datasets/clevrer/manifests/eval_100.json`)

공식 visual-mask는 **학습 보조 손실에만** 쓸 수 있다 (2D 마스크 정렬). 평가 지표의 정답은 시뮬레이터 JSON이다. 테스트 시 공식 mask JSON을 읽으면 원본 전용 누수가 된다.

추론: 같은 $V$, 같은 $S_0$ 시드로 Euler 적분. 제약 다섯 방법은 이 $v_\theta$를 동결하고 샘플링만 교체한다.

가중치 위치:

| 경로 | 역할 |
| :--- | :--- |
| `runs/{run_name}/flowmatch/last.pt` | 해당 run의 학습 체크포인트 (optimizer 포함) |
| `checkpoints/clevrer_flow/last.pt` | 다른 실험이 읽는 고정 경로 (`model.local_dir`) |

학습이 끝나면 둘 다 갱신된다. 이미 `runs/`에만 있는 가중치는 `python scripts/export_clevrer_flow.py --run_name exp_02_sub_video_recognition`으로 복사한다. eval·training-free 방법은 run 파일이 없으면 published 경로를 쓴다.

---

## 7. 평가 지표

제약 만족 (Safety):

- `attr_safe`, `track_safe`, `collision_safe`, `total_safe`
- 제약별 viol. rate / mean $(h)_+$

인식 품질 (GT 대비, Hungarian 슬롯 대응 후):

| 지표 | 과제 | 방향 |
| :--- | :---: | :---: |
| Attribute set F1 | A | $\uparrow$ |
| Count MAE | A | $\downarrow$ |
| Visibility F1 | B | $\uparrow$ |
| ADE / FDE (가시 프레임, world) | B | $\downarrow$ |
| Collision event P/R/F1 ($\pm$ 허용 프레임) | C | $\uparrow$ |
| Unassessable rate | 공통 | 빈 슬롯·전부 비가시 등은 안전으로 세지 않음 |

시스템: 초/클립, NFE (속도장 평가 횟수). 픽셀 FVD는 보고하지 않는다.

공식 `propnet_preds`는 **참고 열**로만 붙인다. 같은 원본 클립의 상한선이지, 이 실험의 학습 타깃이 아니다.

---

## 8. Exp-02 (Wan 생성)과의 관계

```text
본 실험:  V_clevrer  →  vθ  →  S_hat   (인식, 주석 공간)
Exp-02:    z0, text  →  Wan →  V_gen  →  (추후) S_hat_gen
```

이 실험이 끝나면 $S$의 좌표, $h$, 슬롯 대응, 충돌 허용 오차가 고정된다. Exp-02의 `generated_tracks`는 같은 $S$ 스키마로 맞춘다.

Wan 클립에 이 $v_\theta$를 그대로 적용하는 것은 **도메인 이동 점검**이며, 본 실험의 성공 조건이 아니다. COCO 검출 점수로 물리 안전을 주장하지 않는 규칙과 같다.

---

## 9. 단계와 완료 조건

1. **상태 dump**  
   train/eval 클립을 $S$ 텐서 + meta로 저장 (`datasets/clevrer/states/`). 좌표는 world, 시간은 33×16 fps 매핑.  
   완료: eval 100의 $N_{\mathrm{obj}}$, 충돌 수가 JSON 주석과 일치. 짧은 클립·frame 밀림 없음.

2. **무제약 FlowMatch 학습**  
   $V\to S$ CFM. $h$는 평가만.  
   완료: eval에서 attribute F1, ADE, collision F1을 기록. Safety는 제약 방법보다 낮을 것으로 본다.

3. **제약 오라클**  
   `CLEVRERStateConstraint`로 위 $h$를 구현, `test/test_constraints.py`에 GT 상태는 전부 $h\le 0$인지 확인.  
   완료: eval 100의 GT dump가 `total_safe=1`. 원점 바깥·관통 합성 상태는 위반.

4. **다섯 방법 비교**  
   같은 $S_0$, 같은 $v_\theta$.  
   완료: 표 + 궤적 그림. YFlow/HardFlow Safety 목표 아래.

5. **(선택) Wan 도메인 이동**  
   생성 클립이 생긴 뒤에만. 본 실험 게이트가 아니다.

시작 하이퍼 (확정은 dump 이후):

- $K=6$, $T=33$, fps $16$, $R_{xy}=12$, $v_{\max}=3.2$
- $d_0=0.45$, $d_{\min}=0.38$, $\tau_{\mathrm{col}}=0.07$
- `t_on=0.5`, Euler $N=50$, seed $42$

---

## 10. 성공 기준

Default eval 100, 같은 $S_0$:

1. GT dump의 Total Safety $=1$ (오라클 자체 검사)
2. 무제약 FlowMatch보다 제약 방법의 `collision_safe`, `track_safe`가 높음
3. HardFlow 또는 YFlow **Total Safety $\ge 0.95$**
4. 제약을 켠 뒤 attribute F1이 무제약 대비 **5%p 이상 떨어지지 않음** (인식 붕괴 방지)
5. 3D 비관통을 픽셀 겹침만으로 주장하지 않음. 이 실험의 관통은 **world 중심 거리**다.

---

## 11. 한 줄

Exp-02-sub는 Wan으로 영상을 만들기 전에, CLEVRER 주석 공간에서 Flow Matching이 **객체·경로·충돌을 인식**하게 하고, 그 인식 결과에 Swiss roll과 같은 hard constraint를 거는 실험이다.
