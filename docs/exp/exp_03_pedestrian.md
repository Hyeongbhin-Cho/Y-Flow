# Exp-03. Pedestrian Y-Flow (ETH/UCY)

## 구성

Exp-03은 두 갈래다.

| 갈래 | 데이터 객체 | 모델 | 실행 |
| --- | --- | --- | --- |
| **A. Y-Flow 파이프라인 (주)** | `data/pedestrian.py` — NOTICE 인터페이스 구현 (`h`, `cost`, `project_physical`, `project_feasible`, `estimate_lipschitz`, `energy`, `progress`, `get_fmbf`) | 저장소 MLP FlowMatch, 38-d 점(8 obs + 12 fut, agent frame) | `COMMAND=points METHOD=all ./run_exp_03_pedestrian.sh` → Exp-01과 같은 5방법 비교 |
| B. MoFlow 격리 실험 | `experiments/exp_03_pedestrian/` (04와 같은 격리 방식) | 공개 MoFlow teacher / 자체 cfm | `COMMAND=phase1|study ...` |

A가 공지 2번(`data/`에 도메인 데이터셋 객체)에 해당한다. B는 A의 제약 설계를 결정한 예비 실험이며, 그 결론은 아래 "MoFlow 기반 예비 실험" 절에 있다.

### A. 보행자 제약 설계 (`data/pedestrian.py`)

- 점: 20프레임 궤적을 마지막 관측 프레임 원점, 프레임 6→7 진행 방향 +x로 정렬한 agent frame. 프레임 7은 항상 0이라 제외 → $p\in\mathbb{R}^{38}$
- $h$: `speed` $\max_k \|p_k-p_{k-1}\|/\Delta t - v_{\max}$, `acc` $\max_k \|p_k-2p_{k-1}+p_{k-2}\|/\Delta t^2 - a_{\max}$, `box` $\max|p| - R$. $v_{\max}, a_{\max}$는 train 궤적별 최댓값의 q99.5, $R$은 train 범위 + 0.5 m. train GT의 99.1%가 feasible
- $C$: NOTICE 기본형 $\tfrac12\sum\max(0,h_k)^2$ (프레임별 항, $C^1$)
- $P$: feasible set(볼록)으로의 정확한 유클리드 투영 (ADMM + closed-form seal). 비확장이므로 $L_P=1$, feasible GT에서 $P(X_1)=X_1$
- `project_feasible(p, buffer)`: 반경을 buffer만큼 줄여 투영 → $h\le -\text{buffer}$ 엄밀 보장 (float32 포함)
- `energy`: $\tfrac{w_{\text{tube}}}{2}\mathrm{dist}(p,S_{\text{slack}})^2$ (+ `w_cost`·$C$). 1-Lipschitz 기울기라 GuideFlow RFE 스텝이 안정적이다
- `progress` / `command_bins`: 마지막 미래 점의 진행 방향각 → 좌/직/우 계열 명령 (GuideFlow CFG)
- `get_fmbf`: speed·acc·box 3개 $C^1$ 장벽 (프레임별 항을 soft-min으로 집계, SafeFlow QP 제약 수 3)
- 이웃 이격 제약은 A에 넣지 않았다. 무조건부 점 생성에는 장면 정보가 없고, B에서 확인했듯 이웃 GT를 쓰면 oracle이 된다

## 목적

Exp-01의 2D Swiss roll에서 검토한 Y-Flow 메커니즘을 보행자 궤적 예측 문제에
적용한다. 기존 Swiss-roll 구현과 모델은 수정하지 않으며, MoFlow 기반 보행자
환경과 사전학습된 teacher Flow Matching 모델을 별도 실험 패키지에서 사용한다.

## 프로젝트 대응

| 역할 | 경로 |
| --- | --- |
| 고정 설정 | `configs/exp_03_pedestrian.yaml` (A), `configs/exp_03_pedestrian_moflow.yaml` (B) |
| 고정 데이터 (ETH/UCY) | `datasets/pedestrian/default/` |
| 실행 코드 | `experiments/exp_03_pedestrian/` |
| 실행 진입점 | `run_exp_03_pedestrian.sh` |
| 결과와 보고서 | `runs/exp_03_pedestrian/` |
| 구조·데이터 무결성 테스트 | `test/test_exp_03_pedestrian.py` |

`experiments/exp_03_pedestrian/`은 MoFlow commit
`f1b89b0e80ce95646214923e271e3c280d305cfa`을 기준으로 한 독립 실행 루트다.
해당 디렉터리에서 실행하므로 내부 `models`, `utils`, `experiments` import가
기존 Y-Flow 패키지와 섞이지 않는다.

## 대표 실행

```bash
COMMAND=prepare ./run_exp_03_pedestrian.sh
COMMAND=audit   ./run_exp_03_pedestrian.sh
COMMAND=phase1  ./run_exp_03_pedestrian.sh --subset zara1 --seeds 3
COMMAND=eval    ./run_exp_03_pedestrian.sh --subset zara1 \
  --ckpt_path checkpoints/eth_ucy/moflow/zara1/models/checkpoint_best.pt \
  --rotate --rotate_time_frame 6 --batch_size 1000 --sampling_steps 100 \
  --solver lin_poly --lin_poly_p 5 --lin_poly_long_step 1000
```

경로는 `experiments/exp_03_pedestrian/` 기준이다. 체크포인트 준비는
`experiments/exp_03_pedestrian/PEDESTRIAN_STUDY.md`를 따른다. Y-Flow 루트의
`.venv`는 Exp-01용이므로 MoFlow 의존성을 공유한다고 가정하지 않는다.

## 고정 메커니즘

- pretrained MoFlow teacher checkpoint 재사용, 재학습 없음
- ETH/UCY `original` 버전, leave-one-out 5개 subset, 8 → 12 frame (0.4 s)
- MoFlow 공식 평가 설정: `lin_poly` solver, 100 step, $p=5$, $K=20$
- MoFlow ETH/UCY 입력은 에이전트 단위(`A=1`)이며 이웃 정보와 장면 지도를 사용하지 않는다
- 제약 $h$: 에이전트 단위 속도·가속 한계 (Phase 0의 train q99.5, 궤적별 최댓값 기준)
- 이웃 이격 제약 (`--constraint kin_col`, 기본): 같은 20-frame window의 다른 보행자 GT 미래와 $\|p_k-q_{j,k}\|\ge r_{\rm safe}=0.2$ m. 이웃 GT 미래를 알려진 움직이는 장애물로 쓰는 **계획 설정(oracle)**이며 예측 벤치마크가 아니다
- 이격 제약은 비볼록이므로 후보 궤적 방향의 반평면으로 볼록화한다(04의 local convex corridor 대응). 투영은 이 볼록 내부집합에 대한 정확한 투영이며 전역 최근접 해가 아니다
- $P$ 없음 → $\mu=0$, $C\equiv0$ 경로. terminal 문제는 feasible set으로의 정확한 유클리드 투영(ADMM)
- 마지막 스텝은 $\eta=1$로 치환하고 closed-form forward seal로 $h(x_N)\le0$을 보장
- 비교 method: `PLAIN_FM`, `FINAL_PROJECTION`(마지막 스텝만), `YFLOW`, `YFLOW_NO_REPLACE`(04 최종안 대응), 선택 `YFLOW_D05`

## 단계

| phase | 내용 | 산출물 |
| --- | --- | --- |
| 0 | train GT 운동학 한계값 산출, split별 GT 위반율 | `runs/exp_03_pedestrian/phase0_gt_audit/` |
| 1 | Y-Flow 이식, matched noise로 method 비교 | `runs/exp_03_pedestrian/phase1_yflow/<subset>/` |
| 2 | $t_{on}$ × damping × replace factorial | 미착수 |
| 3 | 한계값을 좁힌 스트레스 테스트 | 미착수 |
| 4 | 이웃 이격 제약 (계획 설정) | `--constraint kin_col` |
| 5 | 본 실험: cfm/moflow × 5 설정 × 5 subset, 사전 고정 게이트 | `runs/exp_03_pedestrian/study/` |

## 사전학습 체크포인트의 한계

공개 teacher는 $\hat x_1$이 $y_t$에 거의 반응하지 않는다. seed 간 출력이 동일하고, 중간 보정 후에도 `terminal_raw_shift_m = 0`이며, `COMMAND=probe`에서 정규화 공간 1 단위(zara1 기준 약 9.5 m) 섭동이 모든 샘플러 스텝(terminal 포함)에서 $\hat x_1$을 5–9 mm만 바꾼다. 이 체크포인트에서는 Y-Flow가 terminal projection과 같다.

원인은 미확정이다. 체크포인트는 `--drop_method emb`로 학습되었으나, 학습 drop 확률이 0에 가까운 작은 $t$에서도 같은 둔감성이 나타나므로 이것만으로는 설명되지 않는다. `--drop_method None` 재학습은 검증 실험이지 확인된 해결책이 아니다.

## 저장소 5방법 보행자 예측 벤치마크 (Phase 3)

`configs/exp_03_pedestrian.yaml`의 `hardflow`/`yflow`/`safeflow`/`uniconflow`/`guideflow` 하이퍼파라미터 그대로, 각 `eval/*.py`의 갱신식을 과거 관측 조건부 예측(MoFlow teacher)에 이식했다 (`experiments/phase3_bench/methods.py`). 제약은 12개 예측 프레임의 속도·가속(관측 앵커 고정), 평가는 보행자 예측 지표다.

| 저장소 연산 | 보행자 예측 대응 |
| --- | --- |
| $h$ | speed / acc 두 계열 (프레임 최댓값) |
| $C$ | 프레임별 $\tfrac12\sum\max(0,h_k)^2$ |
| $P$ | 실행 가능 집합으로의 정확한 투영 (ADMM + seal), $L_P=1$ |
| `project_feasible` | 반경을 buffer만큼 줄인 투영. HardFlow/YFlow PGD 내부는 ADMM 60회 + seal (항상 feasible) |
| FMBF (SafeFlow) | speed / acc 두 개의 soft-min $C^1$ 장벽 |
| energy (GuideFlow RFE) | $E=\tfrac{w_{\text{tube}}}{2}\,\mathrm{dist}(p,S_{\text{slack}})^2$, $\nabla E = w_{\text{tube}}(p-P(p))$. Swiss roll의 $(d-\tau)_+^2$(튜브까지 거리²)에 대응. m/s 단위 힌지는 $\Delta t=0.4$에서 곡률이 $10^2$–$10^3$이라 `eta_max`=0.5에서 발산(NaN)했다 |
| anchors (GuideFlow CVF/CF) | train 미래 궤적 중 feasible한 것의 FPS 256개 (agent frame) |

**등가성 기준 `POSTHOC_PROJ`:** FLOWMATCH 출력을 샘플링이 끝난 뒤 실행 가능 집합에 한 번 투영한 것. 각 방법의 `dist_to_posthoc_m`이 0에 가까우면 그 방법은 "제약 없이 생성 → 사후 투영"과 같다(흐름 중 개입이 결과에 기여하지 않음). 한계를 조인 설정(`--v_max 1.0`)에서 보정량이 커져도 이 거리가 0이면, 기저 모델이 흐름 상태에 반응하지 않아 방법 간 차이가 생길 여지가 없다는 직접 증거다.

**기저 모델:** `cfm`(주, 흐름 상태에 반응하는 plain conditional FM, subset별 학습)과 `moflow`(대조군, 공개 teacher). 전체 실행은 `experiments/phase3_bench/run_all.sh`: subset별 cfm 학습 → 응답성 probe (G0) → 6방법 벤치(cfm 3 seed, `--save_raw`) → moflow 대조 재실행 → `BENCH_REPORT.md` (subset별 표 + 장면 단위 쌍 차이의 블록 bootstrap 95% CI).

차이점: 샘플러는 저장소 프로토콜대로 균일 Euler `sample.n_steps`=100 (MoFlow 공식 lin_poly 대신). GuideFlow는 설정 기본값대로 training-free (CFG 끔). SafeFlow terminal filter는 SLSQP 대신 `project_feasible`. 실행: `COMMAND=bench ./run_exp_03_pedestrian.sh --subset <s>`, 표: `COMMAND=bench_report`.

## MoFlow 기반 예비 실험 (갈래 B)

## 본 실험 (Phase 2): 사전 고정 설계

### 기저 모델

| 모델 | 역할 | 샘플러 |
| --- | --- | --- |
| `cfm` | 주 모델. 독립 노이즈 기반 plain conditional flow matching (`experiments/phase2_cfm/`). 다양성이 flow state에 있어 추론 시 가이던스가 결과를 바꿀 수 있다 | euler 20 step |
| `moflow` | 대조군. 공개 teacher. $\hat x_1$이 $y_t$에 둔감하므로 Y-Flow ≈ terminal projection이 예상된다 | lin_poly 100 step (공식) |

두 모델 모두 SocialGAN split, frame 6 회전, train split min-max 정규화, K=20으로 같은 파이프라인·지표를 쓴다. `cfm`은 test를 모델 선택에 쓰지 않는다(마지막 epoch EMA).

### 설정

| 설정 | 제약 | 주장 범위 |
| --- | --- | --- |
| `kin_data` | 속도·가속, train q99.5 | 예측 (oracle 없음) |
| `kin_v12`, `kin_v10` | $v_{\max}$ = 1.2 / 1.0 m/s, $a_{\max}$는 데이터 값. 느린 이동 주체 스트레스 | 예측. 정확도는 GT가 제약을 만족하는 장면(GT-feasible)에서 평가하며, 이는 이동 주체 유형을 안다는 조건부 평가다 |
| `colcv_r020` | 속도·가속 + 이웃의 **관측 과거만으로 등속 외삽한** 미래와 0.2 m 이격 | 예측 (oracle 없음). 사회적 충돌 회피. COL은 이웃 GT로 평가 |
| `col_r020`, `col_r035` | 속도·가속 + 이웃 GT 미래와 이격 | 계획 설정 (oracle) |

방법: `PLAIN_FM`, `FINAL_PROJECTION`, `YFLOW`, `YFLOW_NO_REPLACE`, `POV_ALWAYS`(cfm만). seed: cfm 5, moflow 1. 5개 subset 전체 test.

### 지표 (보행자 예측 표준)

| 지표 | 정의 | 출처 관례 |
| --- | --- | --- |
| minADE / minFDE (K=20) | 20개 샘플 중 최선의 평균·최종 변위 오차 (m) | SocialGAN 이후 ETH/UCY 표준, MoFlow·MID·LED 동일 |
| avgADE / avgFDE | 20개 샘플 평균 오차 | 샘플 전체 품질 |
| KDE-NLL | 프레임별 K 샘플에 Scott 대역폭 Gaussian KDE, GT log-pdf (하한 −20) 평균의 음수 | Trajectron++ 평가 관례. K=20이라 추정 분산이 크다 |
| COL@0.1 / COL@0.2 | 예측이 같은 window 다른 보행자 GT 미래와 같은 프레임에 r 이내로 접근한 샘플 비율. 평가 전용 | Social-NCE 계열 충돌률 |
| APD | 20개 샘플 간 평균 쌍거리 (다양성) | 생성 모델 다양성 지표 |
| viol | 설정의 속도·가속 한계 위반 샘플 비율 | 본 실험 제약 |
| CONST_VEL | 마지막 관측 속도 등속 외삽 (K=1) | 필수 sanity baseline ("What the constant velocity model can teach us") |

예측 설정(`kin_*`)에서 방법은 이웃 정보를 쓰지 않는다. 이웃 GT는 COL 평가에만 쓰인다.

### 판정 기준 (실행 전 고정)

통계: 장면별로 seed 평균 후 쌍 차이, 연속 50 장면 블록 bootstrap 95% CI (2000회).

| 게이트 | 기준 |
| --- | --- |
| G0 응답성 (cfm, subset별) | probe terminal step에서 δ=0.1 섭동 시 $\hat x_1$ 변화 ≥ 1 cm, 그리고 `kin_v10`에서 `YFLOW_NO_REPLACE`의 PLAIN 대비 변화 ≥ 1 cm. 통과한 subset만 H1에 사용 |
| G1 안전 | `FINAL_PROJECTION`/`YFLOW`/`POV_ALWAYS` 모든 실행에서 위반율 ≤ 0.1%, 투영 실패 ≤ 1% |
| H1 정확도 (주) | `kin_v12` 또는 `kin_v10`에서 `YFLOW`(또는 `POV_ALWAYS`) − `FINAL_PROJECTION` minADE(GT-feasible)의 CI 상한 < 0인 subset이 G0 통과 subset 중 3개 이상 |
| H2 비용 | 스트레스 설정 평균 APD(YFLOW)/APD(PLAIN) ≥ 0.95, latency 비 ≤ 3 |
| H3 계획 (보조) | `col_r035`에서 H1과 같은 기준 |
| H4 사회적 예측 (보조) | `colcv_r020`에서 H1과 같은 기준. COL@0.2의 PLAIN 대비 변화는 기술 통계로 보고. MoFlow 대조 실행(zara1, `kin`) 확인 후, 본 실험 전에 추가 |

판정: G0 통과 subset < 3 → INCONCLUSIVE. G1 ∧ H1 ∧ H2 → SUPPORT. G1 ∧ H1 부분(1–2 subset) → LIMITED_SUPPORT. 그 외 → NO_SUPPORT. 실행: `COMMAND=study ./run_exp_03_pedestrian.sh`, 보고서: `runs/exp_03_pedestrian/study/REPORT_STUDY.md`.

## 타당성 점검

결론을 보고하기 전에 모두 통과해야 한다.

| 항목 | 방법 | 상태 |
| --- | --- | --- |
| 파이프라인 재현 | 논문 Table 2(ETH-UCY)는 MID/LED 데이터 버전 기준이라 SocialGAN 버전과 직접 비교할 수 없다. 같은 체크포인트·데이터로 upstream `eval_eth.py`와 러너 `PLAIN_FM`을 비교했다. zara1 전체 test min20 ADE/FDE: upstream(GPU) 0.17041 / 0.29645, 러너(CPU) 0.1704 / 0.2963. 차이 ≤ 2e-4 m로 seed·장치 수치 차이 범위 | 확인 (zara1) |
| 체크포인트 데이터 버전 | zara1: LED 버전은 샘플당 2 agent, 체크포인트 agent embedding은 1행이며 strict load 성공 → SocialGAN 버전으로 학습. eth는 LED도 1 agent라 이 방법으로 구분 불가 | zara1 확인 |
| 체크포인트 y-dropout 설정 | zara1: `emb`, k=20, m=0.5 | 확인 |
| $\hat x_1$의 $y_t$ 민감도 | zara1 샘플러 경로: 모든 $t$에서 1 단위 섭동당 5–9 mm. $t\to1$ 해석 경로(`--probe_t`)는 미확인 | 부분 |
| 이격 평가 대상 | 평가는 전체 이웃, 투영은 도달 가능 이웃만 사용 | 반영 |
| 표본 독립성 | SGAN window가 1 frame씩 겹쳐 장면이 독립이 아니다. 장면 단위 CI는 과소추정이며 최종 보고는 연속 window 블록 bootstrap을 쓴다 | 미구현 |
| seed | 공개 체크포인트는 seed 간 요약 지표가 소수점 4자리까지 같다(샘플 단위로는 mm 수준 차이). seed 반복의 정보량이 거의 없다 | 확인 |
| 결과 범위 | 현재 수치는 zara1 첫 배치(1000/2253)와 eth 전체뿐이다 | 부분 |
| eth GT 운동학 이상 | 원인 미확인. eth는 주 결과와 분리해 보고한다 | 미확인 |
| 주장 범위 | "Y-Flow = terminal projection"은 공개 MoFlow teacher에 한정된다. 도메인 수준 주장은 $y_t$에 반응하는 모델에서만 가능하다 | 원칙 |

## Phase 0 결과 요약

train q99.5 한계에서 GT 위반율은 train 1% 미만이다. eth test는 다른 장면과
운동학적으로 다르며(중앙 속도 1.11 vs 0.57 m/s) GT의 56%가 제약 밖이다. 상세는
`runs/exp_03_pedestrian/phase0_gt_audit/REPORT_PHASE0.md`.
