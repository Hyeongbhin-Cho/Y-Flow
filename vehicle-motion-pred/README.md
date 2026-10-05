# nusc_yflow — nuScenes vehicle trajectory prediction: Flow Matching baseline → Y-Flow

GuideFlow에서 guidance 경로(CVF/CF/RFE/CFG/EBM/anchor)를 전부 뺀 rectified-flow baseline 위에,
training-free Y-Flow(제약 사영 + Lipschitz gating) 샘플러를 얹고 하이퍼파라미터를 탐색해 최종 평가하는 파이프라인.

## 실행 (VESSL JupyterLab)

```bash
unzip nusc_yflow.zip && cd nusc_yflow
pip install -r requirements.txt
python -m pytest tests -q                      # 22 tests
python run_all.py --smoke --device cpu         # 데이터 없이 파이프라인 점검 (~4분)
```

nuScenes는 로그인이 필요하므로 https://www.nuscenes.org/nuscenes#download 에서 로그인 후
**Metadata**(v1.0-trainval 또는 v1.0-mini)와 **Map expansion pack v1.3**의 직접 링크를 복사한다 (센서 blob 불필요, 합계 ~1GB).

```bash
export NUSCENES_META_URL='https://...'  NUSCENES_MAP_URL='https://...'
python run_all.py --profile mini --trials 3 --steps 2000     # 연결 확인
python run_all.py --profile trainval --trials 40             # 본 실험
```

또는 `run_all.ipynb`를 열어 첫 셀의 파라미터(`PROFILE`, URL 등)를 채우고 순서대로 실행. 노트북과 `run_all.py`는 같은 단계를 수행한다.
`vessl/run.yaml`은 VESSL Run 템플릿(URL은 secret env로).

## 단계

| 단계 | 명령 | 산출물 |
|---|---|---|
| 다운로드 | `tools.download_nuscenes` | `data/nuscenes/{v1.0-*, maps/}` |
| 변환 | `tools.convert_nuscenes` | `datasets/nuscenes_*/{train,train_val,val}.npz` (focal frame, +x heading, lane polyline, 정지 장애물, drivable SDF) |
| GT audit | `vfm.evaluate --method gt`, `tools.audit_gt` | GT의 제약 위반율·percentile → threshold 결정 |
| 학습 | `vfm.train` | `outputs/fm_*/best.pt` (train_val minADE 최소 EMA), `last.pt` |
| baseline 평가 | `vfm.evaluate --method cv|fm` | minADE/minFDE/MR@K, 제약 위반율 |
| Y-Flow 탐색 | `vfm.tune --trials N` | `tune/trials.jsonl`, `tune/best.yaml` (objective = minADE + 5·(1−all_hard_safe) + MR, train_val 일부) |
| 최종 | `vfm.evaluate --config tune/best.yaml --method yflow --ref_pred ...` on `val` | `final_summary.json`, `samples.png`, `mode_lost_rate` |

split 규칙: `train` 학습 → `train_val` 모델 선택·튜닝 → `val` 최종 보고만. `val`로 튜닝하지 말 것.

## 제약 (`vfm/constraints.py`, Y-Flow `BaseConstraint` 규약)

| 이름 | h ≤ 0 | 역할 |
|---|---|---|
| speed / accel / continuity | ‖Δp‖/dt ≤ v_max, ‖Δ²p‖/dt² ≤ a_max, 관측 구간과 첫 스텝 사이 가속도 ≤ a_cont | hard, 볼록. 사영은 POCS + 전진 클램프(정확히 feasible) |
| drivable | drivable-area SDF ≤ margin | hard, 비볼록(근사 사영) |
| static | `vehicle.parked`·barrier·cone 박스 거리 ≥ focal 반폭 + margin | hard, 비볼록(근사 사영) |
| lane | 차선 중심선 거리 ≤ lane_half_width | 물리 연산자 P(warm start) + soft cost, Lipschitz gating |

`configs/nuscenes.yaml`의 threshold는 placeholder. `--method gt` 결과의 `viol_rate_*`가 ~0이 되게 정한 뒤 학습·튜닝할 것.
Feasible한 GT는 사영에서 보존된다(P(X₁)=X₁). ablation: `yflow.mu=0`(HardFlow형), `yflow.delta=1e9`(gating 끔), `constraints.hard=[speed,continuity,accel]`(운동학만).

## 검증된 범위 / 안 된 범위

- 검증: 단위 테스트 22개, synthetic 데이터로 다운로드를 제외한 전 단계(train → tune → final → 시각화)가 노트북·스크립트 양쪽에서 실행됨. 좌표 변환·마스크는 가짜 devkit helper로 검증.
- 미검증: 실제 nuScenes 파일에서의 변환(devkit `PredictHelper`/`NuScenesMap` 실호출), GPU 실행, 실데이터 성능. 첫 실행은 `--profile mini`로.
- 속도: Y-Flow는 CPU에서 씬당 수백 ms. GPU에선 훨씬 빠르지만 `data.limit_tune`, `yflow.max_iter`, `constraints.proj_sweeps_final`로 조절.

## 결과 그림과 주행 애니메이션

```bash
# 정적 그림 (정확도/제약 막대, 트레이드오프, 지평선별 오차, 제약 위반, 비용, 장면 그림)
python -m tools.figures --run outputs/fm_trainval --cache datasets/nuscenes --split val \
       --methods cv fm proj yflow --n 4            # -> outputs/fm_trainval/figs/*.png|pdf, table.txt

# 탑다운 주행 애니메이션 (장면당 영상 1개, 방법별 패널)
python -m tools.animate --run outputs/fm_trainval --cache datasets/nuscenes --split val \
       --methods fm proj yflow --pick offroad_fixed --n 3 --format mp4   # -> outputs/fm_trainval/anim/scene*.mp4
```
- 장면 선택: `offroad_fixed`(FM이 도로를 벗어났고 proj/yflow가 고친 장면), `yflow_vs_proj`(두 방법 차이가 큰 장면), `accuracy_loss`(가이던스가 정확도를 가장 많이 깎은 장면), `fm_offroad`, `fm_worst`, `random`. `--scenes 12 40 77`로 직접 지정도 돼.
- 안전 지표는 GT가 같은 검사를 통과한 장면에서만 센 값(`*_gtok`)이라 지도·라벨 오류가 방법 탓으로 잡히지 않아.
- `--format mp4`는 ffmpeg가 필요해. 없으면 자동으로 gif로 떨어져.
- 기본은 GT에 가장 가까운 샘플 하나만 굵게 그려(`--show all`이면 전부), 그린 샘플이 도로를 벗어나는 동안 빨간 테두리와 OFF-ROAD 경고가 뜨고(`--no_flash`로 끔), 마지막 프레임에 minADE와 이탈률 카드가 떠.
- 기본은 카메라가 자차를 따라가는 주행 시점이야. `--zoom`으로 시야(반경, m), `--no_follow`로 장면 전체 고정, `--axes`로 축 표시, `--no_ghost`로 예측 전체 경로 흐리게 깔기를 끌 수 있어.
- 영상 길이 = (12 x `--sub` + 1) / `--fps` + `--hold` 초야. 기본값(sub 6, fps 12, hold 1)이면 약 7초, 실제 주행 6초를 거의 실시간으로 재생해. 천천히 보려면 `--fps 6`(약 13초), 짧게는 `--fps 24 --sub 4`(약 3초).
- 카메라 영상 위에 올리는 렌더링은 지금 캐시로는 안 돼. 변환을 메타데이터와 맵 확장만으로 했고 센서 데이터는 받지 않았어.

### 비교군 추가와 목표점 제약

임의의 실행 결과를 그림에 끼워 넣을 수 있어.
```bash
# 차선 항을 끈 Y-Flow를 별도 run에 저장
python -m vfm.evaluate --config outputs/fm_trainval/config.yaml --method yflow \
       constraints.w_lane=0.0 eval.save_predictions=true data.eval_split=val run_name=abl_nolane

python -m tools.figures --run outputs/fm_trainval --cache datasets/nuscenes_trainval --split val \
       --methods cv fm yflow \
       --pred    nolane=outputs/abl_nolane/pred_yflow_val_seed0.npy \
       --metrics nolane=outputs/abl_nolane/metrics_yflow_val_seed0.json \
       --label   "nolane=Y-Flow (차선 항 없음)"
```
`--pred/--label`은 `tools.animate`에도 똑같이 써.

목표점·경유점 제약(controllability): GT를 읽는 제약이라 **정확도 주장에는 쓰면 안 되고**, "목표를 얼마나 정확히 부과할 수 있는가"를 재는 용도야.
```bash
python -m vfm.evaluate --config outputs/fm_trainval/config.yaml --method yflow \
       constraints.hard="[speed,continuity,accel,drivable,static,goal]" \
       constraints.goal_radius=1.0 constraints.goal_noise=0.5 \
       eval.save_predictions=true data.eval_split=val run_name=goal_r1
```
- `goal_radius`가 공의 반지름(m)이야. 0이면 등식이야. 크면 부드럽게 끌려와.
- `goal_noise`는 GT 종점에 더하는 잡음의 표준편차야. GT를 그대로 주면 너무 유리해서 넣는 거야.
- `waypoint`를 hard에 넣으면 `waypoint_s`(기본 3초) 지점에도 같은 제약이 걸려.

### 목표점을 스스로 제안하기 (GT 누수 없음)

`constraints.goal_source`로 목표를 어디서 얻을지 고른다.
- `gt` (기본): GT 종점 + 잡음. **controllability 실험 전용**이고 정확도 주장에는 못 쓴다.
- `lane`: 지도 차선 중심선에서 도달 가능한 구간의 점들을 후보로 깔고, 무가이던스 샘플 100개의 종점 분포로 점수를 매겨 NMS로 K개를 고른다 (TNT / DenseTNT 방식, 학습 없이).
- `sample`: 무가이던스 샘플 종점의 k-means 중심 K개 (EigenTrajectory식 앵커의 학습 없는 버전).

`lane`과 `sample`은 미래를 읽지 않으므로 **정확도를 그대로 보고할 수 있다.** 고른 K개 목표는 가설 하나당 하나씩 부과되어, 가설 j가 모드 j에 고정된다.

```bash
CK=outputs/fm_trainval/best.pt; H='[speed,continuity,accel,drivable,static,goal]'
for M in yflow proj; do
python -m vfm.evaluate --config outputs/fm_trainval/config.yaml --method $M --ckpt $CK \
   constraints.hard="$H" constraints.w_lane=0.0 constraints.goal_source=lane \
   constraints.goal_radius=1.0 constraints.anchor_samples=100 \
   eval.k=10 data.eval_split=val data.limit_eval=1000
done
```
관련 설정: `goal_band`(도달 가능 거리 비율, 기본 0.25~1.35), `goal_sigma`(점수 커널 폭 3 m), `goal_nms`(목표 간 최소 거리 4 m), `anchor_samples`.

### 학습된 목표 제안기 (TNT 방식, 본체는 그대로 고정)

FM 본체는 건드리지 않고, 그 장면 메모리 위에 **목표 점수기만** 학습한다. 후보는 차선 중심선 위의 도달 가능 지점이고, 손실은 GT 종점에 가장 가까운 후보에 대한 교차 엔트로피와 그 후보에서 실제 종점까지의 오프셋 회귀다. TNT/DenseTNT의 2단계를 그대로 가져온 것이다.

```bash
# 1) 점수기 학습 (파라미터 약 0.1 M, GPU 한 장으로 30분 내외)
python -m tools.train_goalnet --config outputs/fm_trainval/config.yaml \
       --ckpt outputs/fm_trainval/best.pt --out outputs/fm_trainval/goalnet.pt --steps 4000 --k 6

# 2) 학습된 목표로 평가 (GT를 읽지 않으므로 정확도 그대로 보고 가능)
CK=outputs/fm_trainval/best.pt; H='[speed,continuity,accel,drivable,static,goal]'
for M in proj yflow; do
python -m vfm.evaluate --config outputs/fm_trainval/config.yaml --method $M --ckpt $CK \
   constraints.hard="$H" constraints.w_lane=0.0 constraints.goal_source=learned \
   constraints.goal_net=outputs/fm_trainval/goalnet.pt constraints.goal_radius=1.0 \
   eval.k=10 data.eval_split=val data.limit_eval=1000
done
```
`goal_source`: `gt`(오라클 상한) · `learned`(위) · `lane`(점수기 없이 밀도만) · `sample`(샘플 k-means 앵커).
학습 로그의 `goal<2m`와 median이 제안 품질이다. median이 3 m를 넘으면 목표를 고정해봐야 손해다.
