# train 패키지 (train/)

이 패키지는 Flow Matching 학습 루프와 Exp-02-sub의 직접 지도학습 루프를 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Eval 패키지 설명 문서](../eval/README.md)
*   [Data 패키지 설명 문서](../data/README.md)

---

## 2. 파일 목록 및 요약
* `trainer.py`: 설정에 따라 일반 CFM 또는 CLEVRER 직접 인식 학습기로 라우팅
* `clevrer_recognition.py`: ResNet-34 비디오 인식기, Hungarian slot matching 손실, train/dev 분할, early stopping 및 임계값 보정
* `flow_match.py`: 무제약 linear CFM loss
* `ema.py`: exponential moving average
* `checkpoint.py`: `last.pt` 저장/로드. `model.local_dir`이 있으면 `checkpoints/`로 publish
* `guide_flow.py`: 기본은 training-free. `guidance.enabled` 또는 `rfe_train.rfe_loss`이면 자체 backbone 학습
* `safe_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `unicon_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `hard_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `y_flow.py`: training-free. `runs/{run_name}/flowmatch/last.pt`가 있으면 skip, 없으면 flowmatch 학습
* `clevrer_flow.py`: 이전 CLEVRER conditional-flow 실험을 위한 legacy 구현. Exp-02-sub의 기본 설정과 학습 경로에서는 사용하지 않는다.

---

## 3. 세부 명세

### trainer.py

#### run_train
*   **설명**: `data.name=clevrer_recognition`이면 `method=recognition`을 요구하고 직접 감독 학습기로 라우팅한다. 그 외 데이터는 기존 CFM 루프를 사용한다.

### clevrer_recognition.py

#### run_train_recognition
*   **설명**: train scene을 고정 seed로 90/10 train/dev 분할한다. 가시 프레임만 사용해 위치·속도 통계와 train-only collision positive weight를 계산하고, backbone/head learning rate를 분리한 AdamW로 지도학습한다. 개발 손실 기준 early stopping으로 `best.pt`를 선택하고, dev split에서 objectness·visibility·collision threshold를 보정한다. `best.pt`와 `last.pt`, epoch history는 `runs/{run_name}/recognition/`에 두며, 선택된 best 모델을 `model.local_dir/last.pt`로 게시한다.

실행:

```bash
python scripts/setup_resnet34.py
python main.py recognition --mode train --run_name exp_02_sub_video_recognition --config configs/exp_02_sub_video_recognition.yaml
python main.py recognition --mode eval --run_name exp_02_sub_video_recognition --config configs/exp_02_sub_video_recognition.yaml
```

### flow_match.py

#### ConditionalFlowMatching
*   **설명**: $x_t=(1-t)x_0+t x_1$, $\|v_\theta-u\|^2$. $h,C,P$ 없음. `velocity()`는 Euler가 호출.

### hard_flow.py / y_flow.py / safe_flow.py / unicon_flow.py

#### ensure_flowmatch_ckpt
*   **설명**: `runs/{run_name}/flowmatch/last.pt`가 있으면 그 경로를 반환. 없으면 `model.local_dir/last.pt`. 둘 다 없으면 `run_train(..., method="flowmatch")`.

### guide_flow.py

#### ensure_flowmatch_ckpt
*   **설명**: CFG와 EBM이 모두 꺼져 있을 때 GuideFlow는 training-free다. 원논문의 EBM 결합 학습($\mathcal{L}_{\mathrm{RFE}}$) 대신 제약을 추론 시점에 해석적으로 평가한다. 사유는 `docs/GuideFlow.md`.

#### run_train_guideflow
*   **설명**: GuideFlow 자체 backbone을 학습한다. `guidance.enabled`면 Eq. (12)의 조건 마스킹을, `rfe_train.rfe_loss`이면 Eq. (18)의 에너지 항을 더한다. 생성 종단은 기본적으로 샘플러 격자를 그대로 따라 rollout하며, `rollout_steps`로 저비용 근사로 바꿀 수 있다. 둘은 독립적으로 조합된다. 산출물은 `runs/{run_name}/guideflow/last.pt`.

#### build_conditions
*   **설명**: 학습 점마다 $C_p$(최근접 앵커), $C_d$(나선 구간 one-hot), $C_r$(중심선 진행도)를 만든다.
