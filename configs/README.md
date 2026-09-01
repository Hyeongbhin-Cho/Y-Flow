# configs 패키지 (configs/)

이 패키지는 실험 하이퍼파라미터 YAML을 둔다. CLI에서 같은 키를 넘기면 override한다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Docs](../docs/README.md)

---

## 2. 파일 목록 및 요약
* `exp_01_swiss_roll.yaml`: Exp-01 Swiss roll. 데이터 캐시, MLP, CFM, HardFlow/YFlow/SafeFlow/GuideFlow 설정

---

## 3. 세부 명세

### exp_01_swiss_roll.yaml

#### data
*   **설명**: Swiss roll 생성 메타와 `cache_dir`. `regenerate: false`면 dump를 재사용.

#### model / method / train / sample
*   **설명**: MLP, linear CFM, batch/steps/lr, Euler 스텝과 eval 샘플 수.

#### hardflow
*   **설명**: `t_on`, `lambda_oc`, `max_iter`. training-free 샘플링만 사용.

#### safeflow
*   **설명**: `integrator`는 `euler`(기본) 또는 `dopri5`. `t_on=0.5`부터 CFMBF 보정을 적용한다. `phi_schedule`, `phi0`, `phi_gamma`, `slack_weight`가 gain과 relaxed QP를 정하고, `terminal_filter`는 최종 안전 투영의 반복 수와 허용오차를 정한다.

#### guideflow
*   **설명**: `cvf`/`cf`/`rfe`로 모듈을 켜고 끈다. CVF는 `lambda_cvf`, `t_on`. CF는 `cf_mode`(`interp` 기본, `replace`는 Eq. (16) 문자 그대로), `k_c`, `n_anchors`. RFE는 `tau_star`, `eta_max`, `n_refine`, `slack`과 에너지 가중치 `w_tube`/`w_core`/`w_box`/`w_cost`. 셋 다 `false`면 무제약 FlowMatch와 같다.

#### guideflow.guidance
*   **설명**: Classifier-Free Intent and Reward Guidance. `enabled: false`(기본)면 동결 무조건 backbone을 쓰고 Exp-01 비교 프로토콜이 유지된다. `true`면 조건부 backbone을 따로 학습한다. `signal`은 `anchor`($C_p$) 또는 `command`($C_d$), `gamma`는 guidance scale, `p_uncond`는 마스킹 확률, `reward`/`ep`는 RAS. `cond_steps`/`cond_lr`/`cond_batch_size`/`cond_hidden`/`cond_embed_dim`은 GuideFlow 자체 backbone 학습 설정(CFG/EBM 공통).

#### guideflow.rfe_train
*   **설명**: EBM 결합 학습(Eq. 18). `rfe_loss: false`(기본)면 에너지를 추론 시점에만 평가한다. `true`면 $\mathcal{L}_{\mathrm{RFE}}$를 CFM loss에 더해 속도장을 학습한다. `lambda_rfe`는 가중치, `t_min`은 에너지 항을 켜는 시각, `rollout_steps`는 생성 종단 계산 방식(0이면 posterior mean 근사, 양수면 그 수만큼 미분 가능 Euler 적분).

실행 예: `python main.py hardflow --mode eval --run_name exp1 --config configs/exp_01_swiss_roll.yaml`.
