# Exp-04. Robot-arm Y-Flow / POV

## 목적

Exp-01의 2D Swiss roll에서 검토한 Y-Flow 메커니즘을 7-DoF 로봇암 궤적
생성 문제에 적용한다. 기존 Swiss-roll 구현과 모델은 수정하지 않으며,
SafeFlowMPC 기반 로봇 환경과 사전학습된 unsafe Flow Matching 모델을 별도
실험 패키지에서 사용한다.

## 프로젝트 대응

| 역할 | 경로 |
| --- | --- |
| 고정 설정 | `configs/exp_04_robot_arm.yaml` |
| 고정 로봇 태스크 | `datasets/robot_arm/default/` |
| 실행 코드 | `experiments/exp_04_robot_arm/` |
| 실행 진입점 | `run_exp_04_robot_arm.sh` |
| 결과와 보고서 | `runs/exp_04_robot_arm/` |
| 구조·CPU 무결성 테스트 | `test/test_exp_04_robot_arm.py` |

`experiments/exp_04_robot_arm/`은 SafeFlowMPC commit
`3efe4d9522f4112b291a868866e7e4934697a261`을 기준으로 한 독립 실행 루트다.
해당 디렉터리에서 실행하므로 내부 `safe_flow_mpc`와 `experiments` import가
기존 Y-Flow 패키지와 섞이지 않는다.

## 대표 실행

```bash
COMMAND=phase1 ./run_exp_04_robot_arm.sh --seeds 32
COMMAND=phase2 ./run_exp_04_robot_arm.sh
COMMAND=phase15 ./run_exp_04_robot_arm.sh
COMMAND=hybrid ./run_exp_04_robot_arm.sh
COMMAND=factorial ./run_exp_04_robot_arm.sh
COMMAND=ood ./run_exp_04_robot_arm.sh
```

기본 `COMMAND`는 `ood`다. Acados v0.5.1과 robot-arm 전용 Python 의존성은
`experiments/exp_04_robot_arm/experiments/pov_projection/SETUP.md`를 따른다.
Y-Flow 루트의 `.venv`는 Exp-01용이므로 로봇 실험 의존성을 공유한다고
가정하지 않는다.

## 고정 메커니즘

- pretrained unsafe FM checkpoint 재사용, 재학습 없음
- 7-step Flow grid
- `t_on=0.5`, `lambda_oc=10`, `mu=0`
- 권장안 `L05_NO_REPLACE`: `lambda=0.5`, terminal replacement 없음
- Acados의 candidate-dependent local weighted projection surrogate 사용

여기서 projection은 전역 최근접 안전 궤적을 보장하지 않는다. 후보 궤적
주변의 local convex corridor와 SQP-RTI로 얻은 방향을 감쇠 보정으로 사용한다.

## 최종 결과 범위

최종 OOD 평가는 100개 결정적 장애물 배치 × 8개 matched seed × 4개 방법,
총 3,200개 궤적이다. `L05_NO_REPLACE`는 Plain FM과 terminal replacement보다
전체 충돌률이 낮았지만 enlarged-obstacle 범주에서 50% 충돌이 남아 최종 판정은
`LIMITED_SUPPORT`다. 상세 수치와 해석은
`runs/exp_04_robot_arm/ood_no_replace/REPORT_OOD.md`에 보존한다.
