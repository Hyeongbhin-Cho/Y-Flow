# UniConFlow

Swiss Roll Exp-01에는 UniConFlow의 일반 constrained-generation 식을
training-free 방식으로 적용한다. 공통 FlowMatch 백본은 재학습하지 않는다.

## 논문 대응

- `eval/unicon_flow.py::constraint_values`: tube/core/box를 논문의 부등식
  제약 $h_j(T)\leq 0$으로 구성한다.
- `ptzf_reference`: prescribed-time zeroing function(PTZF), 논문 식 (36).
- `qp_guidance`: $\dot T=v_t^\theta(T)+u_t$와 slack QP, 논문 식 (42)-(51).
- `terminal_refinement`: 논문의 2단계 inference 중 terminal refinement를
  Swiss Roll의 정확한 feasible-set 투영으로 특수화한다.

이 실험에는 로봇의 상태-행동 궤적이나 동역학이 없으므로 kinodynamic equality
constraint와 CEM window refinement는 적용하지 않는다. 대신 세 기하 부등식의
인증과 최종 feasible point 보정만 비교한다.

## 실행

```bash
python main.py uniconflow --mode train --run_name exp_01_swiss_roll
python main.py uniconflow --mode eval --run_name exp_01_swiss_roll
```
