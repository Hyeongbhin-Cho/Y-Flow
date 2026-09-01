# test 패키지 (test/)

유닛 테스트와 짧은 train→eval 파이프라인. GPU 없이 CPU에서 돌린다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)

---

## 2. 파일 목록 및 요약
* `test_data.py`: npy+meta 쌍 저장/로드
* `test_model.py`: MLP shape
* `test_flow_match.py`: CFM loss, Euler
* `test_constraints.py`: 데이터 점 oracle safety
* `test_cli.py`: `--mode`, `--run_name`, yaml override
* `test_pipeline.py`: 짧은 학습 후 `metrics.json`
* `test_hard_flow.py`: flowmatch ckpt skip, HardFlow eval
* `test_guide_flow.py`: CVF 기하, 앵커 feasibility, 에너지 기울기·스케줄, CFG guidance scale, GuideFlow eval
* `test_fmbf.py`: FMBF gain, KKT 닫힌형 해, composite QP, smooth barrier와 solver 실패 처리
* `test_safe_flow.py`: checkpoint 재사용, Euler/FlowMatch 동일성, Euler/Dopri5 안전성·진단 지표, `t_on` ablation 산출물

---

## 3. 세부 명세

프로젝트 루트에서:

```bash
conda activate yflow
python -m unittest discover -s test -v
```

`test_pipeline`은 `runs/unittest_pipeline/`을 만들었다가 지운다.
