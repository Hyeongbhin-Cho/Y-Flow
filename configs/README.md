# configs 패키지 (configs/)

이 패키지는 실험 하이퍼파라미터 YAML을 둔다. CLI에서 같은 키를 넘기면 override한다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Docs](../docs/README.md)

---

## 2. 파일 목록 및 요약
* `exp_01_swiss_roll.yaml`: Exp-01 Swiss roll. 데이터 캐시, MLP, CFM, HardFlow $t_{\mathrm{on}}$

---

## 3. 세부 명세

### exp_01_swiss_roll.yaml

#### data
*   **설명**: Swiss roll 생성 메타와 `cache_dir`. `regenerate: false`면 dump를 재사용.

#### model / method / train / sample
*   **설명**: MLP, linear CFM, batch/steps/lr, Euler 스텝과 eval 샘플 수.

#### hardflow
*   **설명**: `t_on`, `lambda_oc`, `max_iter`. training-free 샘플링만 사용.

실행 예: `python main.py hardflow --mode eval --run_name exp1 --config configs/exp_01_swiss_roll.yaml`.
