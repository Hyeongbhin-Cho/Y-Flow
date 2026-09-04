# utils 패키지 (utils/)

이 패키지는 config, 경로, seed, device 등 공통 유틸을 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [configs](../configs/README.md)

---

## 2. 파일 목록 및 요약
* `config.py`: YAML 로드, CLI 키가 yaml과 같으면 override
* `paths.py`: `runs/{run_name}/{method}`, FlowMatch `last.pt`
* `seed.py`: `seed_everything`
* `device.py`: `cuda` / `cpu`
* `logging.py`: 자리만 있음 (헤더)

---

## 3. 세부 명세

### config.py

#### load_config / apply_overrides
*   **설명**: `--train.steps` 또는 유니크 leaf `--steps`로 yaml을 덮어쓴다.

### paths.py

#### method_dir / flowmatch_ckpt
*   **설명**: 체크포인트는 항상 `runs/{run_name}/flowmatch/last.pt`. HardFlow, YFlow, SafeFlow, UniConFlow, training-free GuideFlow도 이 backbone을 공유한다.
