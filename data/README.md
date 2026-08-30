# data 패키지 (data/)

이 패키지는 Exp-01 Swiss roll **좌표점**과 메타를 한 세트로 만들고 고정한다.

매 학습마다 점을 다시 뽑지 않는다. training-free 방법이 같은 데이터·같은 $h$를 보도록 `train.npy` / `eval.npy` / `meta.json`을 디스크에 둔다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)

---

## 2. 파일 목록 및 요약
* `swiss_roll.py`: 2D spiral 생성, 캐시 저장/로드, 정규화
* `../datasets/swiss_roll/default/`: Exp-01 기본 dump

---

## 3. 세부 명세

### swiss_roll.py

#### SwissRollMeta
*   **설명**: $a$, $u$ 구간, `tau`, `R`, `mean/std` 등 제약·정규화에 쓰는 메타.

#### build_swiss_roll
*   **설명**: `cfg.data.cache_dir`에 쌍이 있고 `regenerate=false`면 로드. 없으면 생성 후 저장. 정규화는 저장된 mean/std만 사용.

#### save_swiss_roll / load_swiss_roll
*   **설명**: `train.npy`, `eval.npy`, `meta.json` 쌍.

캐시 디렉터리:

```
datasets/swiss_roll/default/
├── train.npy
├── eval.npy
└── meta.json
```
