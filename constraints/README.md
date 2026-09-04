# constraints 패키지 (constraints/)

이 패키지는 hard constraint $h\le 0$, 비용 $C$, 물리 연산자 $P$를 정의한다. 학습 loss에는 넣지 않고 eval·가이던스에서만 쓴다.

---

## 1. 관련 README 링크
*   [Y-Flow 패키지 설명 문서](../README.md)
*   [Data](../data/README.md)

---

## 2. 파일 목록 및 요약
* `base.py`: 인터페이스 자리 (헤더만)
* `swiss_roll.py`: 튜브 / 코어 / 박스 $h$, $C=d_{\mathcal{M}}^2$, $P=\Pi_{\mathcal{M}}$
* `fmbf.py`: FMBF gain과 단일/복합 최소 보정 QP
* `swiss_roll_fmbf.py`: 논문 부호 $h\geq0$의 smooth Swiss-roll barrier와 terminal filter

---

## 3. 세부 명세

### swiss_roll.py

#### SwissRollConstraint
*   **설명**: dump `meta`를 받는다. 입력은 역정규화한 원좌표 $p\in\mathbb{R}^2$.

#### h
*   **설명**: `tube` $d_{\mathcal{M}}-\tau$, `core` $\rho_{\min}-r$, `box` $\|p\|_\infty-R$. 성립은 $h_j\le 0$.

#### cost / project
*   **설명**: $C(p)=d_{\mathcal{M}}(p)^2$, $P(p)$는 나선 최근접 투영. HardFlow terminal 최적화에 사용.

### fmbf.py / swiss_roll_fmbf.py

*   **설명**: 원래 제약의 safe set 안에 드는 smooth `tube`/`core`/`outer`/`box` barrier를 만든다. composite slack-QP는 모든 active set을 PyTorch 배치 연산으로 풀고, 마지막 최소거리 안전 문제만 SciPy SLSQP를 사용한다. 어느 solver도 실패 시 대체 해를 반환하지 않는다.
