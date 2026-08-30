# FlowMatch

무제약 baseline. Swiss roll 좌표점을 학습하는 표준 Conditional Flow Matching이다.  
생성 중에 $h$, $C$, $P$를 쓰지 않는다. 제약 다섯 방법의 pretrained backbone이기도 하다.

---

## 구현 목표

- [ ] Exp-01과 같은 2D Swiss roll 점, 같은 정규화, 같은 seed로 학습한다.
- [ ] 제약 없이 속도장 $v_t^\theta$를 학습한다. (linear CFM)
- [ ] 같은 Euler $N$으로 $N_{\mathrm{eval}}$개 점을 샘플링한다.
- [ ] 생성 결과는 역정규화한 뒤, 제약 방법과 **같은** $h$로 Safety / Tube·Core 위반 / MMD / Radius MAE를 측정한다.
- [ ] Scatter가 나선을 따르는지 확인한다. Safety는 제약 방법보다 낮을 것으로 본다.
