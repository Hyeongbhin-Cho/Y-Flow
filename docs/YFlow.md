# YFlow

Physical Guidance Flow Matching + HardFlow-style terminal constraint.

타깃을 $X_1$ 근처로 예측한 뒤 **선형 보간**으로 다음 상태로 간다.  
단순 투영 $P(\hat x_1)$ 대신, 예측된 최종 상태에서 $C$와 $h\le 0$을 풀고, $L_P$와 $t$가 안정된 구간에만 제약을 강하게 건다.

---

## 구현 목표

- [ ] 사전학습 $v_t^\theta$로 raw target $\hat x_1^{\mathrm{raw}}=x_t+(1-t)v_t$를 계산한다.
- [ ] 물리 연산자 $P$는 해 자체가 아니라 warm start다. $P$는 1-Lipschitz, $P(X_1)=X_1$을 가정한다.
- [ ] $\hat x_1$에서 $h(\hat x_1)\le 0$과 비용 $C$를 푼 뒤, 현재 상태와 **선형 보간**으로 $x_{t+\Delta t}$를 만든다. HardFlow inverse map은 쓰지 않는다.
- [ ] $\gamma=0$이면 제약을 끈 원본 flow와 같은 궤적이 나온다.
- [ ] $t$와 추정 Lipschitz $\widehat L_P$로 제약 강도를 스케줄한다. 초반 노이즈 구간에서 과도한 투영을 피한다.
- [ ] Exp-01 Swiss roll에서 $P=\Pi_{\mathcal{M}}$, $C=d_{\mathcal{M}}^2$, $h$는 tube / core / box다.
- [ ] HardFlow와 같은 시드·같은 평가 프로토콜로 Safety / MMD / 시간을 비교한다.
- [ ] Default, $n_{\mathrm{eval}}=4000$에서 Safety $\ge 0.99$를 목표로 한다.
