# 다중선형회귀를 통한 강원도 산불 크기 예측 with tf1.x

## 1. 프로그램 개요

### 강원도 지역의 20년 기상 데이터를 수집하여, 산불 발생과 기상인자의 관계에 대해 분석하고 산불 발생에 영향을 미치는 기상 조건을 규명하한다. 이를 바탕으로 선형 회귀 학습을 통해 강원도 산불 발생 시, 그 크기를 예측한다.
기상 인자로는 캐나다 산불 예보 시스템의 구성 요소 중 하나인 FWI*(Forest Fire Weather Index) 지수를 구성하는 강수량, 상대습도, 온도, 풍속을 선택하였다.(VanWagner, 1987)



## 2. 고찰
### 선형 회귀분석을 이용하여 결괏값을 예측했으나, 굉장히 상이한 결과를 얻었다. 산불과 같은 자연재해는 예측하는 것이 쉽지 않고 특히 선형 회귀분석은 산불 		피해와 같은 비선형적인 데이터를 학습하고 예측하는 데는 맞지 않는 것으로 나타났다. 따라서 논문에서 사용하는 로지스틱 회귀분석이나 RNN과 같은 방법을 		이용하는 것이 적합한 것으로 사료된다.


## 3. 개발 기간
### 2019.03 ~ 2019.06 



## 4. 참고문헌
### 기상 데이터를 이용한 산불 피해 규모의 예측 방법 연구(안승환, 2016)
### 봄철과 가을철의 기상에 의한 전국 통합 산불발생 확률 모형 개발(원명수 등 2명, 2018) 
### 기상특성을 이용한 전국 산불발생 확률 모형 개발(이시영 등 4명, 2004)
