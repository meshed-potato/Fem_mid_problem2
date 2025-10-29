# Fem_mid_problem2

## List of packages
- Q3,Q9_element: elastic problem 
## Requirements

- Python 3.10.x
  - NumPy
  - Matplotlib
  - imageio

## Usage
In root directory of the project, run the program with element type 
```sh
python -m stiffness_matrix run -ele Q4
python -m Lagrange_element run -ele Q9

```
## reference
본 프로젝트에서 사용된 수치적분(적분점과 가중치) 방식은 Wikipedia의 “Gaussian quadrature” 문서(링크
)를 참고하여 설계되었습니다. Gaussian quadrature는 다항식 함수의 적분을 높은 정확도로 근사하는 방법으로, 특히 Legendre 다항식의 근을 적분점으로, 대응하는 가중치를 곱하여 계산하는 표준화된 방식을 제공합니다. 본 구현은 이러한 표준 관례를 따름으로써, n차 이하 다항식에 대한 정확한 적분 결과를 보장합니다.

참고: Wikipedia – “Gaussian quadrature” (https://en.wikipedia.org/wiki/Gaussian_quadrature
)
