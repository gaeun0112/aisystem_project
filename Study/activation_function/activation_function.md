

- Sigmoid
    - $a = {1 \over (1+e^{-z})}$
    - 0부터 1까지 나타냄
    - 0 혹은 1을 판별하는 이진 분류를 수행할 때 output layer에 자주 사용된다.
    
    ![Untitled](https://github.com/gaeun0112/aisystem_project/blob/main/image/activate_function/Untitled.png?raw=true)
    
- Tanh
    - $a = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
    - -1부더 1까지 나타냄
    - 거의 항상 시그모이드 함수보다 더 잘 작동
    - 0~1 범위를 나타내는 Sigmoid와 달리 -1~1 범위이기 때문에 데이터의 평균이 0에 가까워져 평균이 0.5에 가까운 시그모이드보다 효율적임 (다음 층의 학습을 더 쉽게 해줌)
    
    ![Untitled](https://github.com/gaeun0112/aisystem_project/blob/main/image/activate_function/Untitled%201.png?raw=true)
    
- Sigmoid와 tanh의 공통적인 단점은 z가 굉장히 크거나 작을 때 미분값이 굉장히 작아진다는 점이다. (기울기가 0에 가까워짐) 이는 경사 하강법의 속도가 느려지는 것을 유발하고, 이를 보완하기 위한 함수가 ReLU다.
- ReLU
    - $a = max(0,z)$
    - z가 양수일 때는 미분값이 1이고, z가 음수이면 미분값이 0이 된다.
    - 엄밀하게 말하자면, z가 0일 때의 도함수는 정의되지 않았지만, 실제로 구현하면 z가 정확히 0이 될 확률은 굉장히 낮기 때문에 z가 0일때 도함수가 1 또는 0이라고 가정해도 잘 작동한다.
    - 보통 은닉층의 활성화 함수로 많이 사용됨.
    - 단점 : z가 음수일 때 도함수가 0이다.
    
    ![Untitled](https://github.com/gaeun0112/aisystem_project/blob/main/image/activate_function/Untitled%202.png?raw=true)
    
- Leaky ReLU
    - $a = max(0.01z, z)$
    - ReLU와 달리, z가 음수일 때도 약간의 기울기를 준다.
    - 실제로는 많이 쓰이지 않지만 ReLU보다 좋은 결과를 보여줌.
    
    ![Untitled](https://github.com/gaeun0112/aisystem_project/blob/main/image/activate_function/Untitled%203.png?raw=true)
    
- ReLU와 Leaky ReLU의 공통적인 장점은, 0보다 큰 활성화 함수의 미분값이 sigmoid와 tanh에 비해 많아서 보다 빠르게 학습할 수 있다. (학습이 느려지는원인인 함수의 기울기가 0에 가까워지는 걸 막아주기 때문)
    - 비록 ReLU에서 z의 절반(음수)에서 미분값이 0이지만, 실제로 은닉층의 z는 0보다 크기 때문에 잘 작동한다.

- 비선형 활성화 함수를 써야 하는 이유
    
    $a^{[1]} = z^{[1]} = W^{[1]}x + b^{[1]}$
    
    $a^{[2]} = z^{[2]} = W^{[2]}x + b^{[2]}$
    
    이러한 두 개의 층에 대한 노드의 계산이 있다고 가정하자. 여기서 $a^{[1]}$, 즉 첫 번째 은닉층의 결과를 두 번째 은닉층의 입력($x$)로 대입한다면 아래와 같이 된다.
    
    $a^{[2]} = z^{[2]} = W^{[2]}(W^{[1]}x + b^{[1]}) + b^{[2]}$
    
    간소화하면 아래와 같다.
    
    $a^{[2]} = (W^{[2]}W^{[1]})x + (W^{[2]}b^{[1]} + b^{[2]})$
    
    두 은닉층의 W와 b를 하나로 뭉쳐서 부르면 다음과 같다.
    
    $a^{[2]} = W'x + b'$
    
    이렇게 선형 함수들만 사용한다면 신경망은 계속해서 선형식만을 출력하게 된다. 이렇게 되면 층을 아무리 많이 쌓아도 층이 하나만 있는 형태보다 얻는 이점이 없기 때문에 선형 활성화 함수는 은닉층에 사용하지 않는 것이다.
