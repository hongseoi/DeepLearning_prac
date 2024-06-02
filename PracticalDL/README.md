# PracticalDL
- Jupyter notebook이 아닌 실제 배포 모델 제작을 위한 templete 구성

## 파일 구조도
![alt text](.\asset\file.png)

## 사용 데이터셋 및 모델 구조
- 데이터셋: MNIST 숫자 데이터셋
![alt text](.\asset\model.png)


## Training Algorithms

1. N개의 입출력 쌍을 모아 데이터셋 구축
$$ D = \{(x_i, y_i)\}^N_{i=1} \\
where \ x \ \in [0,1]^{N \times (28 \times 28)} \ and  \ y \in \{0,1\}^{N \times 10}$$

2. 가중치 파라미터 $\theta$를 갖는 모델 $f_\theta$를 이용해 $f*$를 근사하고자 함

$$\log P_\theta(\centerdot |x_i) = \hat{y}_i = f_\theta(x_i)$$

3. loss function 정의
$$L(\theta) = -\frac{1}{N} \sum_{i=1}^N \log P(y=y_i|x=x_i;\theta) = - \frac{1}{N} \sum_{i=1}^N y_i^T \centerdot \hat{y_i}$$

4. loss function 최소화하는 입력 파라미터 탐색작업 수행

## training
```
python train.py --model_fn tmp.pth --gpu_id -1 --batch_size 256 --n_epochs 20 --n_layers 5
```
### Reference
- [Ki's Blog의 Practical Exerciese 챕터](https://kh-kim.github.io/nlp_with_deep_learning_blog/docs/1-15-practical_exercise/03-exercise_briefing/)