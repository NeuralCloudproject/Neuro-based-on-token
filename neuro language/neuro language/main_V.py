import cupy as cp
import string
import requests
import numpy as np

class NeuronLayer:
    """뉴런 레이어 (이온 농도, 이온 펌프, 이온 채널 포함)"""
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.potentials = cp.full((num_neurons,), -70.0)  # 막전위
        self.thresholds = cp.full((num_neurons,), -55.0)  # 발화 임계값
        self.absolute_refractory_period = cp.zeros(num_neurons)  # 불응기

        # 🔹 이온 농도 (내부 & 외부)
        self.na_in = cp.full((num_neurons,), 15.0)
        self.na_out = cp.full((num_neurons,), 145.0)
        self.k_in = cp.full((num_neurons,), 140.0)
        self.k_out = cp.full((num_neurons,), 4.0)
        self.ca_in = cp.full((num_neurons,), 0.0001)
        self.ca_out = cp.full((num_neurons,), 2.0)
        self.cl_in = cp.full((num_neurons,), 10.0)
        self.cl_out = cp.full((num_neurons,), 110.0)
        self.h_concentration = cp.full((num_neurons,), 7.2)  # pH

        # 🔹 이온 채널 (Na⁺, K⁺, Ca²⁺, Cl⁻, H⁺)
        self.na_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.k_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.ca_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.cl_channel_open = cp.zeros(num_neurons, dtype=bool)
        self.h_channel_open = cp.zeros(num_neurons, dtype=bool)

    def update_potential(self, time_step=0.1):
        """막전위 업데이트"""
        self.potentials[self.na_channel_open] += 10.0 * time_step
        self.potentials[self.k_channel_open] -= 10.0 * time_step
        self.potentials[self.ca_channel_open] += 2.0 * time_step
        self.potentials[self.cl_channel_open] -= 5.0 * time_step
        self.potentials[self.h_channel_open] += 0.5 * time_step  # H⁺ 통로 개방 시 막전위 상승

        self.potentials = cp.clip(self.potentials, -80.0, 50.0)

    def update_ion_pumps(self):
        """이온 펌프 (Na⁺/K⁺, Ca²⁺, H⁺, 독성 제거)"""
        pump_activity = 0.1  
        self.na_in += pump_activity * (self.na_out - self.na_in) * 0.1
        self.k_in -= pump_activity * (self.k_out - self.k_in) * 0.1
        self.ca_in -= 0.05 * self.ca_in
        self.h_concentration -= 0.02 * (self.h_concentration - 7.2)

    def fire(self):
        """뉴런 발화"""
        fired = self.potentials >= self.thresholds
        self.potentials[fired] = -70.0  # 발화 후 초기화
        return fired

    def update(self, time_step=0.1):
        """전체 뉴런 상태 업데이트"""
        self.update_potential(time_step)
        self.update_ion_pumps()


class TokenLayer:
    """입력층 (문자 토큰화 및 뉴런 변환)"""
    def __init__(self):
        self.alphabet = list(string.ascii_uppercase) + list(string.digits) + ['_', '.', ',', '!', '?']
        self.num_tokens = len(self.alphabet)
        self.neurons = NeuronLayer(self.num_tokens)

    def tokenize(self, text):
        """텍스트를 문자 토큰으로 변환"""
        return [char.upper() if char.upper() in self.alphabet else '_' for char in text]

    def encode(self, tokens):
        """토큰을 뉴런 인덱스로 변환"""
        return [self.alphabet.index(token) for token in tokens]

    def decode(self, indices):
        """뉴런 인덱스를 문자로 변환"""
        return ''.join([self.alphabet[i] for i in indices])


class Synapse:
    """뉴런 간 시냅스 연결"""
    def __init__(self, pre_layer, post_layer):
        self.pre_layer = pre_layer
        self.post_layer = post_layer
        self.weights = cp.random.uniform(0.1, 1.0, size=(pre_layer.num_neurons, post_layer.num_neurons))

    def propagate(self):
        """신호 전달"""
        signals = cp.dot(self.pre_layer.potentials, self.weights)
        self.post_layer.potentials += signals


class NLPModel:
    """자연어 처리 모델 (입력층 -> 연산층(다중 레이어) -> 출력층)"""
    def __init__(self, num_layers=3):
        self.token_layer = TokenLayer()
        self.processing_layers = [NeuronLayer(self.token_layer.num_tokens) for _ in range(num_layers)]
        self.output_layer = NeuronLayer(self.token_layer.num_tokens)

        # 시냅스 연결
        self.synapses = [Synapse(self.token_layer.neurons, self.processing_layers[0])]
        for i in range(num_layers - 1):
            self.synapses.append(Synapse(self.processing_layers[i], self.processing_layers[i + 1]))
        self.synapses.append(Synapse(self.processing_layers[-1], self.output_layer))

    def train(self, texts, epochs=5000, batch_size=32):
        """대량 데이터 학습 (GPT-2와 유사한 성능을 목표로)"""
        training_sentences = [self.token_layer.tokenize(text) for text in texts]
        training_indices = [self.token_layer.encode(sentence) for sentence in training_sentences]

        # ✅ 1. 최대 문장 길이 계산
        max_length = max(len(indices) for indices in training_indices)

        # ✅ 2. 패딩 적용 (짧은 문장은 0으로 채움)
        padded_training_indices = []
        for indices in training_indices:
            padded_indices = indices + [0] * (max_length - len(indices))  # 고정된 길이로 맞추기
            padded_training_indices.append(padded_indices)

        # ✅ 3. NumPy 배열로 변환 후 Cupy 배열로 변환
        padded_training_indices = cp.array(np.array(padded_training_indices, dtype=np.int32))

        for epoch in range(epochs):
            batch_losses = []

            # ✅ 4. 배치 샘플링 (올바른 방식으로 변경)
            batch_indices = cp.random.choice(len(padded_training_indices), size=batch_size, replace=False)
            batch_sentences = padded_training_indices[batch_indices]

            for indices in batch_sentences:
                for i in range(len(indices) - 1):
                    current_idx = int(indices[i])  # 🚀 정수 변환 (Cupy 오류 방지)
                    next_idx = int(indices[i + 1])

                    # 현재 문자 뉴런 활성화
                    self.token_layer.neurons.potentials[current_idx] = 50.0

                    # 연산층 신호 전달
                    for synapse in self.synapses:
                        synapse.propagate()

                    # 출력층 활성화 및 발화 여부 확인
                    fired = self.output_layer.fire()

                    # 가중치 업데이트 (발화한 뉴런만 학습)
                    if fired[next_idx]:
                        for synapse in self.synapses:
                            synapse.weights[current_idx, next_idx] += 0.005  # 더 작은 학습률 적용
                        batch_losses.append(1)  # 성공한 케이스

                    # 뉴런 상태 업데이트 (이온 농도, 이온 펌프 포함)
                    self.token_layer.neurons.update()
                    for layer in self.processing_layers:
                        layer.update()
                    self.output_layer.update()

            # ✅ 5. 학습 정확도 출력
            if epoch % 100 == 0:
                avg_loss = sum(batch_losses) / max(len(batch_losses), 1)  # 0으로 나누는 것 방지
                print(f"Epoch {epoch+1}/{epochs} - 평균 학습 정확도: {avg_loss:.4f}")
    
    def predict(self, text, length=10):
        """다음 문자 예측"""
        tokens = self.token_layer.tokenize(text)
        indices = self.token_layer.encode(tokens)
        output_text = tokens

        for _ in range(length):
            current_idx = indices[-1]
            self.token_layer.neurons.potentials[current_idx] = 50.0

            # 연산층 신호 전달
            for synapse in self.synapses:
                synapse.propagate()

            # 출력층에서 가장 강한 신호 찾기
            next_idx = int(cp.argmax(self.output_layer.potentials))  # 🚀 정수형으로 변환

            output_text.append(self.token_layer.alphabet[next_idx])
            indices.append(next_idx)

            # 뉴런 초기화
            self.token_layer.neurons.update()
            for layer in self.processing_layers:
                layer.update()
            self.output_layer.update()

        return self.token_layer.decode(indices)
    
    def compute_perplexity(self, text):
        """Perplexity 계산 (GPT-2 성능 평가 방식)"""
        tokens = self.token_layer.tokenize(text)
        indices = self.token_layer.encode(tokens)
        loss = 0.0

        for i in range(len(indices) - 1):
            current_idx = indices[i]
            next_idx = indices[i + 1]

            prob = cp.exp(self.synapses[-1].weights[current_idx, next_idx])  # 시냅스 가중치를 확률로 변환
            loss += -cp.log(prob + 1e-9)  # 로그 손실 계산 (안정성 위해 1e-9 추가)

        perplexity = cp.exp(loss / len(indices))  # 평균 perplexity 계산
        return float(perplexity)

def load_large_text_data():
    """대규모 텍스트 데이터 가져오기 (예: 프로젝트 구텐베르크)"""
    urls = [
        "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
        "https://www.gutenberg.org/files/11/11-0.txt"  # Alice in Wonderland
    ]
    texts = []

    for url in urls:
        response = requests.get(url)
        text_data = response.text
        sentences = text_data.split('.')
        texts.extend([s.strip().upper() for s in sentences if len(s.strip()) > 10])

    return texts

def main():
    """GPT-2 수준의 반복 학습 실행"""
    model = NLPModel(num_layers=6)  # 더 깊은 모델 사용

    # 대규모 데이터 로드
    training_texts = load_large_text_data()
    print(f"훈련 시작: {len(training_texts)}개의 문장 학습")

    model.train(training_texts, epochs=20000, batch_size=128)  # 장기 학습

    # 모델 성능 평가
    test_text = "MACHINE LEARNING IS"
    predicted_text = model.predict(test_text, length=10)
    ppl = model.compute_perplexity(test_text)

    print(f"입력: {test_text}")
    print(f"예측된 텍스트: {predicted_text}")
    print(f"Perplexity: {ppl:.4f}")

if __name__ == "__main__":
    main()
